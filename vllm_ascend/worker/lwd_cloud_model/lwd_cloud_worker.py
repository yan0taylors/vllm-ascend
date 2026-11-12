# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD (edge-cloud) worker subclass.

All LWD worker-side logic lives here (moved out of ``worker.py``):
init bring-up of the duplex channels + recv managers, the EDGE_EMBED
dispatch hook, the cloud finished-flush, and the
DOWN send (draining the runner's per-step hidden payload + attaching
``lwd_c2e_meta`` to the ModelRunnerOutput).

Selected via ``parallel_config.worker_cls`` (see platform.py:
``vllm_ascend.worker.lwd_cloud_worker.LwdCloudWorker``) when
``lwd_config`` is set (LWD deployment).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch
from vllm.lwd_logger import lwd_logger
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

# ops must initialize before the model_runner chain, otherwise device_op
# and the ops package circularly import (mirrors worker.py's import order)
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import lwd_get_data_plane
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (  # 【MR2】引入连接类型：早收状态、DOWN 序号都按连接分开
    ConnectionId,
    LwdChannelType,
    LwdTransferRequest,
)
from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase
from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (
    LwdCloudModelRunner,
)
from vllm_ascend.worker.lwd_cloud_model.lwd_host_stage import (
    LwdHostStager,
    LwdHostStageSlotBusy,
)
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import hash_layer_count, hash_payload_numel

if TYPE_CHECKING:
    from typing import Any

    from vllm.distributed.device_communicators.shm_broadcast import Handle
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

    from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import LwdRecvHintFields

# Early-recv thread loop tuning: MQ dequeue timeout, backoff sleep after a
# dequeue error, and the per-iteration sleep that caps CPU while the MQ
# reader window spins.
_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S = 0.0005
_LWD_EARLY_RECV_ERROR_SLEEP_S = 0.01
_LWD_EARLY_RECV_LOOP_SLEEP_S = 0.002
_LWD_EARLY_RECV_MAX_INFLIGHT = 2
_LWD_HOST_STAGE_SLOTS = 2
_LWD_SHUTDOWN_TIMEOUT_S = 5.0  # 【MR2】shutdown 等待早收线程退出的超时


class LwdCloudWorker(LwdWorkerBase):
    """LwdWorkerBase + LWD data-plane wiring (cloud side only).

    Selected by ``platform.py`` only when LWD is enabled on a cloud
    process; the data-plane wiring below runs unconditionally."""

    def init_device(self) -> None:
        super().init_device()
        self._lwd_down_next_seqno: dict[ConnectionId, int] = {}  # 【MR2】每条连接独立的 DOWN seqno 计数器
        self._init_lwd_early_recv_state()
        if self.use_v2_model_runner:
            raise RuntimeError("LWD data plane requires the V1 model runner; use_v2_model_runner is not supported")
        self.model_config.enable_prompt_embeds = True
        self.model_runner = LwdCloudModelRunner(self.vllm_config, self.device, worker=self)
        self._init_lwd_data_plane()
        import torch.distributed as dist

        lwd_config = self.vllm_config.lwd_config
        assert lwd_config is not None
        self._lwd_dp_idx = self.vllm_config.parallel_config.data_parallel_index
        self._lwd_connections = tuple(  # 【MR2】从拓扑取属于本 DP 的所有连接（一条边一条）
            connection.connection_id
            for connection in lwd_config.connections
            if connection.connection_id.dp_idx == self._lwd_dp_idx
        )
        if not self._lwd_connections:  # 【MR2】拓扑没有显式连接时退化为默认单连接，保持 1E1C 兼容
            self._lwd_connections = (ConnectionId(0, lwd_config.instance.id, self._lwd_dp_idx),)
        from vllm.distributed.lwd import LwdParallelGroupsSpec

        global_rank = dist.get_rank()
        spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)  # 【MR2】从拓扑推导每条连接的通道端点
        local = set(self._lwd_connections)
        self._lwd_up_src_ranks = {  # 【MR2】每条本地连接的 UP 源 rank（只有该 rank 负责早收）
            endpoints.connection_id: endpoints.cloud_rank
            for endpoints in spec.channel_endpoints
            if endpoints.connection_id in local
        }
        if set(self._lwd_up_src_ranks) != local:  # 【MR2】本地连接必须都能在拓扑里找到端点，否则装配错误
            raise RuntimeError("[Lwd][cloud-worker] missing channel endpoint for a local connection")
        self._lwd_early_recv_connections = tuple(  # 【MR2】本 rank 真正负责早收的连接集合
            connection for connection in self._lwd_connections if self._lwd_up_src_ranks[connection] == global_rank
        )
        self._lwd_early_recv_endpoint = bool(self._lwd_early_recv_connections)  # 【MR2】本 rank 是否是至少一条连接的端点
        self._lwd_up_hidden_size = self.model_config.get_hidden_size()
        self._lwd_up_num_hash_layers = hash_layer_count(self.model_config.hf_config)
        self._lwd_up_num_experts_per_tok = getattr(self.model_config.hf_config, "num_experts_per_tok", 0)
        if self._lwd_early_recv_endpoint:
            self._lwd_init_lwd_host_stager()
            self._start_lwd_early_recv_thread()
        lwd_logger.info(
            "[Lwd][pre-recv] worker local_rank=%d global_rank=%d dp_idx=%d endpoint=%s hidden=%d",
            self.rank,
            global_rank,
            self._lwd_dp_idx,
            self._lwd_early_recv_endpoint,
            self._lwd_up_hidden_size,
        )
        self._register_lwd_prompt_embeds_provider()

    def _lwd_init_lwd_host_stager(self) -> None:
        """Create endpoint host staging with fixed limits."""
        self._early_recv_max_inflight = _LWD_EARLY_RECV_MAX_INFLIGHT
        max_numel = hash_payload_numel(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            self._lwd_up_hidden_size,
            self._lwd_up_num_hash_layers,
            self._lwd_up_num_experts_per_tok,
        )
        self._lwd_stager = LwdHostStager(device=str(self.device), slots=_LWD_HOST_STAGE_SLOTS, max_numel=max_numel)

    def load_model(self) -> None:
        """Log the real layer split after loading: layer count / first-last
        layer names / pp slicing args, so a half-model suspicion is decided
        at startup (full coverage from layer 0 = normal; halved or nonzero
        start = wrong pp slicing)."""
        super().load_model()
        # [layer-trace] Parity instrumentation (env-gated, zero overhead
        # when off): the cloud install point; edge and centralized share
        # the same code (see lwd_debug.lwd_layer_trace).
        from vllm_ascend.worker.lwd_debug.lwd_layer_trace import (
            install_lwd_layer_trace,
        )

        install_lwd_layer_trace(self.model_runner.get_model())
        model = self.model_runner.get_model()
        backbone = getattr(model, "model", model)
        layers = getattr(backbone, "layers", None) or getattr(backbone, "decoder_layers", None)
        pc = self.vllm_config.parallel_config
        layer_names = [
            n for n, _ in model.named_modules() if n.count("layers.") == 1 and n.endswith(tuple("0123456789"))
        ]
        lwd_logger.info(
            "[Lwd][cloud-model] loaded layers=%s first=%s last=%s "
            "(config pp=%d tp=%d, my rank=%d) — full load should cover every layer from 0",
            len(layers) if layers is not None else "?",
            layer_names[0] if layer_names else "?",
            layer_names[-1] if layer_names else "?",
            pc.pipeline_parallel_size,
            pc.tensor_parallel_size,
            self.rank,
        )

    # ------------------------------------------------------------------ #
    # Engine step wiring                                                  #
    # ------------------------------------------------------------------ #

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # [Lwd][perf] Worker start time (subtract the engine's rpc-enqueue
        # ts = dispatch-to-start latency; spread across ranks = collective
        # alignment cost)
        lwd_logger.info(
            "[Lwd][perf] worker-exec-start rank=%d ts=%.3f",
            self.rank,
            time.monotonic(),
        )
        self._lwd_up_post_recvs(scheduler_output)

        # finished_req_ids releases runner and host-stage request state.
        if scheduler_output.finished_req_ids:
            self._lwd_cloud_flush_finished(scheduler_output.finished_req_ids)

        lwd_get_data_plane().poll_completions()  # lazy keepalive reap
        _t0 = time.monotonic()
        output = super().execute_model(scheduler_output)
        # [Lwd][perf] Per-step exec_model timing on the cloud: total = the
        # whole forward pass (sample/collect are itemized separately in
        # sample_tokens)
        lwd_logger.info(
            "[Lwd][perf] cloud-exec total=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return output

    def _lwd_up_post_recvs(self, scheduler_output: SchedulerOutput) -> None:
        """Apply abort notices carried by normal or abort-only scheduler batches."""
        from vllm.v1.core.sched.output import LwdBatchType

        batch = getattr(scheduler_output, "lwd_batch", None)
        if batch is None or batch.batch_type is not LwdBatchType.LWD_EMBED:
            return
        meta = batch.batch_meta
        if meta is None or self._lwd_stager is None:
            return
        from vllm_ascend.distributed.lwd_data_plane.lwd_types import connection_from  # 【MR2】abort 批次也要解析连接，只处理本 rank 负责的连接

        if connection_from(meta) not in self._lwd_early_recv_connections:  # 【MR2】不是本 rank 的连接直接忽略
            return
        for req_id in meta.aborted_req_ids or ():
            self._lwd_stager.abort_request(req_id)

    # ------------------------------------------------------------------ #
    # Early receive: comm thread (hint -> post recv -> report watermark) #
    # ------------------------------------------------------------------ #

    def _init_lwd_early_recv_state(self) -> None:
        """Initialize queued, submitted, and delivered UP receive state."""
        self._early_recv_max_inflight = _LWD_EARLY_RECV_MAX_INFLIGHT
        self._early_recv_futures: dict[tuple[ConnectionId, LwdChannelType, int], LwdTransferFuture] = {}  # 【MR2】在飞接收按（连接, 通道, seqno）分开
        self._early_recv_pending: dict[tuple[ConnectionId, LwdChannelType, int], dict[str, Any]] = {}  # 【MR2】排队的 hint 也按三元组分开
        self._early_recv_reserved: set[tuple[ConnectionId, LwdChannelType, int]] = set()  # 【MR2】已预留、尚未提交的 key
        self._early_recv_last_delivered: dict[ConnectionId, int] = {}  # 【MR2】每条连接已交付的最大 seqno
        self._early_recv_failed_connections: set[ConnectionId] = set()  # 【MR2】已失败的连接集合
        self._early_recv_current_key: tuple[ConnectionId, LwdChannelType, int] | None = None  # 【MR2】当前正在提交/汇报的 key，异常时用来定位连接
        self._early_recv_rr = 0  # 【MR2】连接间轮转下标
        self._early_recv_lock = threading.Lock()
        self._lwd_stager: LwdHostStager | None = None
        self._irecv_hint_mqs: dict[ConnectionId, Any] = {}  # 【MR2】每条连接一个 hint 队列
        self._irecv_done_mqs: dict[ConnectionId, Any] = {}  # 【MR2】每条连接一个完成上报队列
        self._lwd_early_recv_thread: threading.Thread | None = None
        self._lwd_early_recv_stop = threading.Event()  # 【MR2】早收线程停止事件

    def _lwd_build_up_recv_request(
        self,
        fields: LwdRecvHintFields,
        recv_buffer: Any,
        stage_segments: list[tuple[Any, int, int]],
    ) -> LwdTransferRequest:
        """Build a staged UP receive with the exact hash-payload geometry."""
        return LwdTransferRequest(
            channel=LwdChannelType.UP,
            op="recv",
            dp_idx=fields.connection.dp_idx,  # 【MR2】接收请求带上完整连接坐标
            edge_id=fields.connection.edge_id,  # 【MR2】edge_id 决定使用哪条边的通道
            cloud_id=fields.connection.cloud_id,  # 【MR2】cloud_id 决定本侧的通道端点
            num_elements=hash_payload_numel(
                fields.num_tokens,
                self._lwd_up_hidden_size,
                self._lwd_up_num_hash_layers,
                self._lwd_up_num_experts_per_tok,
            ),
            seqno=fields.seqno,
            aux_num_elements=fields.aux_rows * 3,
            recv_buffer=recv_buffer,
            stage_segments=stage_segments,
        )

    def _start_lwd_early_recv_thread(self) -> None:
        """Rebuild the hint MQ reader, create the done MQ writer, and start
        the early-recv comm thread.

        One log line before and after the thread start: "starting" before
        (proof this code ran) and "up" from inside the thread body after
        setting the device (proof the thread is alive). Only the first
        line = the thread died during device setup (traceback on stderr);
        both lines but no SUBMIT = no hints arrived (hint MQ not attached;
        the log will say attached/none).
        """
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            attach_hint_mq,
            create_report_mq,
        )

        connections = getattr(self, "_lwd_early_recv_connections", ())  # 【MR2】本 rank 负责早收的连接列表
        for connection in connections:  # 【MR2】逐连接创建队列
            self._irecv_hint_mqs[connection] = attach_hint_mq(connection)  # 【MR2】从环境变量 attach 该连接的 hint MQ
            self._irecv_done_mqs[connection] = create_report_mq()  # 【MR2】创建该连接的 done MQ
        # The engine fetches this MQ's handle through
        # lwd_early_recv_done_handle over collective_rpc once the executor
        # is up, so the vllm executor stays free of LWD plumbing.
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQ=%s",
            self.rank,
            "attached"  # 【MR2】日志区分是否至少 attach 到一条 hint 队列
            if any(mq is not None for mq in self._irecv_hint_mqs.values())
            else "none",
        )
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()

    def lwd_early_recv_done_handle(  # 【MR2】返回某条连接的 done MQ handle，供引擎 reader 使用
        self, connection_id: ConnectionId
    ) -> Handle | None:
        """Handle of one connection's done MQ, for the engine reader."""
        done_mq = self._irecv_done_mqs.get(connection_id)  # 【MR2】按连接查 done MQ
        if done_mq is None:
            return None
        return done_mq.export_handle()  # 【MR2】导出 handle

    def _lwd_early_recv_loop(self) -> None:
        """Comm thread body: take hints -> pre-post recvs; poll futures ->
        report completions.

        It only does four featherweight things: ``dequeue`` /
        ``submit_recv`` / ``future.done()`` / ``enqueue`` — it never
        ``wait()``s any future and never touches the model, so it cannot
        conflict with the busy loop's HCCL usage (an irecv on one side of
        a channel plus a cross-thread wait on the other is disallowed by
        HCCL).
        """
        # The current device is thread-local state; the comm thread must
        # set it once itself (or the irecv lands on card 0)
        torch.npu.set_device(self.device)
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread up rank=%d device=%s hint MQ=%s",
            self.rank,
            self.device,
            "attached"  # 【MR2】线程启动日志同样标明是否 attach 到 hint 队列
            if any(mq is not None for mq in self._irecv_hint_mqs.values())
            else "none",
        )
        hint_mqs = {  # 【MR2】只轮询成功 attach 的 hint 队列
            connection: mq
            for connection, mq in self._irecv_hint_mqs.items()
            if mq is not None
        }
        while not self._lwd_early_recv_stop.is_set():  # 【MR2】早收主循环
            for connection, hint_mq in hint_mqs.items():  # 【MR2】逐连接轮询 hint
                if connection in self._early_recv_failed_connections:  # 【MR2】失败的连接不再轮询
                    continue
                try:
                    hint = hint_mq.dequeue(timeout=_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S)  # 【MR2】带超时的非阻塞取，避免忙等某一条连接
                    self.start_early_irecv(hint, connection_id=connection)  # 【MR2】提交接收时把期望连接传下去，校验 hint 归属
                except TimeoutError:
                    continue
                except Exception as exc:  # 【MR2】单连接异常只让这条连接失败
                    self._lwd_fail_early_recv_connection(connection, self._early_recv_error_seqno(connection), exc)
                    time.sleep(_LWD_EARLY_RECV_ERROR_SLEEP_S)  # 【MR2】失败后短暂退避，避免刷错误日志
            try:
                self._report_irecv_completions()
            except Exception as exc:
                key = self._early_recv_current_key  # 【MR2】汇报异常时取当前 key
                if key is None:  # 【MR2】没有当前 key 说明是全局错误，直接抛出
                    raise
                connection = key[0]  # 【MR2】否则只让当前连接失败
                self._lwd_fail_early_recv_connection(connection, self._early_recv_error_seqno(connection), exc)
            if hint_mqs and all(connection in self._early_recv_failed_connections for connection in hint_mqs):  # 【MR2】所有有 hint 的连接都失败后线程退出
                return
            # The ring's read side sched_yield-spins within the
            # just-read window (busy_loop_s=1s), so a 0.5ms timeout alone
            # does not cap the CPU — this sleep does; the cost is ~2ms of
            # added visibility latency on hints and watermarks each.
            time.sleep(_LWD_EARLY_RECV_LOOP_SLEEP_S)

    def _early_recv_error_seqno(self, connection: ConnectionId) -> int:  # 【MR2】推断某条连接出错时的 seqno
        key = self._early_recv_current_key
        if key is not None and key[0] == connection:  # 【MR2】当前 key 属于该连接就用它的 seqno
            return key[2]
        candidates = [  # 【MR2】否则从在飞/预留/排队中找该连接的 key
            key[2]
            for key in (*self._early_recv_futures, *self._early_recv_reserved, *self._early_recv_pending)
            if key[0] == connection
        ]
        return min(candidates) if candidates else -1  # 【MR2】取最小 seqno；找不到返回 -1

    def _lwd_fail_early_recv_connection(self, connection: ConnectionId, seqno: int, exc: Exception) -> None:  # 【MR2】连接级失败清理：只影响这一条连接
        """Clean one failed peer and report it through the engine error ABI."""
        with self._early_recv_lock:  # 【MR2】加锁修改早收状态
            if connection in self._early_recv_failed_connections:  # 【MR2】重复失败只处理一次
                return
            self._early_recv_failed_connections.add(connection)  # 【MR2】标记失败，后续 hint 直接丢弃
            for key in list(self._early_recv_reserved):  # 【MR2】清掉该连接的预留 key
                if key[0] == connection:
                    self._early_recv_reserved.discard(key)
            for key in list(self._early_recv_pending):  # 【MR2】清掉该连接排队的 hint
                if key[0] == connection:
                    self._early_recv_pending.pop(key, None)
            futures = [future for key, future in self._early_recv_futures.items() if key[0] == connection]  # 【MR2】收集该连接的在飞 future
            for key in list(self._early_recv_futures):  # 【MR2】从在飞表移除
                if key[0] == connection:
                    self._early_recv_futures.pop(key, None)
            self._early_recv_last_delivered.pop(connection, None)  # 【MR2】清掉交付水位
            if self._early_recv_current_key is not None and self._early_recv_current_key[0] == connection:  # 【MR2】清掉当前 key
                self._early_recv_current_key = None
        for future in futures:  # 【MR2】锁外释放 device 张量，避免长时间持锁
            try:
                future.release_device_tensor()  # 【MR2】释放接收张量
            except Exception:
                lwd_logger.exception("[Lwd][pre-recv] failed to release receive tensor connection=%s", connection)
        if self._lwd_stager is not None:  # 【MR2】清理该连接的 host 暂存（批次、槽、请求）
            try:
                self._lwd_stager.abandon_connection(connection)  # 【MR2】按连接回滚暂存状态
            except Exception:
                lwd_logger.exception("[Lwd][pre-recv] failed to clean host staging connection=%s", connection)
        try:
            lwd_get_data_plane().abort_connection(connection, exc)  # 【MR2】废弃该连接的数据面队列，不影响其它连接
        except Exception:
            lwd_logger.exception("[Lwd][pre-recv] failed to clean data plane connection=%s", connection)  # 【MR2】数据面清理失败只记日志
        lwd_logger.exception("[Lwd][pre-recv] connection failed connection=%s seqno=%d", connection, seqno)  # 【MR2】记录连接失败信息
        done_mq = self._irecv_done_mqs.get(connection)  # 【MR2】取该连接的 done MQ
        if done_mq is not None:
            try:
                done_mq.enqueue((connection, "error", seqno, repr(exc)))  # 【MR2】把（连接, error, seqno, 异常）上报给引擎
            except Exception:
                lwd_logger.exception(
                    "[Lwd][pre-recv] failed to report comm error connection=%s seqno=%d",
                    connection,
                    seqno,
                )

    def start_early_irecv(  # 【MR2】提交一条 UP 接收，显式带上期望连接
        self,
        hint: dict[str, Any],
        connection_id: ConnectionId,
    ) -> None:
        """Queue one validated UP hint and submit it when capacity allows."""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        fields = hint_fields(hint)
        if (  # 【MR2】hint 必须格式正确、连接匹配、通道是 UP
            fields is None
            or fields.connection != connection_id
            or hint.get("channel") is not LwdChannelType.UP
        ):
            try:  # 【MR2】尽量取出 seqno 用于报错定位
                seqno = int(hint.get("seqno", -1)) if isinstance(hint, dict) else -1
            except (TypeError, ValueError):
                seqno = -1
            self._lwd_fail_early_recv_connection(  # 【MR2】hint 非法时只让这条连接失败
                connection_id,
                seqno,
                ValueError(f"malformed UP receive hint: {hint!r}"),
            )
            return
        key = (fields.connection, LwdChannelType.UP, fields.seqno)  # 【MR2】key 用完整连接编号，避免多边冲突
        with self._early_recv_lock:
            if fields.connection in self._early_recv_failed_connections:  # 【MR2】失败连接的新 hint 直接忽略
                return
            if (
                fields.seqno <= self._early_recv_last_delivered.get(fields.connection, -1)  # 【MR2】已交付或更旧的 seqno 视为重复
                or key in self._early_recv_futures
                or key in self._early_recv_reserved  # 【MR2】已预留/已排队也算重复
                or key in self._early_recv_pending
            ):
                return
            self._early_recv_pending[key] = hint
        self._early_recv_pump()

    def _lwd_submit_early_recv(self, fields: LwdRecvHintFields) -> LwdTransferFuture:
        """Plan host destinations, allocate a device slot, and submit one receive."""
        stager = self._lwd_stager
        assert stager is not None
        numel = hash_payload_numel(
            fields.num_tokens,
            self._lwd_up_hidden_size,
            self._lwd_up_num_hash_layers,
            self._lwd_up_num_experts_per_tok,
        )
        stage_key = (fields.connection, fields.seqno)  # 【MR2】暂存 key = (连接, seqno)
        recv_buffer, _slot = stager.allocate(stage_key, numel)  # 【MR2】按连接分配暂存槽
        segments = stager.plan_recv_segments(
            seqno=stage_key,  # 【MR2】接收分段计划也按连接 key
            numel=numel,
            hidden_size=self._lwd_up_hidden_size,
            req_ids=fields.req_ids,
            num_tokens_per_req=fields.rows,
            prompt_lens=fields.prompt_lens,
            row_base=fields.prompt_offsets,
            num_hash_layers=self._lwd_up_num_hash_layers,
            num_experts_per_tok=self._lwd_up_num_experts_per_tok,
            refills=fields.refills,
            prefix_nonces=fields.prefix_nonces,
        )
        request = self._lwd_build_up_recv_request(fields, recv_buffer, segments)
        return lwd_get_data_plane().submit_recv(request)

    def _early_recv_quota(self, connections: list[ConnectionId]) -> int:  # 【MR2】每条连接的软配额，空闲容量仍可被其它连接使用
        """Soft per-connection share; unused capacity remains work conserving."""
        if not connections:  # 【MR2】没有连接时退回全局上限
            return self._early_recv_max_inflight
        return max(1, (self._early_recv_max_inflight + len(connections) - 1) // len(connections))  # 【MR2】按连接数均分在飞上限，至少为 1

    def _pick_pending_locked(  # 【MR2】在待提交 hint 中挑一个连接，优先满足软配额
        self, blocked: set[ConnectionId]
    ) -> tuple[tuple[ConnectionId, LwdChannelType, int], dict[str, Any]] | None:
        """Pick one connection head fairly, preferring its soft quota."""
        by_connection: dict[ConnectionId, list[tuple[ConnectionId, LwdChannelType, int]]] = {}
        for key in self._early_recv_pending:  # 【MR2】把待提交 hint 按连接分组
            by_connection.setdefault(key[0], []).append(key)
        connections = sorted(connection for connection in by_connection if connection not in blocked)  # 【MR2】跳过本轮已阻塞的连接
        if not connections:
            return None
        quota = self._early_recv_quota(connections)  # 【MR2】当前连接数对应的软配额
        start = self._early_recv_rr % len(connections)  # 【MR2】轮转起点
        ordered = connections[start:] + connections[:start]  # 【MR2】旋转连接顺序，保证公平
        counts: dict[ConnectionId, int] = {connection: 0 for connection in connections}  # 【MR2】统计每条连接的在飞数量
        for connection, _channel, _seqno in (*self._early_recv_futures, *self._early_recv_reserved):  # 【MR2】在飞 + 已预留都算占用
            counts[connection] = counts.get(connection, 0) + 1
        selected = next((connection for connection in ordered if counts.get(connection, 0) < quota), None)  # 【MR2】优先挑未超配额的连接
        if selected is None:
            selected = ordered[0]  # 【MR2】都超配额时退化为第一个
        key = min(by_connection[selected], key=lambda item: item[2])  # 【MR2】同连接内按 seqno 最小提交
        self._early_recv_rr = (connections.index(selected) + 1) % len(connections)  # 【MR2】推进轮转下标
        return key, self._early_recv_pending[key]

    def _early_recv_pump(self) -> None:
        """Reserve under lock, then submit independent FIFOs without cross-connection HOL blocking."""  # 【MR2】先预留再提交，各连接 FIFO 互不阻塞
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        blocked: set[ConnectionId] = set()  # 【MR2】本轮被阻塞的连接
        while True:
            with self._early_recv_lock:
                if len(self._early_recv_futures) + len(self._early_recv_reserved) >= self._early_recv_max_inflight:  # 【MR2】全局在飞数量达到上限就停
                    return
                picked = self._pick_pending_locked(blocked)  # 【MR2】挑下一个可提交的 hint
                if picked is None:
                    return
                key, hint = picked  # 【MR2】取出 key 和 hint
                fields = hint_fields(hint)
                assert fields is not None
                self._early_recv_pending.pop(key)  # 【MR2】从排队表移除
                self._early_recv_reserved.add(key)  # 【MR2】标记为已预留
                self._early_recv_current_key = key  # 【MR2】记录当前正在提交的 key
            try:
                future = self._lwd_submit_early_recv(fields)  # 【MR2】提交接收
            except LwdHostStageSlotBusy:  # 【MR2】暂存槽满：该连接的 hint 放回队列，其它连接继续
                with self._early_recv_lock:
                    self._early_recv_reserved.discard(key)  # 【MR2】撤销预留
                    if fields.connection not in self._early_recv_failed_connections:  # 【MR2】hint 放回排队表（连接未失败时）
                        self._early_recv_pending[key] = hint
                    if self._early_recv_current_key == key:  # 【MR2】清当前 key
                        self._early_recv_current_key = None
                blocked.add(fields.connection)  # 【MR2】只阻塞这条连接，继续处理其它连接
                continue
            except Exception as exc:  # 【MR2】其它异常：判该连接失败
                with self._early_recv_lock:  # 【MR2】撤销预留
                    self._early_recv_reserved.discard(key)
                    if self._early_recv_current_key == key:  # 【MR2】清当前 key
                        self._early_recv_current_key = None
                if self._lwd_stager is not None:  # 【MR2】回滚该 seqno 的暂存分配
                    self._lwd_stager.abandon((fields.connection, fields.seqno))
                self._lwd_fail_early_recv_connection(fields.connection, fields.seqno, exc)  # 【MR2】清理并上报该连接失败
                blocked.add(fields.connection)
                continue

            release_future = False  # 【MR2】提交成功后连接可能已被判失败
            with self._early_recv_lock:
                self._early_recv_reserved.discard(key)  # 【MR2】撤销预留
                if fields.connection in self._early_recv_failed_connections:  # 【MR2】连接已失败：稍后释放 future
                    release_future = True
                else:
                    self._early_recv_futures[key] = future  # 【MR2】正常情况记录在飞 future
                    inflight = len(self._early_recv_futures)  # 【MR2】更新在飞数
                    queued = len(self._early_recv_pending)  # 【MR2】更新排队数
                if self._early_recv_current_key == key:  # 【MR2】清当前 key
                    self._early_recv_current_key = None
            if release_future:
                try:
                    future.release_device_tensor()  # 【MR2】释放刚提交的接收张量
                except Exception:
                    lwd_logger.exception(  # 【MR2】释放失败只记日志
                        "[Lwd][pre-recv] failed to release receive tensor connection=%s", fields.connection
                    )
                blocked.add(fields.connection)  # 【MR2】该连接本轮不再提交
                continue
            lwd_logger.info(
                "[Lwd][pre-recv] SUBMIT connection=%s channel=%s seqno=%d "  # 【MR2】SUBMIT 日志带上连接编号
                "num_tokens=%d aux_rows=%d inflight=%d queued=%d",
                fields.connection,
                key[1].value,
                fields.seqno,
                fields.num_tokens,
                fields.aux_rows,
                inflight,
                queued,
            )

    def _report_irecv_completions(self) -> None:
        """Publish each connection's contiguous completion prefix independently."""  # 【MR2】每条连接独立推进连续的完成前缀
        stager = self._lwd_stager
        if stager is None:  # 【MR2】没有暂存区就没有要汇报的到达行
            return
        with self._early_recv_lock:
            pending = sorted(self._early_recv_futures.items(), key=lambda item: (item[0][0], item[0][2]))  # 【MR2】在飞 future 按（连接, seqno）排序
        blocked: set[ConnectionId] = set()  # 【MR2】本轮已阻塞的连接
        for key, future in pending:
            connection, channel, seqno = key  # 【MR2】拆出连接、通道、seqno
            if connection in blocked or connection in self._early_recv_failed_connections:  # 【MR2】失败或已阻塞的连接跳过
                continue
            done_mq = self._irecv_done_mqs.get(connection)  # 【MR2】按连接取 done MQ
            self._early_recv_current_key = key
            try:
                if done_mq is None:  # 【MR2】没有上报队列时保留在飞项，不回收
                    # No report reader for this connection: leave the future
                    # queued (aligned with the memc baseline; no consume-point
                    # fallback recycling).
                    blocked.add(connection)
                    continue
                if not future.done():  # 【MR2】接收未完成：阻塞该连接，其它连接继续
                    blocked.add(connection)
                    continue
                future.result()  # 【MR2】检查接收是否出错
                if not future.staged_done():  # 【MR2】数据还没落到暂存区就继续等
                    blocked.add(connection)
                    continue
                future.result()  # 【MR2】等待暂存完成
                stage_key = (connection, seqno)  # 【MR2】暂存 key = (连接, seqno)
                published = stager.publish_received(stage_key)  # 【MR2】发布到达行
                stager.release_if_drained(stage_key)  # 【MR2】该批次 host 内存释放完后回收元数据
                arrived = [(req_id, offset + rows) for req_id, rows, offset in published]
                future.release_device_tensor()
                for req_id, nonce in stager.take_refill_done(connection):  # 【MR2】按连接取 refill 完成项
                    done_mq.enqueue(("refill_done", req_id, nonce))  # 【MR2】把 refill_done 发给引擎
                done_mq.enqueue((key, arrived))  # 【MR2】把完成项和水位发给引擎
            except Exception as exc:  # 【MR2】汇报异常
                self._lwd_fail_early_recv_connection(connection, seqno, exc)  # 【MR2】只判当前连接失败
                blocked.add(connection)
                continue
            finally:
                self._early_recv_current_key = None  # 【MR2】清当前 key
            with self._early_recv_lock:
                self._early_recv_futures.pop(key, None)
                self._early_recv_last_delivered[connection] = seqno  # 【MR2】推进该连接的已交付 seqno
            self._early_recv_current_key = None  # 【MR2】清当前 key
            lwd_logger.info(
                "[Lwd][pre-recv] REPORT connection=%s channel=%s seqno=%d arrivals=%s",  # 【MR2】REPORT 日志带上连接
                connection,
                channel.value,
                seqno,
                arrived,
            )
        self._early_recv_pump()  # 【MR2】汇报完继续泵下一批

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN: send the hidden packet (channel-async,
        # in-stream ordered). The executor output thread harvests each  # 【MR2】DOWN 发送：执行器输出线程按连接的 ready event 收割 device 元数据
        # connection's device meta behind its per-step ready event, keeping
        # this worker thread free to prepare and execute subsequent steps.
        payloads = self.model_runner.take_lwd_pending_down_payload()  # 【MR2】取本步所有连接的 DOWN 载荷
        if payloads:
            from vllm.distributed.parallel_state import get_tp_group

            is_driver = get_tp_group().is_first_rank  # 【MR2】只有 TP0（driver）负责把 C2E 元数据回给引擎
            carriers = []  # 【MR2】多连接 carrier 列表
            legacy_carrier = None
            ready_events = {}  # 【MR2】旧单数 carrier（取第一份）
            legacy_ready_event = None  # 【MR2】每连接的 ready event
            for payload in payloads:  # 【MR2】逐连接处理载荷
                connection = payload.connection  # 【MR2】本载荷的连接
                hidden = payload.hidden
                meta_dev = payload.meta_dev
                seg_lens_host = payload.seg_lens_host
                ready_event = payload.ready_event
                req_ids = payload.req_ids
                seqno = self._lwd_next_down_seqno(connection)  # 【MR2】取该连接下一个 DOWN seqno
                if connection in self._lwd_early_recv_connections:  # 【MR2】只有该连接的真实端点 rank 才真正发张量
                    lwd_logger.info(
                        "[Lwd][cloud-worker] DOWN send connection=%s seqno=%d numel=%d",
                        connection,
                        seqno,
                        hidden.numel(),
                    )
                    lwd_get_data_plane().submit_send(  # 【MR2】提交 DOWN 发送
                        LwdTransferRequest(
                            channel=LwdChannelType.DOWN,
                            op="send",
                            num_elements=hidden.numel(),
                            dp_idx=connection.dp_idx,  # 【MR2】发送请求带上完整连接坐标
                            edge_id=connection.edge_id,
                            cloud_id=connection.cloud_id,
                            tensor=hidden,
                            seqno=seqno,
                        )
                    )
                if not is_driver:  # 【MR2】非 TP0 只推进 seqno，不返回元数据
                    # Every rank advances the same seqno, but only TP0 returns
                    # C2E metadata through the executor's single output rank.
                    continue
                # The device meta and its ready event travel with the output;  # 【MR2】carrier 带上连接编号，执行器按连接收割
                # the executor output thread owns the fenced D2H harvest.
                carrier = (meta_dev, seg_lens_host, req_ids, hidden.numel(), seqno, connection)  # 【MR2】carrier = (device 元数据, 分段长度, 请求 id, 元素数, seqno, 连接)
                carriers.append(carrier)  # 【MR2】加入多连接列表
                ready_events[connection] = ready_event  # 【MR2】记录该连接的 ready event
                if legacy_carrier is None:
                    legacy_carrier = carrier[:5]  # 【MR2】第一份同时填旧单数 carrier，兼容旧接口
                    legacy_ready_event = ready_event
                lwd_logger.info(  # 【MR2】每连接性能日志
                    "[Lwd][perf] cloud-step connection=%s seqno=%d sample=%.2f send=%.2f total=%.2fms rows=%d",
                    connection,
                    seqno,
                    (_t_sample - _t0) * 1000,
                    (time.monotonic() - _t_sample) * 1000,
                    (time.monotonic() - _t0) * 1000,
                    hidden.shape[0],
                )
            if is_driver and output is not None:  # 【MR2】TP0 把 carrier 挂到输出上
                target = getattr(output, "_model_runner_output", output)
                target.lwd_down_carriers = carriers  # 【MR2】多连接路径
                target.lwd_down_carrier = legacy_carrier  # 【MR2】旧单数路径
                # The events belong to this output generation and must be
                # consumed before the output crosses the process boundary.
                target.lwd_meta_ready_events = ready_events  # 【MR2】每连接 ready event
                target.lwd_meta_ready_event = legacy_ready_event  # 【MR2】旧单数 ready event
        return output

    # ------------------------------------------------------------------ #
    # LWD housekeeping (control-plane glue)                               #
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:  # 【MR2】退出顺序：先停早收线程，再关数据面，最后原生清理
        """Quiesce early receive before closing its data-plane FIFOs."""
        self._lwd_early_recv_stop.set()  # 【MR2】通知早收线程退出
        thread = self._lwd_early_recv_thread  # 【MR2】取线程句柄
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():  # 【MR2】等待线程退出（带超时）
            thread.join(timeout=_LWD_SHUTDOWN_TIMEOUT_S)
            if thread.is_alive():
                lwd_logger.warning("[Lwd][pre-recv] comm thread did not stop within %.1fs", _LWD_SHUTDOWN_TIMEOUT_S)  # 【MR2】超时未退出只告警，不强制杀线程
        try:
            lwd_get_data_plane().shutdown(timeout=_LWD_SHUTDOWN_TIMEOUT_S)  # 【MR2】关闭数据面，拒绝新提交
            if thread is not None and thread.is_alive():  # 【MR2】线程还活着时不能销毁通道
                # A blocked HCCL call may outlive this worker. The data plane is
                # permanently closed against new submissions, but its channels
                # must remain valid until the process exits and reaps the thread.
                lwd_logger.warning(  # 【MR2】HCCL 调用可能仍阻塞在线程里，通道必须保持有效
                    "[Lwd][pre-recv] skip channel destruction while comm thread is alive"
                )
            else:
                from vllm_ascend.distributed.lwd_data_plane.lwd_channels import (
                    lwd_destroy_channels,
                )

                lwd_destroy_channels()  # 【MR2】线程已退出才销毁通道
        finally:
            super().shutdown()  # 【MR2】无论前面是否异常都执行基类 shutdown

    def _lwd_cloud_flush_finished(self, finished_req_ids: list[str]) -> None:
        """Release runner bookkeeping and all host-stage request storage."""
        lwd_logger.info("[Lwd][cloud-worker] flush finished reqs=%s", list(finished_req_ids))
        self.model_runner.lwd_release_finished(finished_req_ids)
        if self._lwd_stager is None:
            return
        for req_id in finished_req_ids:
            seqnos = self._lwd_stager.batches_holding(req_id)
            self._lwd_stager.release_request(req_id)
            for seqno in seqnos:
                self._lwd_stager.release_if_drained(seqno)

    def _register_lwd_prompt_embeds_provider(self) -> None:
        """Wire the draft proposer's first-pass prompt-embeds provider.

        The provider resolves the request's prompt embeds from the
        runner's ``input_batch.req_prompt_embeds`` (already injected by
        ``_lwd_inject_remote_embeds`` during prefill); a missing entry
        (not scheduled / not LWD) returns None and the proposer falls
        back to the token-id path.
        """
        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        runner = self.model_runner

        def _lwd_prompt_embeds_provider(req_id: str) -> torch.Tensor | None:
            idx = runner.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                return None
            return runner.input_batch.req_prompt_embeds.get(idx)

        AscendSpecDecodeBaseProposer.set_lwd_prompt_embeds_provider(_lwd_prompt_embeds_provider)

    def _lwd_next_down_seqno(self, connection: ConnectionId | int = ConnectionId(0, 0, 0)) -> int:  # 【MR2】每条连接独立的 DOWN seqno 分配
        if isinstance(connection, int):  # 【MR2】兼容旧 int 调用
            connection = ConnectionId(0, 0, connection)
        seqno = self._lwd_down_next_seqno.get(connection, 0)
        self._lwd_down_next_seqno[connection] = seqno + 1  # 【MR2】自增该连接的 seqno
        return seqno  # 【MR2】返回分配到的 seqno
