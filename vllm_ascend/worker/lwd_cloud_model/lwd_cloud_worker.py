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
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
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
_LWD_SHUTDOWN_TIMEOUT_S = 5.0


class LwdCloudWorker(LwdWorkerBase):
    """LwdWorkerBase + LWD data-plane wiring (cloud side only).

    Selected by ``platform.py`` only when LWD is enabled on a cloud
    process; the data-plane wiring below runs unconditionally."""

    def init_device(self) -> None:
        super().init_device()
        self._lwd_down_next_seqno: dict[ConnectionId, int] = {}
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
        self._lwd_connections = tuple(
            connection.connection_id
            for connection in lwd_config.connections
            if connection.connection_id.dp_idx == self._lwd_dp_idx
        )
        if not self._lwd_connections:
            self._lwd_connections = (ConnectionId(0, lwd_config.instance.id, self._lwd_dp_idx),)
        from vllm.distributed.lwd import LwdParallelGroupsSpec

        global_rank = dist.get_rank()
        spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
        local = set(self._lwd_connections)
        self._lwd_up_src_ranks = {
            endpoints.connection_id: endpoints.cloud_rank
            for endpoints in spec.channel_endpoints
            if endpoints.connection_id in local
        }
        if set(self._lwd_up_src_ranks) != local:
            raise RuntimeError("[Lwd][cloud-worker] missing channel endpoint for a local connection")
        self._lwd_early_recv_connections = tuple(
            connection for connection in self._lwd_connections if self._lwd_up_src_ranks[connection] == global_rank
        )
        self._lwd_early_recv_endpoint = bool(self._lwd_early_recv_connections)
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
        from vllm_ascend.distributed.lwd_data_plane.lwd_types import connection_from

        if connection_from(meta) not in self._lwd_early_recv_connections:
            return
        for req_id in meta.aborted_req_ids or ():
            self._lwd_stager.abort_request(req_id)

    # ------------------------------------------------------------------ #
    # Early receive: comm thread (hint -> post recv -> report watermark) #
    # ------------------------------------------------------------------ #

    def _init_lwd_early_recv_state(self) -> None:
        """Initialize queued, submitted, and delivered UP receive state."""
        self._early_recv_max_inflight = _LWD_EARLY_RECV_MAX_INFLIGHT
        self._early_recv_futures: dict[tuple[ConnectionId, LwdChannelType, int], LwdTransferFuture] = {}
        self._early_recv_pending: dict[tuple[ConnectionId, LwdChannelType, int], dict[str, Any]] = {}
        self._early_recv_reserved: set[tuple[ConnectionId, LwdChannelType, int]] = set()
        self._early_recv_last_delivered: dict[ConnectionId, int] = {}
        self._early_recv_failed_connections: set[ConnectionId] = set()
        self._early_recv_current_key: tuple[ConnectionId, LwdChannelType, int] | None = None
        self._early_recv_rr = 0
        self._early_recv_lock = threading.Lock()
        self._lwd_stager: LwdHostStager | None = None
        self._irecv_hint_mq = None
        self._irecv_done_mq = None
        self._irecv_hint_mqs: dict[ConnectionId, Any] = {}
        self._irecv_done_mqs: dict[ConnectionId, Any] = {}
        self._lwd_early_recv_thread: threading.Thread | None = None
        self._lwd_early_recv_stop = threading.Event()

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
            dp_idx=fields.connection.dp_idx,
            edge_id=fields.connection.edge_id,
            cloud_id=fields.connection.cloud_id,
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

        connections = getattr(self, "_lwd_early_recv_connections", ())
        for connection in connections:
            self._irecv_hint_mqs[connection] = attach_hint_mq(connection)
            self._irecv_done_mqs[connection] = create_report_mq(connection)
        default = connections[0]
        self._irecv_hint_mq = self._irecv_hint_mqs[default]
        self._irecv_done_mq = self._irecv_done_mqs[default]
        # The engine fetches this MQ's handle through
        # lwd_early_recv_done_handle over collective_rpc once the executor
        # is up, so the vllm executor stays free of LWD plumbing.
        lwd_logger.info(
            "[Lwd][pre-recv] comm thread starting rank=%d hint MQ=%s",
            self.rank,
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        self._lwd_early_recv_thread = threading.Thread(
            target=self._lwd_early_recv_loop,
            name="lwd-early-recv",
            daemon=True,
        )
        self._lwd_early_recv_thread.start()

    def lwd_early_recv_done_handle(self, connection_id: ConnectionId | None = None) -> Handle | None:
        """Handle of one connection's done MQ, for the engine reader."""
        done_mq = self._irecv_done_mqs.get(connection_id) if connection_id is not None else self._irecv_done_mq
        if done_mq is None:
            return None
        return done_mq.export_handle()

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
            "attached" if self._irecv_hint_mq is not None else "none",
        )
        hint_mqs = self._irecv_hint_mqs or {ConnectionId(0, 0, 0): self._irecv_hint_mq}
        hint_mqs = {connection: mq for connection, mq in hint_mqs.items() if mq is not None}
        while not self._lwd_early_recv_stop.is_set():
            for connection, hint_mq in hint_mqs.items():
                if connection in self._early_recv_failed_connections:
                    continue
                try:
                    hint = hint_mq.dequeue(timeout=_LWD_EARLY_RECV_DEQUEUE_TIMEOUT_S)
                    self.start_early_irecv(hint, connection_id=connection)
                except TimeoutError:
                    continue
                except Exception as exc:
                    self._lwd_fail_early_recv_connection(connection, self._early_recv_error_seqno(connection), exc)
                    time.sleep(_LWD_EARLY_RECV_ERROR_SLEEP_S)
            try:
                self._report_irecv_completions()
            except Exception as exc:
                key = self._early_recv_current_key
                connection = key[0] if key is not None else ConnectionId(0, 0, 0)
                self._lwd_fail_early_recv_connection(connection, self._early_recv_error_seqno(connection), exc)
            if hint_mqs and all(connection in self._early_recv_failed_connections for connection in hint_mqs):
                return
            # The ring's read side sched_yield-spins within the
            # just-read window (busy_loop_s=1s), so a 0.5ms timeout alone
            # does not cap the CPU — this sleep does; the cost is ~2ms of
            # added visibility latency on hints and watermarks each.
            time.sleep(_LWD_EARLY_RECV_LOOP_SLEEP_S)

    def _early_recv_error_seqno(self, connection: ConnectionId) -> int:
        key = self._early_recv_current_key
        if key is not None and key[0] == connection:
            return key[2]
        candidates = [
            key[2]
            for key in (*self._early_recv_futures, *self._early_recv_reserved, *self._early_recv_pending)
            if key[0] == connection
        ]
        return min(candidates) if candidates else -1

    def _lwd_fail_early_recv_connection(self, connection: ConnectionId, seqno: int, exc: Exception) -> None:
        """Clean one failed peer and report it through the engine error ABI."""
        with self._early_recv_lock:
            if connection in self._early_recv_failed_connections:
                return
            self._early_recv_failed_connections.add(connection)
            for key in list(self._early_recv_reserved):
                if key[0] == connection:
                    self._early_recv_reserved.discard(key)
            for key in list(self._early_recv_pending):
                if key[0] == connection:
                    self._early_recv_pending.pop(key, None)
            futures = [future for key, future in self._early_recv_futures.items() if key[0] == connection]
            for key in list(self._early_recv_futures):
                if key[0] == connection:
                    self._early_recv_futures.pop(key, None)
            self._early_recv_last_delivered.pop(connection, None)
            if self._early_recv_current_key is not None and self._early_recv_current_key[0] == connection:
                self._early_recv_current_key = None
        for future in futures:
            try:
                future.release_device_tensor()
            except Exception:
                lwd_logger.exception("[Lwd][pre-recv] failed to release receive tensor connection=%s", connection)
        if self._lwd_stager is not None:
            try:
                self._lwd_stager.abandon_connection(connection)
            except Exception:
                lwd_logger.exception("[Lwd][pre-recv] failed to clean host staging connection=%s", connection)
        try:
            lwd_get_data_plane().abort_connection(connection, exc)
        except Exception:
            lwd_logger.exception("[Lwd][pre-recv] failed to clean data plane connection=%s", connection)
        lwd_logger.exception("[Lwd][pre-recv] connection failed connection=%s seqno=%d", connection, seqno)
        done_mq = self._irecv_done_mqs.get(connection, self._irecv_done_mq)
        if done_mq is not None:
            try:
                done_mq.enqueue((connection, "error", seqno, repr(exc)))
            except Exception:
                lwd_logger.exception(
                    "[Lwd][pre-recv] failed to report comm error connection=%s seqno=%d",
                    connection,
                    seqno,
                )

    def _lwd_notify_early_recv_error(self, exc: Exception) -> None:
        """Compatibility entry point for a failure on the current connection."""
        key = self._early_recv_current_key
        connection = key[0] if key is not None else ConnectionId(0, 0, 0)
        self._lwd_fail_early_recv_connection(connection, key[2] if key is not None else -1, exc)

    def start_early_irecv(
        self,
        hint: dict[str, Any],
        connection_id: ConnectionId | None = None,
    ) -> None:
        """Queue one validated UP hint and submit it when capacity allows."""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        fields = hint_fields(hint)
        if fields is None or hint.get("channel") is not LwdChannelType.UP:
            connection = connection_id
            if connection is None and isinstance(hint, dict):
                raw_connection = hint.get("connection_id")
                if isinstance(raw_connection, ConnectionId):
                    connection = raw_connection
                else:
                    try:
                        connection = ConnectionId(
                            int(hint.get("edge_id", 0)),
                            int(hint.get("cloud_id", 0)),
                            int(hint.get("dp_idx", 0)),
                        )
                    except (TypeError, ValueError):
                        pass
            connection = connection or ConnectionId(0, 0, 0)
            try:
                seqno = int(hint.get("seqno", -1)) if isinstance(hint, dict) else -1
            except (TypeError, ValueError):
                seqno = -1
            self._lwd_fail_early_recv_connection(
                connection,
                seqno,
                ValueError(f"malformed UP receive hint: {hint!r}"),
            )
            return
        key = (fields.connection, LwdChannelType.UP, fields.seqno)
        with self._early_recv_lock:
            if fields.connection in self._early_recv_failed_connections:
                return
            if (
                fields.seqno <= self._early_recv_last_delivered.get(fields.connection, -1)
                or key in self._early_recv_futures
                or key in self._early_recv_reserved
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
        stage_key = (fields.connection, fields.seqno)
        recv_buffer, _slot = stager.allocate(stage_key, numel)
        segments = stager.plan_recv_segments(
            seqno=stage_key,
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

    def _early_recv_quota(self, connections: list[ConnectionId]) -> int:
        """Soft per-connection share; unused capacity remains work conserving."""
        if not connections:
            return self._early_recv_max_inflight
        return max(1, (self._early_recv_max_inflight + len(connections) - 1) // len(connections))

    def _pick_pending_locked(
        self, blocked: set[ConnectionId]
    ) -> tuple[tuple[ConnectionId, LwdChannelType, int], dict[str, Any]] | None:
        """Pick one connection head fairly, preferring its soft quota."""
        by_connection: dict[ConnectionId, list[tuple[ConnectionId, LwdChannelType, int]]] = {}
        for key in self._early_recv_pending:
            by_connection.setdefault(key[0], []).append(key)
        connections = sorted(connection for connection in by_connection if connection not in blocked)
        if not connections:
            return None
        quota = self._early_recv_quota(connections)
        start = self._early_recv_rr % len(connections)
        ordered = connections[start:] + connections[:start]
        counts: dict[ConnectionId, int] = {connection: 0 for connection in connections}
        for connection, _channel, _seqno in (*self._early_recv_futures, *self._early_recv_reserved):
            counts[connection] = counts.get(connection, 0) + 1
        selected = next((connection for connection in ordered if counts.get(connection, 0) < quota), None)
        if selected is None:
            selected = ordered[0]
        key = min(by_connection[selected], key=lambda item: item[2])
        self._early_recv_rr = (connections.index(selected) + 1) % len(connections)
        return key, self._early_recv_pending[key]

    def _early_recv_pump(self) -> None:
        """Reserve under lock, then submit independent FIFOs without cross-connection HOL blocking."""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import hint_fields

        blocked: set[ConnectionId] = set()
        while True:
            with self._early_recv_lock:
                if len(self._early_recv_futures) + len(self._early_recv_reserved) >= self._early_recv_max_inflight:
                    return
                picked = self._pick_pending_locked(blocked)
                if picked is None:
                    return
                key, hint = picked
                fields = hint_fields(hint)
                assert fields is not None
                self._early_recv_pending.pop(key)
                self._early_recv_reserved.add(key)
                self._early_recv_current_key = key
            try:
                future = self._lwd_submit_early_recv(fields)
            except LwdHostStageSlotBusy:
                with self._early_recv_lock:
                    self._early_recv_reserved.discard(key)
                    if fields.connection not in self._early_recv_failed_connections:
                        self._early_recv_pending[key] = hint
                    if self._early_recv_current_key == key:
                        self._early_recv_current_key = None
                blocked.add(fields.connection)
                continue
            except Exception as exc:
                with self._early_recv_lock:
                    self._early_recv_reserved.discard(key)
                    if self._early_recv_current_key == key:
                        self._early_recv_current_key = None
                if self._lwd_stager is not None:
                    self._lwd_stager.abandon((fields.connection, fields.seqno))
                self._lwd_fail_early_recv_connection(fields.connection, fields.seqno, exc)
                blocked.add(fields.connection)
                continue

            release_future = False
            with self._early_recv_lock:
                self._early_recv_reserved.discard(key)
                if fields.connection in self._early_recv_failed_connections:
                    release_future = True
                else:
                    self._early_recv_futures[key] = future
                    inflight = len(self._early_recv_futures)
                    queued = len(self._early_recv_pending)
                if self._early_recv_current_key == key:
                    self._early_recv_current_key = None
            if release_future:
                try:
                    future.release_device_tensor()
                except Exception:
                    lwd_logger.exception(
                        "[Lwd][pre-recv] failed to release receive tensor connection=%s", fields.connection
                    )
                blocked.add(fields.connection)
                continue
            lwd_logger.info(
                "[Lwd][pre-recv] SUBMIT connection=%s channel=%s seqno=%d "
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
        """Publish each connection's contiguous completion prefix independently."""
        stager = self._lwd_stager
        if stager is None:
            return
        with self._early_recv_lock:
            pending = sorted(self._early_recv_futures.items(), key=lambda item: (item[0][0], item[0][2]))
        blocked: set[ConnectionId] = set()
        for key, future in pending:
            connection, channel, seqno = key
            if connection in blocked or connection in self._early_recv_failed_connections:
                continue
            done_mq = self._irecv_done_mqs.get(connection, self._irecv_done_mq)
            self._early_recv_current_key = key
            try:
                if done_mq is None:
                    # No report reader for this connection: leave the future
                    # queued (aligned with the memc baseline; no consume-point
                    # fallback recycling).
                    blocked.add(connection)
                    continue
                if not future.done():
                    blocked.add(connection)
                    continue
                future.result()
                if not future.staged_done():
                    blocked.add(connection)
                    continue
                future.result()
                stage_key = (connection, seqno)
                published = stager.publish_received(stage_key)
                stager.release_if_drained(stage_key)
                arrived = [(req_id, offset + rows) for req_id, rows, offset in published]
                future.release_device_tensor()
                for req_id, nonce in stager.take_refill_done(connection):
                    done_mq.enqueue(("refill_done", req_id, nonce))
                done_mq.enqueue((key, arrived))
            except Exception as exc:
                self._lwd_fail_early_recv_connection(connection, seqno, exc)
                blocked.add(connection)
                continue
            finally:
                self._early_recv_current_key = None
            with self._early_recv_lock:
                self._early_recv_futures.pop(key, None)
                self._early_recv_last_delivered[connection] = seqno
            self._early_recv_current_key = None
            lwd_logger.info(
                "[Lwd][pre-recv] REPORT connection=%s channel=%s seqno=%d arrivals=%s",
                connection,
                channel.value,
                seqno,
                arrived,
            )
        self._early_recv_pump()

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        _t0 = time.monotonic()
        output = self.model_runner.sample_tokens(grammar_output)
        _t_sample = time.monotonic()
        # rank-replay DOWN: send the hidden packet (channel-async,
        # in-stream ordered). The executor output thread harvests each
        # connection's device meta behind its per-step ready event, keeping
        # this worker thread free to prepare and execute subsequent steps.
        payloads = self.model_runner.take_lwd_pending_down_payload()
        if payloads:
            from vllm.distributed.parallel_state import get_tp_group

            is_driver = get_tp_group().is_first_rank
            carriers = []
            legacy_carrier = None
            ready_events = {}
            legacy_ready_event = None
            for payload in payloads:
                connection = payload.connection
                hidden = payload.hidden
                meta_dev = payload.meta_dev
                seg_lens_host = payload.seg_lens_host
                ready_event = payload.ready_event
                req_ids = payload.req_ids
                seqno = self._lwd_next_down_seqno(connection)
                if connection in self._lwd_early_recv_connections:
                    lwd_logger.info(
                        "[Lwd][cloud-worker] DOWN send connection=%s seqno=%d numel=%d",
                        connection,
                        seqno,
                        hidden.numel(),
                    )
                    lwd_get_data_plane().submit_send(
                        LwdTransferRequest(
                            channel=LwdChannelType.DOWN,
                            op="send",
                            num_elements=hidden.numel(),
                            dp_idx=connection.dp_idx,
                            edge_id=connection.edge_id,
                            cloud_id=connection.cloud_id,
                            tensor=hidden,
                            seqno=seqno,
                        )
                    )
                if not is_driver:
                    # Every rank advances the same seqno, but only TP0 returns
                    # C2E metadata through the executor's single output rank.
                    continue
                # The device meta and its ready event travel with the output;
                # the executor output thread owns the fenced D2H harvest.
                carrier = (meta_dev, seg_lens_host, req_ids, hidden.numel(), seqno, connection)
                carriers.append(carrier)
                ready_events[connection] = ready_event
                if legacy_carrier is None:
                    legacy_carrier = carrier[:5]
                    legacy_ready_event = ready_event
                lwd_logger.info(
                    "[Lwd][perf] cloud-step connection=%s seqno=%d sample=%.2f send=%.2f total=%.2fms rows=%d",
                    connection,
                    seqno,
                    (_t_sample - _t0) * 1000,
                    (time.monotonic() - _t_sample) * 1000,
                    (time.monotonic() - _t0) * 1000,
                    hidden.shape[0],
                )
            if is_driver and output is not None:
                target = getattr(output, "_model_runner_output", output)
                target.lwd_down_carriers = carriers
                target.lwd_down_carrier = legacy_carrier
                # The events belong to this output generation and must be
                # consumed before the output crosses the process boundary.
                target.lwd_meta_ready_events = ready_events
                target.lwd_meta_ready_event = legacy_ready_event
        return output

    # ------------------------------------------------------------------ #
    # LWD housekeeping (control-plane glue)                               #
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        """Quiesce early receive before closing its data-plane FIFOs."""
        self._lwd_early_recv_stop.set()
        thread = self._lwd_early_recv_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_LWD_SHUTDOWN_TIMEOUT_S)
            if thread.is_alive():
                lwd_logger.warning("[Lwd][pre-recv] comm thread did not stop within %.1fs", _LWD_SHUTDOWN_TIMEOUT_S)
        try:
            lwd_get_data_plane().shutdown(timeout=_LWD_SHUTDOWN_TIMEOUT_S)
            if thread is not None and thread.is_alive():
                # A blocked HCCL call may outlive this worker. The data plane is
                # permanently closed against new submissions, but its channels
                # must remain valid until the process exits and reaps the thread.
                lwd_logger.warning(
                    "[Lwd][pre-recv] skip channel destruction while comm thread is alive"
                )
            else:
                from vllm_ascend.distributed.lwd_data_plane.lwd_channels import (
                    lwd_destroy_channels,
                )

                lwd_destroy_channels()
        finally:
            super().shutdown()

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

    def _lwd_next_down_seqno(self, connection: ConnectionId | int = ConnectionId(0, 0, 0)) -> int:
        if isinstance(connection, int):
            connection = ConnectionId(0, 0, connection)
        seqno = self._lwd_down_next_seqno.get(connection, 0)
        self._lwd_down_next_seqno[connection] = seqno + 1
        return seqno
