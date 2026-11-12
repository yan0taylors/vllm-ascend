# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD early receive hints and completion watermarks.

The engine sends one hint per announced UP chunk over a dedicated message
queue. The cloud endpoint plans per-request host staging from the hint's row
tables, posts the receive, and reports each completed chunk as its channel
watermark plus cumulative per-request arrival rows. A connection failure is
reported as ``(connection, "error", seqno, repr(exc))``; its receive state is
retired without stopping service to other connections.

Both queues bypass ``rpc_broadcast_mq``, whose single consumer can block in
model execution. The hint reader handle travels through the environment; the
engine obtains the done-writer handle through ``collective_rpc``. Handles use
JSON rather than pickle.

Non-endpoint ranks own neither a cross-node receive nor host staging, so they
do not participate in this protocol.
"""  # 【MR2】模块说明更新：连接失败按 (connection, error, seqno, repr) 上报，只退休该连接的接收状态

from __future__ import annotations

import base64
import json
import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, NamedTuple

from vllm_ascend.distributed.lwd_data_plane.lwd_types import ConnectionId, LwdChannelType  # 【MR2】引入 ConnectionId 和通道类型：hint/水位都按连接区分

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue

#: Env var carrying the downstream hint MQ handle. The executor exports it
#: before creating the worker; the worker rebuilds the reader in
#: ``init_device`` — both sides honor this single constant name.
LWD_RECV_HINT_MQ_ENV = "VLLM_ASCEND_LWD_RECV_HINT_HANDLE"

_SIDEBAND_MAX_CHUNK_BYTES = 4096
_SIDEBAND_MAX_CHUNKS = 64


class LwdRecvHintFields(NamedTuple):
    connection: ConnectionId  # 【MR2】hint 字段第一项就是连接编号
    seqno: int
    num_tokens: int
    aux_rows: int
    req_ids: list[str]
    rows: list[int]
    prompt_offsets: list[int]
    prompt_lens: list[int]
    refills: list[bool]
    prefix_nonces: list[int]


def make_recv_hint(
    seqno: int,
    num_tokens: int,
    aux_rows: int = 0,
    *,
    connection_id: ConnectionId,  # 【MR2】构造 hint 必须显式给连接编号（关键字参数，避免漏传）
    req_ids: list[str],
    rows: list[int],
    prompt_offsets: list[int],
    prompt_lens: list[int],
    refills: list[bool] | None = None,
    prefix_nonces: list[int] | None = None,
) -> dict[str, Any]:
    """Build one UP hint with the complete per-request chunk geometry."""
    return {
        "connection_id": connection_id,  # 【MR2】hint 载荷写入 connection_id
        "channel": LwdChannelType.UP,
        "seqno": int(seqno),
        "num_tokens": int(num_tokens),
        "aux_rows": int(aux_rows),
        "req_ids": list(req_ids),
        "rows": [int(value) for value in rows],
        "prompt_offsets": [int(value) for value in prompt_offsets],
        "prompt_lens": [int(value) for value in prompt_lens],
        "refills": [bool(value) for value in (refills or ())],
        "prefix_nonces": list(prefix_nonces or ()),
    }


def hint_fields(hint: dict[str, Any]) -> LwdRecvHintFields | None:
    """Return validated hint fields, or None for inconsistent geometry."""
    if not isinstance(hint, dict):
        return None
    table_names = ("req_ids", "rows", "prompt_offsets", "prompt_lens")
    if any(not isinstance(hint.get(name), list) for name in table_names):
        return None
    if any(not isinstance(hint.get(name, []), list) for name in ("refills", "prefix_nonces")):
        return None
    try:
        connection = hint.get("connection_id")  # 【MR2】解析 hint 时先取 connection_id
        if not isinstance(connection, ConnectionId):
            return None  # 【MR2】连接不是 ConnectionId 说明 hint 非法，直接返回 None
        fields = LwdRecvHintFields(
            connection=connection,  # 【MR2】把连接放进结构化字段
            seqno=int(hint["seqno"]),
            num_tokens=int(hint["num_tokens"]),
            aux_rows=int(hint["aux_rows"]),
            req_ids=list(hint["req_ids"]),
            rows=[int(value) for value in hint["rows"]],
            prompt_offsets=[int(value) for value in hint["prompt_offsets"]],
            prompt_lens=[int(value) for value in hint["prompt_lens"]],
            refills=[bool(value) for value in hint.get("refills", [])],
            prefix_nonces=list(hint.get("prefix_nonces", [])),
        )
    except (KeyError, TypeError, ValueError):
        return None
    tables = (fields.req_ids, fields.rows, fields.prompt_offsets, fields.prompt_lens)
    if not fields.req_ids or len({len(table) for table in tables}) != 1:
        return None
    if any(not isinstance(req_id, str) or not req_id for req_id in fields.req_ids):
        return None
    if len(set(fields.req_ids)) != len(fields.req_ids):
        return None
    if fields.seqno < 0 or fields.num_tokens <= 0 or not 0 <= fields.aux_rows <= fields.num_tokens:
        return None
    if any(rows <= 0 or prompt_len <= 0 for rows, prompt_len in zip(fields.rows, fields.prompt_lens)):
        return None
    if any(offset < 0 for offset in fields.prompt_offsets) or sum(fields.rows) != fields.num_tokens:
        return None
    if any(
        offset + rows > prompt_len
        for offset, rows, prompt_len in zip(fields.prompt_offsets, fields.rows, fields.prompt_lens)
    ):
        return None
    if any(not isinstance(nonce, int) or isinstance(nonce, bool) or nonce < 0 for nonce in fields.prefix_nonces):
        return None
    if len(fields.refills) > len(fields.req_ids) or len(fields.prefix_nonces) > len(fields.req_ids):
        return None
    fields.refills.extend([False] * (len(fields.req_ids) - len(fields.refills)))
    fields.prefix_nonces.extend([0] * (len(fields.req_ids) - len(fields.prefix_nonces)))
    return fields


# ---------------------------------------------------------------------- #
# Bypass MQ lifecycle                                                    #
# ---------------------------------------------------------------------- #
def _new_sideband_mq() -> MessageQueue:
    """A 1-reader 1-writer bypass MQ (hint and report share the spec; only the endpoint rank participates)."""
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue(
        1,
        1,
        max_chunk_bytes=_SIDEBAND_MAX_CHUNK_BYTES,
        max_chunks=_SIDEBAND_MAX_CHUNKS,
    )


def _encode_handle(handle: Handle) -> str:
    """Encode a MessageQueue handle into an env-safe ASCII string.

    Every Handle field is a JSON-native type, so the sideband handoff
    never needs pickle (a forbidden import in this repo).
    """
    return base64.b64encode(json.dumps(asdict(handle)).encode("ascii")).decode("ascii")


def _decode_handle(raw: str) -> Handle:
    """Rebuild a Handle from its JSON encoding.

    JSON arrays come back as lists; restore the tuple the MessageQueue
    expects in buffer_handle.
    """
    fields = json.loads(base64.b64decode(raw))
    if fields.get("buffer_handle") is not None:
        fields["buffer_handle"] = tuple(fields["buffer_handle"])
    from vllm.distributed.device_communicators.shm_broadcast import Handle

    return Handle(**fields)


def _hint_env_key(connection: ConnectionId) -> str:  # 【MR2】每条连接一个 hint 队列的环境变量名
    if connection == ConnectionId(0, 0, 0):  # 【MR2】默认连接沿用旧变量名，兼容 1E1C
        return LWD_RECV_HINT_MQ_ENV
    return f"{LWD_RECV_HINT_MQ_ENV}_E{connection.edge_id}C{connection.cloud_id}D{connection.dp_idx}"  # 【MR2】其它连接加 _E{edge}C{cloud}D{dp} 后缀，互不覆盖


def create_hint_mq(connection_id: ConnectionId) -> MessageQueue:  # 【MR2】创建 hint MQ 时按连接导出 handle
    """Engine side: create the hint MQ (writer) and export the handle to
    the environment.

    Must be called before workers are created (they rebuild the reader
    from the env at startup); both fork and spawn inherit via env vars.
    """
    mq = _new_sideband_mq()
    os.environ[_hint_env_key(connection_id)] = _encode_handle(mq.export_handle())  # 【MR2】handle 写到该连接专属的环境变量
    return mq


def attach_hint_mq(connection_id: ConnectionId) -> MessageQueue | None:  # 【MR2】worker 侧按连接从环境变量重建 hint MQ
    """Worker side: rebuild the hint MQ reader from the handle in the
    environment.

    Returns None when no handle exists (this role takes no hints): the
    comm thread still runs reporting, so a missing hint only delays the
    early recv, never stalls the watermark.
    """
    raw = os.environ.get(_hint_env_key(connection_id))  # 【MR2】读取该连接专属的环境变量
    if raw is None:
        return None
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue.create_from_handle(_decode_handle(raw), 0)


def create_report_mq() -> MessageQueue:
    """Worker side: create the done MQ (writer); the engine fetches its
    handle over ``collective_rpc`` once the executor is up."""
    return _new_sideband_mq()


# ---------------------------------------------------------------------- #
# Scheduler process side: hint dispatch + completion watermark           #
# ---------------------------------------------------------------------- #
_hint_senders: dict[ConnectionId, Callable[[dict[str, Any]], None]] = {}  # 【MR2】hint 发送器按连接登记
_watermarks: dict[tuple[ConnectionId, LwdChannelType], int] = {}  # 【MR2】水位按（连接, 通道）分开，seqno 相同也不冲突
_lock = threading.Lock()


def register_hint_sender(  # 【MR2】为某条连接登记发送器
    sender: Callable[[dict[str, Any]], None],
    connection_id: ConnectionId,
) -> None:
    """Install the reliable hint transport for one engine connection.

    Losing a hint leaves its receive unposted and stalls the watermark.
    """  # 【MR2】发送器按连接登记；丢 hint 会卡住该连接的预收和水位，所以必须可靠
    _hint_senders[connection_id] = sender  # 【MR2】按连接保存发送器


def post_irecv_hint(hint: dict[str, Any]) -> None:
    """Send one validated recv hint to its connection's comm thread."""  # 【MR2】发送前先校验 hint 格式
    fields = hint_fields(hint)  # 【MR2】解析出连接等字段
    if fields is None:
        raise ValueError(f"[LWD] malformed early-recv hint: {hint!r}")
    sender = _hint_senders.get(fields.connection)  # 【MR2】按连接找发送器
    if sender is None:
        # The engine registers every sender during startup, before accepting  # 【MR2】找不到只报错，绝不借用其它连接的发送器
        # edge messages. Never substitute another connection's sender: losing
        # or misrouting the hint would stall the pre-recv gate forever.
        raise RuntimeError(  # 【MR2】直接失败：错投 hint 会让另一条连接卡死
            f"[LWD] early-recv hint for {fields.connection} seqno "
            f"{fields.seqno} has no registered sender"
        )
    sender(hint)


def record_irecv_completions(  # 【MR2】完成项统一合入
    items: Iterable[tuple],  # 【MR2】每项是 (连接, 通道, seqno) 三元组
) -> list[tuple]:
    """Merge completions into independent per-connection watermarks."""  # 【MR2】合并成每条连接独立的完成水位
    advanced: list[tuple] = []
    with _lock:
        for item in items:  # 【MR2】拆出连接、通道、seqno
            connection, channel, seqno = item
            key = (connection, channel)  # 【MR2】水位 key 包含连接
            if seqno > _watermarks.get(key, -1):  # 【MR2】只推进更大的 seqno
                _watermarks[key] = seqno
                advanced.append(item)
    return advanced


def watermark(  # 【MR2】查询某条连接某通道的水位
    channel: LwdChannelType,
    connection_id: ConnectionId | None = None,
) -> int:
    """Current watermark on one connection (-1 = none yet)."""
    connection = connection_id or ConnectionId(0, 0, 0)  # 【MR2】不传连接时按默认连接（兼容旧调用）
    with _lock:
        return _watermarks.get((connection, channel), -1)  # 【MR2】返回水位，-1 表示还没有完成项


def is_irecv_complete(  # 【MR2】判断某条连接的某个 seqno 是否已完成
    channel: LwdChannelType,
    seqno: int,
    connection_id: ConnectionId | None = None,
) -> bool:
    """Whether one connection's receive has completed."""
    return seqno <= watermark(channel, connection_id)  # 【MR2】与水位比较即可
