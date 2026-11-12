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
"""

from __future__ import annotations

import base64
import json
import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, NamedTuple

from vllm_ascend.distributed.lwd_data_plane.lwd_types import ConnectionId, LwdChannelType

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue

#: Env var carrying the downstream hint MQ handle. The executor exports it
#: before creating the worker; the worker rebuilds the reader in
#: ``init_device`` — both sides honor this single constant name.
LWD_RECV_HINT_MQ_ENV = "VLLM_ASCEND_LWD_RECV_HINT_HANDLE"

_SIDEBAND_MAX_CHUNK_BYTES = 4096
_SIDEBAND_MAX_CHUNKS = 64


class LwdRecvHintFields(NamedTuple):
    connection: ConnectionId
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
    req_ids: list[str],
    rows: list[int],
    prompt_offsets: list[int],
    prompt_lens: list[int],
    refills: list[bool] | None = None,
    prefix_nonces: list[int] | None = None,
    connection_id: ConnectionId | None = None,
    edge_id: int = 0,
    cloud_id: int = 0,
    dp_idx: int = 0,
) -> dict[str, Any]:
    """Build one UP hint with the complete per-request chunk geometry."""
    connection = connection_id or ConnectionId(edge_id, cloud_id, dp_idx)
    return {
        "connection_id": connection,
        "edge_id": connection.edge_id,
        "cloud_id": connection.cloud_id,
        "dp_idx": connection.dp_idx,
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
        connection = hint.get("connection_id")
        if connection is None:
            connection = ConnectionId(
                int(hint.get("edge_id", 0)),
                int(hint.get("cloud_id", 0)),
                int(hint.get("dp_idx", 0)),
            )
        if not isinstance(connection, ConnectionId):
            return None
        fields = LwdRecvHintFields(
            connection=connection,
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


def _hint_env_key(connection: ConnectionId) -> str:
    if connection == ConnectionId(0, 0, 0):
        return LWD_RECV_HINT_MQ_ENV
    return f"{LWD_RECV_HINT_MQ_ENV}_E{connection.edge_id}C{connection.cloud_id}D{connection.dp_idx}"


def create_hint_mq(connection_id: ConnectionId | None = None) -> MessageQueue:
    """Engine side: create the hint MQ (writer) and export the handle to
    the environment.

    Must be called before workers are created (they rebuild the reader
    from the env at startup); both fork and spawn inherit via env vars.
    """
    connection = connection_id or ConnectionId(0, 0, 0)
    mq = _new_sideband_mq()
    os.environ[_hint_env_key(connection)] = _encode_handle(mq.export_handle())
    return mq


def attach_hint_mq(connection_id: ConnectionId | None = None) -> MessageQueue | None:
    """Worker side: rebuild the hint MQ reader from the handle in the
    environment.

    Returns None when no handle exists (this role takes no hints): the
    comm thread still runs reporting, so a missing hint only delays the
    early recv, never stalls the watermark.
    """
    connection = connection_id or ConnectionId(0, 0, 0)
    raw = os.environ.get(_hint_env_key(connection))
    if raw is None:
        return None
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    return MessageQueue.create_from_handle(_decode_handle(raw), 0)


def create_report_mq(connection_id: ConnectionId | None = None) -> MessageQueue:
    """Worker side: create the done MQ (writer); the engine fetches its
    handle over ``collective_rpc`` once the executor is up."""
    return _new_sideband_mq()


# ---------------------------------------------------------------------- #
# Scheduler process side: hint dispatch + completion watermark           #
# ---------------------------------------------------------------------- #
_hint_sender: Callable[[dict[str, Any]], None] | None = None
_hint_senders: dict[ConnectionId, Callable[[dict[str, Any]], None]] = {}
_watermarks: dict[tuple[ConnectionId, LwdChannelType], int] = {}
_lock = threading.Lock()


def register_hint_sender(
    sender: Callable[[dict[str, Any]], None],
    connection_id: ConnectionId | None = None,
) -> None:
    """Install the transport used to dispatch hints (once per engine
    process).

    The closure passed in must **deliver reliably** (enqueue the hint
    into the hint MQ): with the gate on, hints are a correctness
    dependency — lose one and that recv is never pre-posted, the
    watermark stalls, and the gated batch is never dispatched.
    """
    global _hint_sender
    if connection_id is None:
        _hint_sender = sender
    else:
        _hint_senders[connection_id] = sender


def post_irecv_hint(hint: dict[str, Any]) -> None:
    """Send one recv hint to the local worker's comm thread.

    The healthy path logs nothing (the caller ``lwd_cloud_engine`` logs
    ``HINT ...``): this module keeps only should-never-happen diagnostics,
    and they must be visible in the logs.
    """
    fields = hint_fields(hint)
    sender = _hint_senders.get(fields.connection) if fields is not None else None
    sender = sender or _hint_sender
    if sender is None:
        # The engine registers the sender during startup, before it accepts
        # any edge message, so a missing one is a wiring error; dropping the
        # hint would stall the pre-recv gate forever (nobody would pre-post
        # that recv, the watermark would never advance).
        raise RuntimeError(f"[LWD] early-recv hint for seqno {hint.get('seqno')} has no registered sender")
    sender(hint)


def record_irecv_completions(
    items: Iterable[tuple],
) -> list[tuple]:
    """Merge completions into independent per-connection watermarks."""
    advanced: list[tuple] = []
    with _lock:
        for item in items:
            if len(item) == 2:
                channel, seqno = item
                connection = ConnectionId(0, 0, 0)
                advanced_item = (channel, seqno)
            else:
                connection, channel, seqno = item
                advanced_item = item
            key = (connection, channel)
            if seqno > _watermarks.get(key, -1):
                _watermarks[key] = seqno
                advanced.append(advanced_item)
    return advanced


def watermark(
    channel: LwdChannelType,
    connection_id: ConnectionId | None = None,
) -> int:
    """Current watermark on one connection (-1 = none yet)."""
    connection = connection_id or ConnectionId(0, 0, 0)
    with _lock:
        return _watermarks.get((connection, channel), -1)


def is_irecv_complete(
    channel: LwdChannelType,
    seqno: int,
    connection_id: ConnectionId | None = None,
) -> bool:
    """Whether one connection's receive has completed."""
    return seqno <= watermark(channel, connection_id)
