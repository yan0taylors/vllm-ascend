# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Type definitions of the LWD data plane."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Literal

from vllm.config.lwd.topology import LwdConnectionId as ConnectionId


def connection_from(metadata: Any) -> ConnectionId:
    """Read canonical batch identity, falling back at the legacy boundary."""
    connection = getattr(metadata, "connection_id", None)
    if connection is not None:
        return connection
    return ConnectionId(
        getattr(metadata, "edge_id", 0),
        getattr(metadata, "cloud_id", 0),
        getattr(metadata, "dp_idx", 0),
    )


class LwdChannelType(enum.Enum):
    """The two physical data-plane channels, direction-only.

    Each channel maps 1:1 to a dedicated HCCL communicator + NPU stream
    (see ``vllm_ascend.distributed.lwd_data_plane.lwd_channels``).  HCCL P2P matching order
    per (communicator, peer) is exactly the per-channel ``seqno``
    submission order — HCCL does not support tags.
    """

    UP = "lwd_up"  # edge -> cloud: embeddings + optional V4 expert-ID bytes
    DOWN = "lwd_down"  # cloud -> edge: combined c2e packet


class LwdTransferStatus(enum.Enum):
    PENDING = "pending"
    OK = "ok"
    ERROR = "error"


@dataclass
class LwdTransferRequest:
    """One communication task submitted to the service.

    Everything the data plane needs travels with the request — it never
    reaches back into scheduler/model state.
    """

    channel: LwdChannelType
    op: Literal["send", "recv"]
    # Number of bf16 elements of the payload (UP: N*H, or N*(H+4*L*K)
    # for V4 with L Hash layers and K experts per token; DOWN:
    # 32 + R*(H+3K)).  recv: used to allocate the exact-size buffer
    # (HCCL P2P requires matching numel on both ends).
    num_elements: int
    # Connection this transfer belongs to. The three fields remain separate
    # for compatibility with the current vLLM batch metadata; new callers
    # should treat ``connection`` as the single routing identity.
    dp_idx: int = 0
    edge_id: int = 0
    cloud_id: int = 0
    # send: payload tensor (snapshotted into a communication-owned buffer
    # at submit time).  recv: None.
    tensor: Any | None = None
    # Per-channel request-level sequence number.  Contract: each
    # channel's counter starts at 0 and increments by one per request on
    # both peers; ops are posted to HCCL only once all lower seqnos have
    # been submitted, so send order and the peer's recv-post order always
    # agree.  Do not mix sequenced and unsequenced requests on one
    # channel.
    seqno: int | None = None
    # Explicit global peer rank; None -> the channel's configured peer.
    peer_rank: int | None = None
    # Optional second payload of the SAME logical request (currently
    # UP-only: mrope positions, int64 [n,3], accompanying an embeds chunk
    # when the batch has mm data).  Physically a second P2P issued right
    # after the main one on the same channel, but sequencing stays at
    # logical-request granularity: one seqno covers both frames, so
    # skip/drain semantics are unchanged.  send: tensor; recv: None and
    # ``aux_num_elements`` sizes the exact buffer.
    #
    # Extension point: to add another payload kind, turn this group of
    # fields (aux_tensor / aux_num_elements / aux_dtype) into a
    # list[TensorSpec]. Snapshot, keepalive, handle bridging and recv buffer
    # allocation in channel/future are already written as "main + aux ops"
    # and extend mechanically; the seqno discipline and the size
    # announcement (the notify carries per-frame numel/dtype) extend with it.
    aux_tensor: Any | None = None
    aux_num_elements: int = 0
    # Element dtype of the aux frame (torch.dtype; None = int64, mrope
    # positions). The receiver allocates its exact buffer from it; a later
    # payload (e.g. a bf16 deepstack frame) only sets it, without changing
    # the channel structure.
    aux_dtype: Any | None = None
    # recv: optional caller-owned device slot for the main frame. Its numel
    # must exactly match num_elements; when omitted, the queue allocates it.
    recv_buffer: Any | None = None
    # recv: optional segmented D2H destinations. Each tuple contains a pinned
    # host view, a flat source element offset, and the number of elements.
    stage_segments: list[tuple[Any, int, int]] | None = None

    @property
    def connection(self) -> ConnectionId:
        return ConnectionId(self.edge_id, self.cloud_id, self.dp_idx)


@dataclass
class LwdTransferResult:
    """Outcome of a completed request (cross-node P2P finished on device)."""

    status: LwdTransferStatus
    tensor: Any | None = None
    error: BaseException | None = None
    # Received aux payload (mrope positions int64), None when the
    # request carried no aux frame.
    aux_tensor: Any | None = None
