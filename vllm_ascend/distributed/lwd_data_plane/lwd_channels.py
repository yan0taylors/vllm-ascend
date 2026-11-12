# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UP and DOWN channels of the LWD edge-cloud data plane.

Owns the physical channels (UP edge->cloud, DOWN cloud->edge):
  * channel HCCL communicators: two dedicated ``new_group`` groups per DP
    connection, over its endpoint pair (``LwdChannelEndpoints``: the first
    edge rank and the first cloud rank);
  * one NPU stream per channel of each DP connection the local rank
    endpoints (channel ops are bridged onto it, keeping the compute stream
    free);
  * init-time warmup of every connection's channels in a fixed global
    order (moves the first-use HCCL rendezvous out of the pipeline, where
    the two sides could otherwise arrive on different channels and
    deadlock).

In a single_instance topology the edge process serves every DP
connection, so its endpoint rank holds the channel pairs of all of them;
a cloud endpoint rank holds exactly its own DP's pair.

Design: docs/design/lwd_data_plane.md in the vLLM repository.
"""

from __future__ import annotations

import threading
import time

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.distributed.lwd_data_plane.lwd_types import ConnectionId, LwdChannelType

_LWD_WARMUP_NUM_ELEMENTS = 5
_LWD_WIRE_MAGIC = 0x4C5744
_LWD_WIRE_VERSION = 1

# (connection, channel) -> (device_group, peer_global_rank), endpoint ranks only
_LWD_CHANNEL_GROUPS: dict[tuple[ConnectionId, LwdChannelType], tuple[dist.ProcessGroup, int]] = {}
_LWD_CHANNEL_STREAMS: dict[tuple[ConnectionId, LwdChannelType], torch.npu.Stream] = {}
_LWD_ENDPOINTS: tuple[LwdChannelEndpoints, ...] = ()
_LWD_CHANNELS_INITIALIZED = False


def lwd_init_channels() -> None:
    """Create the UP and DOWN channels of every DP connection. Idempotent.

    Called once by every worker of the shared world after its distributed
    environment is up. Group creation is a world collective, so every rank
    creates the groups of every connection in the order of
    ``LwdParallelGroupsSpec``; only the two endpoint ranks keep them.
    """
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    if _LWD_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    assert lwd_config is not None, "the data plane requires lwd_config"
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()
    for endpoints in spec.channel_endpoints:
        connection = endpoints.connection_id
        ranks = [endpoints.edge_rank, endpoints.cloud_rank]
        for channel in LwdChannelType:
            # Both channels span the same rank pair; the direction is given
            # by who sends.
            group = dist.new_group(ranks, backend=backend)
            if my_rank in ranks:
                peer = endpoints.cloud_rank if my_rank == endpoints.edge_rank else endpoints.edge_rank
                _LWD_CHANNEL_GROUPS[(connection, channel)] = (group, peer)
                _LWD_CHANNEL_STREAMS[(connection, channel)] = torch.npu.Stream()
    _LWD_ENDPOINTS = spec.channel_endpoints_of(my_rank)
    # The cloud-internal embeds broadcast does not build its own group: a
    # dedicated new_group delivered nothing under torch_npu in testing
    # (suspected group creation order mismatch), so it uses the framework TP
    # device_group, whose creation order the framework guarantees. The
    # broadcast is issued on the compute stream at injection time, FIFO with
    # the forward TP collectives, so the two cannot interleave.
    _LWD_CHANNELS_INITIALIZED = True
    logger.info(
        "[LWD] data-plane channels created for %s (my_rank=%d, endpoints=%s)",
        spec.channel_endpoints,
        my_rank,
        _LWD_ENDPOINTS,
    )
    # Every connection's two channels, always warmed up: the first-use HCCL
    # rendezvous is moved to init time, where both sides are guaranteed to
    # arrive in the same fixed order.
    lwd_warm_up_channels()


def _lwd_echo_handshake(endpoints: LwdChannelEndpoints, connection: ConnectionId, timeout_s: float) -> None:
    """Bound the whole echo, including HCCL posting and D2H synchronization.

    A timed-out HCCL operation cannot be cancelled safely. Startup must fail
    and the process must be restarted rather than reusing this channel.
    """
    done = threading.Event()
    errors: list[BaseException] = []
    phase = ["posting"]
    device = torch.npu.current_device()

    def run_echo() -> None:
        try:
            # NPU device selection is thread-local; do not fall back to card 0.
            torch.npu.set_device(device)
            _lwd_echo_handshake_impl(endpoints, connection, phase)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=run_echo, name=f"lwd-echo-{connection}", daemon=True).start()
    if not done.wait(timeout_s * 2):
        raise TimeoutError(
            f"[LWD] channel handshake timed out after {timeout_s * 2}s: "
            f"{connection}, rank {dist.get_rank()}, phase {phase[0]}; restart required"
        )
    if errors:
        raise RuntimeError(f"[LWD] channel handshake failed on {connection}: {errors[0]!r}") from errors[0]


def _lwd_echo_handshake_impl(
    endpoints: LwdChannelEndpoints, connection: ConnectionId, phase: list[str]
) -> None:
    my_rank = dist.get_rank()
    up_group, up_peer = _LWD_CHANNEL_GROUPS[(connection, LwdChannelType.UP)]
    down_group, down_peer = _LWD_CHANNEL_GROUPS[(connection, LwdChannelType.DOWN)]
    expected = [
        _LWD_WIRE_MAGIC,
        _LWD_WIRE_VERSION,
        connection.edge_id,
        connection.cloud_id,
        connection.dp_idx,
    ]
    if my_rank == endpoints.edge_rank:
        send = torch.tensor(expected, dtype=torch.int32, device="npu")
        recv = torch.zeros(_LWD_WARMUP_NUM_ELEMENTS, dtype=torch.int32, device="npu")
        send_work = dist.isend(send, dst=up_peer, group=up_group)
        recv_work = dist.irecv(recv, src=down_peer, group=down_group)
        phase[0] = "UP send wait"
        send_work.wait()
        phase[0] = "DOWN receive wait"
        recv_work.wait()
    else:
        recv = torch.zeros(_LWD_WARMUP_NUM_ELEMENTS, dtype=torch.int32, device="npu")
        recv_work = dist.irecv(recv, src=up_peer, group=up_group)
        phase[0] = "UP receive wait"
        recv_work.wait()
        phase[0] = "UP receive D2H"
        received = [int(value) for value in recv.cpu().tolist()]
        if received != expected:
            raise RuntimeError(f"[LWD] channel identity/version mismatch on {connection}: expected {expected}, got {received}")
        send_work = dist.isend(recv, dst=down_peer, group=down_group)
        phase[0] = "DOWN echo send wait"
        send_work.wait()
    phase[0] = "echo verification D2H"
    if not torch.equal(recv.cpu(), torch.tensor(expected, dtype=torch.int32)):
        raise RuntimeError(f"[LWD] channel echo mismatch on {connection}: expected {expected}, got {recv.cpu().tolist()}")


def lwd_warm_up_channels() -> None:
    """Validate version, connection identity, and both directions of every local pair."""
    timeout_s = envs.VLLM_ASCEND_LWD_HCCL_HANDSHAKE_TIMEOUT_S
    if timeout_s <= 0:
        raise ValueError("VLLM_ASCEND_LWD_HCCL_HANDSHAKE_TIMEOUT_S must be positive")
    started = time.monotonic()
    for endpoints in _LWD_ENDPOINTS:
        connection = endpoints.connection_id
        _lwd_echo_handshake(endpoints, connection, timeout_s)
        logger.info(
            "[LWD] data-plane echo validated connection=%s version=%d edge_rank=%d cloud_rank=%d",
            connection,
            _LWD_WIRE_VERSION,
            endpoints.edge_rank,
            endpoints.cloud_rank,
        )
    logger.info(
        "[LWD] data-plane handshake complete rank=%d connections=%d duration=%.1fms",
        dist.get_rank(),
        len(_LWD_ENDPOINTS),
        (time.monotonic() - started) * 1000,
    )


def lwd_channels_initialized() -> bool:
    return _LWD_CHANNELS_INITIALIZED


def _lwd_channel(
    dp_idx: int,
    channel: LwdChannelType,
    edge_id: int,
    cloud_id: int,
) -> tuple[dist.ProcessGroup, int]:
    connection = ConnectionId(edge_id, cloud_id, dp_idx)
    try:
        return _LWD_CHANNEL_GROUPS[(connection, channel)]
    except KeyError:
        raise RuntimeError(
            f"[LWD] rank {dist.get_rank()} holds no {channel.value} channel of "
            f"connection {connection}; only its endpoint ranks do"
        ) from None


def lwd_get_channel_device_group(
    channel: LwdChannelType,
    dp_idx: int = 0,
    *,
    edge_id: int = 0,
    cloud_id: int = 0,
) -> dist.ProcessGroup:
    group, _ = _lwd_channel(dp_idx, channel, edge_id, cloud_id)
    return group


def lwd_get_channel_peer(
    channel: LwdChannelType,
    dp_idx: int = 0,
    *,
    edge_id: int = 0,
    cloud_id: int = 0,
) -> int:
    """Global rank of this process's P2P peer on the connection's channel."""
    _, peer = _lwd_channel(dp_idx, channel, edge_id, cloud_id)
    return peer


def lwd_get_channel_stream(
    channel: LwdChannelType,
    dp_idx: int = 0,
    *,
    edge_id: int = 0,
    cloud_id: int = 0,
) -> torch.npu.Stream:
    _lwd_channel(dp_idx, channel, edge_id, cloud_id)
    return _LWD_CHANNEL_STREAMS[(ConnectionId(edge_id, cloud_id, dp_idx), channel)]


def lwd_destroy_channels() -> None:
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    for (connection, channel), (group, _) in list(_LWD_CHANNEL_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning("[LWD] failed to destroy the data-plane group for %s %s", connection, channel.value)
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_ENDPOINTS = ()
    _LWD_CHANNELS_INITIALIZED = False
