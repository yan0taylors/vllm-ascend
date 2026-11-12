# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UP and DOWN channels of the LWD edge-cloud data plane.

Each ``LwdConnectionId`` owns two HCCL communicators and, on its endpoint
ranks, two NPU streams. Every rank creates groups in connection lexicographic
order, UP before DOWN. Warmup then performs an independent bounded SEND/RECV
identity handshake per connection without a cross-edge/cloud world barrier
or all-reduce.

Design: docs/design/lwd_data_plane.md in the vLLM repository.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.config.lwd.topology import LwdConnectionId
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane.lwd_channel_probe import (
    LWD_PROBE_TIMEOUT_S,
    lwd_probe_frame,
    lwd_probe_verdict_frame,
    lwd_probe_verdict_verify,
    lwd_probe_verify,
    lwd_probe_wait,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LWD_DEFAULT_CONNECTION_ID,
    LwdChannelType,
)

_LWD_CHANNEL_GROUPS: dict[tuple[LwdConnectionId, LwdChannelType], tuple[dist.ProcessGroup, int]] = {}
_LWD_CHANNEL_STREAMS: dict[tuple[LwdConnectionId, LwdChannelType], torch.npu.Stream] = {}
_LWD_ENDPOINTS: tuple[LwdChannelEndpoints, ...] = ()
_LWD_CHANNELS_INITIALIZED = False
_LWD_CHANNEL_ORDER = (LwdChannelType.UP, LwdChannelType.DOWN)


def _lwd_ordered_endpoints(spec: LwdParallelGroupsSpec) -> tuple[LwdChannelEndpoints, ...]:
    endpoints = tuple(
        sorted(
            spec.channel_endpoints,
            key=lambda item: (
                item.connection_id.edge_id,
                item.connection_id.cloud_id,
                item.connection_id.dp_idx,
            ),
        )
    )
    seen: set[LwdConnectionId] = set()
    duplicates: set[LwdConnectionId] = set()
    for item in endpoints:
        if item.connection_id in seen:
            duplicates.add(item.connection_id)
        seen.add(item.connection_id)
    if duplicates:
        raise ValueError(f"[LWD] duplicate channel endpoint connection_id: {sorted(duplicates)}")
    return endpoints


def lwd_init_channels() -> None:
    """Create every connection's UP/DOWN channels in a world-consistent order.

    Group creation remains a world collective. Only endpoint ranks retain
    groups and streams, and only those ranks participate in warmup P2P.
    """
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    if _LWD_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    assert lwd_config is not None, "the data plane requires lwd_config"
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    ordered_endpoints = _lwd_ordered_endpoints(spec)
    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()
    for endpoints in ordered_endpoints:
        connection_id = endpoints.connection_id
        ranks = [endpoints.edge_rank, endpoints.cloud_rank]
        for channel in _LWD_CHANNEL_ORDER:
            group = dist.new_group(ranks, backend=backend)
            if my_rank in ranks:
                peer = endpoints.cloud_rank if my_rank == endpoints.edge_rank else endpoints.edge_rank
                key = (connection_id, channel)
                _LWD_CHANNEL_GROUPS[key] = (group, peer)
                _LWD_CHANNEL_STREAMS[key] = torch.npu.Stream()
    _LWD_ENDPOINTS = tuple(
        endpoints for endpoints in ordered_endpoints if my_rank in (endpoints.edge_rank, endpoints.cloud_rank)
    )
    # The cloud TP broadcast uses the framework device_group, whose creation
    # order and compute-stream FIFO ordering are already guaranteed.
    _LWD_CHANNELS_INITIALIZED = True
    logger.info(
        "[LWD] data-plane channels created for %s (my_rank=%d, endpoints=%s)",
        ordered_endpoints,
        my_rank,
        _LWD_ENDPOINTS,
    )
    lwd_warm_up_channels()


class _ProbeOutcome(Enum):
    OK = "ok"
    MISMATCH = "mismatch"  # Both P2Ps completed; later pairs stay aligned.
    TIMEOUT = "timeout"  # Unknown HCCL state; do not post more operations.


@dataclass(frozen=True)
class _ChannelProbeResult:
    outcome: _ProbeOutcome
    detail: str = ""


class _LwdProbeMismatch(RuntimeError):
    pass


class _LwdProbeTimeout(RuntimeError):
    pass


def _lwd_wait_for_probe(handle: object) -> None:
    """Observe actual P2P completion before reading the receive buffer."""
    if not lwd_probe_wait(handle):
        raise _LwdProbeTimeout("HCCL P2P completion was not observed")


def _lwd_exchange_probe_verdict(
    endpoints: LwdChannelEndpoints,
    phase: list[str],
    local_problem: str,
) -> None:
    """Exchange local outcomes so neither endpoint can half-start."""
    connection_id = endpoints.connection_id
    my_rank = dist.get_rank()
    up_group, up_peer = _LWD_CHANNEL_GROUPS[(connection_id, LwdChannelType.UP)]
    down_group, down_peer = _LWD_CHANNEL_GROUPS[(connection_id, LwdChannelType.DOWN)]
    if my_rank == endpoints.edge_rank:
        send_group, send_peer = up_group, up_peer
        recv_group, recv_peer = down_group, down_peer
    else:
        send_group, send_peer = down_group, down_peer
        recv_group, recv_peer = up_group, up_peer
    send_payload = lwd_probe_verdict_frame(not local_problem)
    recv_payload = torch.zeros_like(send_payload)
    phase[0] = "verdict SEND / RECV posting"
    send_handle = dist.isend(send_payload, dst=send_peer, group=send_group)
    recv_handle = dist.irecv(recv_payload, src=recv_peer, group=recv_group)
    phase[0] = "verdict SEND wait"
    _lwd_wait_for_probe(send_handle)
    phase[0] = "verdict RECV wait"
    _lwd_wait_for_probe(recv_handle)
    phase[0] = "peer verdict verification"
    peer_problem = lwd_probe_verdict_verify(recv_payload)
    problems = []
    if local_problem:
        problems.append(f"local {local_problem}")
    if peer_problem:
        problems.append(peer_problem)
    if problems:
        raise _LwdProbeMismatch("; ".join(problems))


def _lwd_probe_connection_impl(endpoints: LwdChannelEndpoints, phase: list[str]) -> None:
    """Exchange identities, then exchange both local verification verdicts."""
    connection_id = endpoints.connection_id
    my_rank = dist.get_rank()
    up_group, up_peer = _LWD_CHANNEL_GROUPS[(connection_id, LwdChannelType.UP)]
    down_group, down_peer = _LWD_CHANNEL_GROUPS[(connection_id, LwdChannelType.DOWN)]
    for channel, group in (
        (LwdChannelType.UP, up_group),
        (LwdChannelType.DOWN, down_group),
    ):
        backend = str(dist.get_backend(group))
        if backend != "hccl":
            raise RuntimeError(
                f"[connection={connection_id} channel={channel.value}] "
                f"channel group backend must be hccl, got {backend}"
            )
    if my_rank == endpoints.edge_rank:
        phase[0] = "UP SEND / DOWN RECV posting"
        up_payload = lwd_probe_frame(my_rank, connection_id, LwdChannelType.UP)
        down_payload = torch.zeros_like(lwd_probe_frame(endpoints.cloud_rank, connection_id, LwdChannelType.DOWN))
        up_handle = dist.isend(up_payload, dst=up_peer, group=up_group)
        down_handle = dist.irecv(down_payload, src=down_peer, group=down_group)
        phase[0] = "UP SEND wait"
        _lwd_wait_for_probe(up_handle)
        phase[0] = "DOWN RECV wait"
        _lwd_wait_for_probe(down_handle)
        phase[0] = "DOWN identity verification"
        problem = lwd_probe_verify(
            down_payload,
            endpoints.cloud_rank,
            connection_id,
            LwdChannelType.DOWN,
        )
        local_problem = (
            f"[connection={connection_id} channel=lwd_down peer={down_peer}] frame mismatch: {problem}"
            if problem
            else ""
        )
    elif my_rank == endpoints.cloud_rank:
        phase[0] = "UP RECV posting"
        up_payload = torch.zeros_like(lwd_probe_frame(endpoints.edge_rank, connection_id, LwdChannelType.UP))
        up_handle = dist.irecv(up_payload, src=up_peer, group=up_group)
        phase[0] = "UP RECV wait"
        _lwd_wait_for_probe(up_handle)
        phase[0] = "UP identity verification"
        problem = lwd_probe_verify(
            up_payload,
            endpoints.edge_rank,
            connection_id,
            LwdChannelType.UP,
        )
        local_problem = (
            f"[connection={connection_id} channel=lwd_up peer={up_peer}] frame mismatch: {problem}"
            if problem
            else ""
        )
        # Complete DOWN even after an UP mismatch. Both sides then finish the
        # identity pair before exchanging their local verdicts.
        phase[0] = "DOWN SEND posting"
        down_payload = lwd_probe_frame(my_rank, connection_id, LwdChannelType.DOWN)
        down_handle = dist.isend(down_payload, dst=down_peer, group=down_group)
        phase[0] = "DOWN SEND wait"
        _lwd_wait_for_probe(down_handle)
    else:
        raise RuntimeError(f"[LWD] rank {my_rank} is not an endpoint of {connection_id}")
    _lwd_exchange_probe_verdict(endpoints, phase, local_problem)


def _lwd_probe_connection(endpoints: LwdChannelEndpoints) -> _ChannelProbeResult:
    """Bound the complete handshake, including posting and verification D2H.

    A timed-out HCCL operation cannot be cancelled safely. The daemon thread
    is deliberately abandoned for process restart rather than reusing its
    communicator or posting traffic on another connection.
    """
    connection_id = endpoints.connection_id
    my_rank = dist.get_rank()
    peer = endpoints.cloud_rank if my_rank == endpoints.edge_rank else endpoints.edge_rank
    done = threading.Event()
    errors: list[Exception] = []
    phase = ["handshake setup"]
    device = torch.npu.current_device()

    def run_probe() -> None:
        try:
            # NPU device selection is thread-local; do not fall back to card 0.
            torch.npu.set_device(device)
            _lwd_probe_connection_impl(endpoints, phase)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(
        target=run_probe,
        name=f"lwd-probe-e{connection_id.edge_id}-c{connection_id.cloud_id}-d{connection_id.dp_idx}",
        daemon=True,
    )
    thread.start()
    if not done.wait(LWD_PROBE_TIMEOUT_S):
        return _ChannelProbeResult(
            _ProbeOutcome.TIMEOUT,
            f"[connection={connection_id} peer={peer} rank={my_rank} phase={phase[0]}] "
            f"probe TIMEOUT after {LWD_PROBE_TIMEOUT_S:g}s; restart required",
        )
    if errors:
        error = errors[0]
        if isinstance(error, _LwdProbeMismatch):
            return _ChannelProbeResult(_ProbeOutcome.MISMATCH, str(error))
        if isinstance(error, _LwdProbeTimeout):
            return _ChannelProbeResult(
                _ProbeOutcome.TIMEOUT,
                f"[connection={connection_id} peer={peer} rank={my_rank} phase={phase[0]}] {error}; restart required",
            )
        raise RuntimeError(
            f"[connection={connection_id} peer={peer} rank={my_rank} phase={phase[0]}] probe raised: {error!r}"
        ) from error
    return _ChannelProbeResult(_ProbeOutcome.OK)


def lwd_warm_up_channels() -> None:
    """Verify every local connection without any edge-cloud world collective.

    Identity mismatches are accumulated after their paired P2Ps complete. A
    timeout or unexpected exception stops all further posting because the
    communicator state is unknown. Non-endpoint ranks return immediately.
    """
    failures: list[str] = []
    aborted = False
    for index, endpoints in enumerate(_LWD_ENDPOINTS):
        try:
            result = _lwd_probe_connection(endpoints)
        except Exception as exc:
            aborted = True
            result = _ChannelProbeResult(_ProbeOutcome.TIMEOUT, str(exc) or repr(exc))
        if result.outcome is _ProbeOutcome.OK:
            logger.info(
                "[LWD] data-plane warmup done: connection=%s my_rank=%d",
                endpoints.connection_id,
                dist.get_rank(),
            )
            continue
        failures.append(result.detail)
        if result.outcome is _ProbeOutcome.TIMEOUT:
            aborted = True
        if aborted:
            skipped = [str(item.connection_id) for item in _LWD_ENDPOINTS[index + 1 :]]
            if skipped:
                logger.warning(
                    "[LWD] channel probe aborted; unprobed connections: %s",
                    ", ".join(skipped),
                )
            break
    if not failures:
        return
    detail = "; ".join(failures)
    logger.error("[LWD] channel probe failed: %s", detail)
    if not aborted:
        # All P2Ps completed, so healthy-group rollback cannot inherit a
        # wedged operation. Timeout/raise paths are left for process exit.
        lwd_destroy_channels()
    raise RuntimeError(f"[LWD] channel probe failed: {detail}")


def lwd_channels_initialized() -> bool:
    return _LWD_CHANNELS_INITIALIZED


def _lwd_channel(connection_id: LwdConnectionId, channel: LwdChannelType) -> tuple[dist.ProcessGroup, int]:
    try:
        return _LWD_CHANNEL_GROUPS[(connection_id, channel)]
    except KeyError:
        raise RuntimeError(
            f"[LWD] rank {dist.get_rank()} holds no {channel.value} channel of "
            f"connection {connection_id}; only its endpoint ranks do"
        ) from None


def lwd_get_channel_device_group(
    channel: LwdChannelType,
    connection_id: LwdConnectionId = LWD_DEFAULT_CONNECTION_ID,
) -> dist.ProcessGroup:
    group, _ = _lwd_channel(connection_id, channel)
    return group


def lwd_get_channel_peer(
    channel: LwdChannelType,
    connection_id: LwdConnectionId = LWD_DEFAULT_CONNECTION_ID,
) -> int:
    """Return this process's global P2P peer on the selected connection."""
    _, peer = _lwd_channel(connection_id, channel)
    return peer


def lwd_get_channel_stream(
    channel: LwdChannelType,
    connection_id: LwdConnectionId = LWD_DEFAULT_CONNECTION_ID,
) -> torch.npu.Stream:
    _lwd_channel(connection_id, channel)
    return _LWD_CHANNEL_STREAMS[(connection_id, channel)]


def lwd_destroy_channels() -> None:
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    for (connection_id, channel), (group, _) in list(_LWD_CHANNEL_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning(
                "[LWD] failed to destroy data-plane group for %s %s",
                connection_id,
                channel.value,
            )
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_ENDPOINTS = ()
    _LWD_CHANNELS_INITIALIZED = False
