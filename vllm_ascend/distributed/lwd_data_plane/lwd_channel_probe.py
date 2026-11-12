# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Identity probe exchanged by each connection's UP/DOWN warmup handshake.

Both peers independently derive a frame from the sender's global rank, the
full ``LwdConnectionId`` and the channel direction. The receiver compares
every field, so a wrong peer, crossed connection, swapped direction or
corrupted buffer fails during startup rather than on the first business
payload. Int32 fields retain large topology IDs exactly; the previous bf16
frame could only represent IDs below 256.

This module is deliberately a leaf beside ``lwd_payload_scrambler``. It reads
no configuration and performs no world collective. Corruption of business
payloads still needs runtime checks; payload geometry and large-message
acceptance are covered separately.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
from vllm.config.lwd.topology import LwdConnectionId

from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

__all__ = [
    "LWD_PROBE_TIMEOUT_S",
    "lwd_probe_frame",
    "lwd_probe_verdict_frame",
    "lwd_probe_verdict_verify",
    "lwd_probe_verify",
    "lwd_probe_wait",
]

# One bounded budget covers posting, all identity/verdict P2P operations and
# verification D2H for a connection. The caller reports the current phase on timeout.
LWD_PROBE_TIMEOUT_S = 30.0
_LWD_PROBE_POLL_INTERVAL_S = 0.02

_LWD_PROBE_FRAME_NUM_ELEMENTS = 12
_LWD_PROBE_FIELD_LIMIT = 2**31 - 1
_LWD_PROBE_MAGIC = 90
_LWD_PROBE_CHECKSUM_MOD = 2**31 - 1
_LWD_PROBE_VERDICT_FRAME_NUM_ELEMENTS = 3
_LWD_PROBE_VERDICT_MAGIC = 91
_LWD_PROBE_VERDICT_FAILED = 0
_LWD_PROBE_VERDICT_OK = 1

_LWD_PROBE_FIELD_NAMES = {
    0: "not a probe frame / recv not written",
    1: "peer rank mismatch",
    2: "edge_id mismatch (crossed connection)",
    3: "cloud_id mismatch (crossed connection)",
    4: "dp_idx mismatch (crossed connection)",
    5: "channel direction mismatch",
    6: "frame length mismatch",
    7: "peer rank checksum mismatch",
    8: "edge_id checksum mismatch",
    9: "cloud_id checksum mismatch",
    10: "dp_idx checksum mismatch",
    11: "identity/direction checksum mismatch",
}


def _lwd_probe_field(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < _LWD_PROBE_FIELD_LIMIT:
        raise ValueError(f"[LWD] probe {name}={value!r} must be an integer in [0, 2^31-1)")
    return value


def lwd_probe_frame(
    sender_rank: int,
    connection_id: LwdConnectionId,
    channel: LwdChannelType,
    device: str | torch.device = "npu",
) -> torch.Tensor:
    """Build the identity frame from facts independently known by both peers."""
    if not isinstance(channel, LwdChannelType):
        raise ValueError(f"[LWD] invalid probe channel: {channel!r}")
    rank = _lwd_probe_field(sender_rank, "sender_rank")
    edge = _lwd_probe_field(connection_id.edge_id, "edge_id")
    cloud = _lwd_probe_field(connection_id.cloud_id, "cloud_id")
    dp = _lwd_probe_field(connection_id.dp_idx, "dp_idx")
    direction = 0 if channel is LwdChannelType.UP else 1
    values = (
        _LWD_PROBE_MAGIC,
        rank,
        edge,
        cloud,
        dp,
        direction,
        _LWD_PROBE_FRAME_NUM_ELEMENTS,
        (rank + 5) % _LWD_PROBE_CHECKSUM_MOD,
        (edge * 7 + 11) % _LWD_PROBE_CHECKSUM_MOD,
        (cloud * 13 + 17) % _LWD_PROBE_CHECKSUM_MOD,
        (dp * 19 + 23) % _LWD_PROBE_CHECKSUM_MOD,
        (direction * 29 + rank * 31 + edge * 37 + cloud * 41 + dp * 43 + 47) % _LWD_PROBE_CHECKSUM_MOD,
    )
    return torch.tensor(values, dtype=torch.int32).to(device)


def _lwd_probe_verdict_checksum(status: int) -> int:
    return (_LWD_PROBE_VERDICT_MAGIC * 3 + status * 37 + 59) % _LWD_PROBE_CHECKSUM_MOD


def lwd_probe_verdict_frame(local_ok: bool, device: str | torch.device = "npu") -> torch.Tensor:
    """Encode the sender's identity-check result for the second handshake round."""
    status = _LWD_PROBE_VERDICT_OK if local_ok else _LWD_PROBE_VERDICT_FAILED
    values = (_LWD_PROBE_VERDICT_MAGIC, status, _lwd_probe_verdict_checksum(status))
    return torch.tensor(values, dtype=torch.int32).to(device)


def lwd_probe_verdict_verify(got: torch.Tensor) -> str:
    """Return an empty string for peer success, or the peer/verdict failure."""
    if got.numel() != _LWD_PROBE_VERDICT_FRAME_NUM_ELEMENTS:
        return (
            f"peer verdict frame length mismatch: got {got.numel()} elements, "
            f"expected {_LWD_PROBE_VERDICT_FRAME_NUM_ELEMENTS}"
        )
    if got.dtype != torch.int32:
        return f"peer verdict frame dtype mismatch: got {got.dtype}, expected torch.int32"
    magic, status, checksum = _lwd_probe_values(got)
    if magic != _LWD_PROBE_VERDICT_MAGIC:
        return f"peer verdict frame magic mismatch: got {magic}"
    if status not in (_LWD_PROBE_VERDICT_FAILED, _LWD_PROBE_VERDICT_OK):
        return f"peer verdict status mismatch: got {status}"
    expected_checksum = _lwd_probe_verdict_checksum(status)
    if checksum != expected_checksum:
        return f"peer verdict checksum mismatch: got {checksum}, expected {expected_checksum}"
    if status == _LWD_PROBE_VERDICT_FAILED:
        return "peer reported identity mismatch"
    return ""


def lwd_probe_wait(handle: Any, timeout_s: float = LWD_PROBE_TIMEOUT_S) -> bool:
    """Return false on timeout, or wait to surface a completed op's error."""
    deadline = time.monotonic() + timeout_s
    while not handle.is_completed():
        if time.monotonic() >= deadline:
            return False
        time.sleep(_LWD_PROBE_POLL_INTERVAL_S)
    handle.wait()
    return True


def _lwd_probe_values(frame: torch.Tensor) -> list[int]:
    """Read integer fields without lossy bf16/float32 conversion."""
    values: list[int] = []
    for value in frame.reshape(-1).cpu().tolist():
        if isinstance(value, int):
            values.append(value)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            values.append(-1)
            continue
        values.append(int(numeric) if math.isfinite(numeric) and numeric.is_integer() else -1)
    return values


def lwd_probe_verify(
    got: torch.Tensor,
    peer: int,
    connection_id: LwdConnectionId,
    channel: LwdChannelType,
) -> str:
    """Compare every identity field; return an empty string on success."""
    if got.numel() != _LWD_PROBE_FRAME_NUM_ELEMENTS:
        return f"frame length mismatch: got {got.numel()} elements, expected {_LWD_PROBE_FRAME_NUM_ELEMENTS}"
    if got.dtype != torch.int32:
        return f"frame dtype mismatch: got {got.dtype}, expected torch.int32"
    expected = lwd_probe_frame(peer, connection_id, channel, device=str(got.device))
    got_values = _lwd_probe_values(got)
    expected_values = _lwd_probe_values(expected)
    problems = [
        f"[{index}] {_LWD_PROBE_FIELD_NAMES[index]}: got {got_values[index]}, expected {expected_values[index]}"
        for index in range(_LWD_PROBE_FRAME_NUM_ELEMENTS)
        if got_values[index] != expected_values[index]
    ]
    return "; ".join(problems)
