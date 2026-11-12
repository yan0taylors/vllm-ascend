# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Event-protected rotating pinned buffers for LWD host transfers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class LwdPinnedStaging:
    """Rotate pinned slots without overwriting an in-flight H2D source.

    A slot event is recorded on the producing stream after every H2D. Reusing
    that slot waits for its previous copy, and growing or rebuilding the ring
    drains every old slot before releasing its storage. Depth can grow when a
    topology needs more in-flight windows; it never shrinks implicitly.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        min_capacity: int,
        depth: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self._dtype = dtype
        self._min_capacity = min_capacity
        self._depth = depth
        self._device = device
        self._pinned_slots: list[torch.Tensor] | None = None
        self._device_slots: list[torch.Tensor] | None = None
        self._slot_events: list[object | None] = []
        self._index = 0

    def drain(self) -> None:
        """Wait only for this ring's copy events, never for the whole device."""
        for index, event in enumerate(self._slot_events):
            if event is not None:
                event.synchronize()
                self._slot_events[index] = None

    def _ensure(self, n: int, min_depth: int | None = None) -> None:
        current_depth = len(self._pinned_slots or ())
        depth = max(self._depth, min_depth or 0, current_depth)
        slots = self._pinned_slots
        if slots is not None and slots[0].numel() >= n:
            if len(slots) < depth:
                self.drain()
                capacity = slots[0].numel()
                while len(slots) < depth:
                    slots.append(torch.empty(capacity, dtype=self._dtype, pin_memory=True))
                    if self._device_slots is not None:
                        self._device_slots.append(
                            torch.empty(capacity, dtype=self._dtype, device=self._device)
                        )
                    self._slot_events.append(None)
            return

        self.drain()
        capacity = max(n, self._min_capacity)
        self._pinned_slots = [
            torch.empty(capacity, dtype=self._dtype, pin_memory=True)
            for _ in range(depth)
        ]
        self._device_slots = (
            None
            if self._device is None
            else [
                torch.empty(capacity, dtype=self._dtype, device=self._device)
                for _ in range(depth)
            ]
        )
        self._slot_events = [None] * depth
        self._index = 0

    def _rotate(self) -> int:
        assert self._pinned_slots
        index = self._index
        self._index = (index + 1) % len(self._pinned_slots)
        return index

    def _wait_slot(self, index: int) -> None:
        previous = self._slot_events[index]
        if previous is not None:
            previous.synchronize()
            self._slot_events[index] = None

    def _record_h2d(self, index: int) -> None:
        device_type = getattr(self._device, "type", None)
        if device_type is None:
            device_type = str(self._device).split(":", 1)[0]
        accelerator = torch.npu if device_type == "npu" else torch.cuda
        event = accelerator.Event()
        event.record()
        self._slot_events[index] = event

    def stage_to_device(
        self,
        src: Sequence[int] | np.ndarray,
        n: int,
        min_depth: int | None = None,
    ) -> torch.Tensor:
        """Stage host values and issue an asynchronous, event-tracked H2D."""
        self._ensure(n, min_depth)
        index = self._rotate()
        self._wait_slot(index)
        assert self._pinned_slots is not None and self._device_slots is not None
        stage = self._pinned_slots[index]
        stage.numpy()[:n] = src
        device = self._device_slots[index]
        device[:n].copy_(stage[:n], non_blocking=True)
        self._record_h2d(index)
        return device[:n]

    def next_pinned(self, n: int, min_depth: int | None = None) -> torch.Tensor:
        """Return the next pinned slot after its previous consumer completes."""
        self._ensure(n, min_depth)
        index = self._rotate()
        self._wait_slot(index)
        assert self._pinned_slots is not None
        return self._pinned_slots[index]
