# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Rotating pinned staging buffers shared by the LWD edge/cloud host transfers.

Consolidates the hand-rolled rotation loops that used to live in
``LwdEdgeWorker._lwd_embed_ids_h2d`` and ``LwdCloudModelRunner
._lwd_h2d_stage_i64`` (pinned + device mirror pairs). Rotation depth 4
covers batch_queue depth 2 plus one step of engine lag, so a later step
cannot overwrite a buffer whose transfer is still queued.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


class LwdPinnedStaging:
    """Pinned staging slots rotated per use, with an optional device mirror.

    ``stage_to_device`` stages host data into the next pinned slot and
    issues a ``non_blocking`` H2D into the mirrored device slot, returning
    a ``[:n]`` device view; ``next_pinned`` rotates and returns the full
    pinned slot for the caller to fill and slice. Capacity grows (never
    shrinks) to ``max(n, min_capacity)`` on first use or overflow,
    rebuilding every slot and restarting rotation at slot 0.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        min_capacity: int,
        depth: int = 4,
        device: str | torch.device | None = None,
    ) -> None:
        self._dtype = dtype
        self._min_capacity = min_capacity
        self._depth = depth
        self._device = device
        self._pinned_slots: list[torch.Tensor] | None = None
        self._device_slots: list[torch.Tensor] | None = None
        self._slot_events: list[object | None] = []  # 【MR2】每个 pinned 槽一个 H2D 完成 event，防止槽被下次 copy 覆盖时旧 DMA 还没读完
        self._index = 0

    def _ensure(self, n: int, min_depth: int | None = None) -> None:  # 【MR2】扩容入口增加 min_depth：调用方可以要求至少多深的槽位
        depth = max(self._depth, min_depth or 0)  # 【MR2】实际深度取默认深度和调用方要求的较大值
        if self._pinned_slots is not None and self._pinned_slots[0].numel() >= n:
            while len(self._pinned_slots) < depth:  # 【MR2】容量够时只按需补齐槽位数量
                capacity = self._pinned_slots[0].numel()
                self._pinned_slots.append(torch.empty(capacity, dtype=self._dtype, pin_memory=True))  # 【MR2】新增一个 pinned host 槽
                if self._device_slots is not None:
                    self._device_slots.append(torch.empty(capacity, dtype=self._dtype, device=self._device))  # 【MR2】如果启用 device 槽，同步新增对应的 device 槽
                self._slot_events.append(None)  # 【MR2】新槽还没有在飞 copy，event 置空
            return
        for event in self._slot_events:  # 【MR2】重建缓冲区前，先等待所有旧槽上的 copy 完成
            if event is not None:
                event.synchronize()  # 【MR2】同步旧 event，确保 DMA 不再读旧 pinned 内存
        capacity = max(n, self._min_capacity)
        self._pinned_slots = [torch.empty(capacity, dtype=self._dtype, pin_memory=True) for _ in range(depth)]  # 【MR2】按新容量重建 pinned 槽列表
        self._device_slots = (
            None
            if self._device is None
            else [torch.empty(capacity, dtype=self._dtype, device=self._device) for _ in range(depth)]  # 【MR2】重建 device 槽列表
        )
        self._slot_events = [None] * depth  # 【MR2】event 列表重置为与新槽数量一致
        self._index = 0

    def _rotate(self) -> int:
        index = self._index
        self._index = (index + 1) % len(self._pinned_slots)  # 【MR2】槽位轮转：轮流使用，避免刚写入的槽立刻被覆盖
        return index

    def stage_to_device(  # 【MR2】把数据搬到 pinned 再异步 H2D，返回 device 视图；支持指定最小深度
        self,
        src: Sequence[int] | np.ndarray,
        n: int,
        min_depth: int | None = None,
    ) -> torch.Tensor:
        """Stage ``src`` and issue an async H2D; return the ``[:n]`` device view.

        Same-stream ordering guarantees the consumer kernel reads this
        step's values (H2D enqueued before the consumer), matching the
        retired inline staging loops.
        """
        self._ensure(n, min_depth)  # 【MR2】先保证容量和槽深满足本次写入
        index = self._rotate()
        previous = self._slot_events[index]  # 【MR2】取本槽上一次 H2D 的完成 event
        if previous is not None:
            previous.synchronize()  # 【MR2】上一次 copy 没完成前不能覆盖该 pinned 槽
        stage = self._pinned_slots[index]
        stage.numpy()[:n] = src
        device = self._device_slots[index]
        device[:n].copy_(stage[:n], non_blocking=True)
        accelerator = torch.npu if str(self._device).startswith("npu") else torch.cuda  # 【MR2】按设备类型选择 NPU 或 CUDA 的 Event
        event = accelerator.Event()  # 【MR2】创建完成事件
        event.record()  # 【MR2】在 H2D copy 入队后记录事件
        self._slot_events[index] = event  # 【MR2】事件挂到槽上，下一轮写该槽前等待
        return device[:n]

    def next_pinned(self, n: int) -> torch.Tensor:
        """Rotate and return the next full pinned slot (caller fills/slices)."""
        self._ensure(n)
        return self._pinned_slots[self._rotate()]
