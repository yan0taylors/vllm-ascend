# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared bring-up for the LWD edge/cloud workers (worker layer only).

Both subclasses are selected by ``platform.py`` only for LWD deployments.
They diverge after ``init_device``: the edge hijacks ``execute_model`` for
embed/unembed direct calls (no runner forward, no KV cache), the cloud
delegates to a swapped-in ``LwdCloudModelRunner`` and owns the UP/DOWN
data plane. This base only owns what the two workers used to duplicate
verbatim: the data-plane channel handshake (after the native distributed
bring-up, which routes itself to the LWD layout when lwd_config is
present) plus the shared data-plane handle.
"""

from __future__ import annotations

from vllm_ascend.distributed.lwd_data_plane.lwd_data_plane import (
    lwd_get_data_plane,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_payload_scrambler import (
    lwd_get_payload_scrambler,
)
from vllm_ascend.worker.worker import NPUWorker


class LwdWorkerBase(NPUWorker):
    """Common bring-up shared by :class:`LwdEdgeWorker` / :class:`LwdCloudWorker`."""

    def init_device(self) -> None:
        super().init_device()
        self._init_payload_scrambler()
        # Shared handle to the LWD data-plane singleton; both workers talk
        # to the UP/DOWN channels through it.
        self.comm_service = lwd_get_data_plane()

    def _init_payload_scrambler(self) -> None:
        """Bind the process-wide scrambler when feature_ctrl enables it."""
        lwd_config = getattr(self.vllm_config, "lwd_config", None)
        if lwd_config is None:
            return
        features = lwd_config.topology.feature_ctrl
        scramble = getattr(features, "scramble", None)
        if scramble is None or not scramble.enable:
            return
        if not scramble.device_id or not scramble.custom_factor:
            raise ValueError("[LWD] feature_ctrl.scramble.enable requires both device_id and custom_factor")
        lwd_get_payload_scrambler(
            device_id=scramble.device_id,
            custom_factor=scramble.custom_factor,
        )

    def _init_lwd_data_plane(self) -> None:
        # The data-plane channels MUST be built here, never in ``__init__``.
        # ``lwd_init_channels`` creates the HCCL process groups of every DP
        # connection (``dist.new_group``) and then warms each pair with a
        # bounded, two-sided identity handshake. Group creation still needs
        # the distributed environment built inside ``init_device``; warmup
        # itself deliberately has no cross-edge/cloud world barrier or
        # all-reduce. The executor constructs the worker first and calls
        # ``init_device`` afterwards, so running this from ``__init__`` would
        # trip the "group is not initialized" assertions. Subclasses call
        # this at exactly the position they previously built channels inline.
        #
        # Ascend-side MC2 groups and the vllm-side TP/EP groups need no
        # override here: native ``init_ascend_model_parallel`` routes to
        # the LWD layout itself when ``lwd_config`` is present.
        from vllm_ascend.distributed.lwd_data_plane.lwd_channels import (
            lwd_init_channels,
        )

        lwd_init_channels()
