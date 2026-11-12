# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD edge-cloud data plane.

Two physical channels per ``LwdConnectionId``, direction-only (no per-type
channels):

    UP   edge -> cloud : prompt embeddings of one prefill batch, plus an
                         optional mrope-positions aux frame
    DOWN cloud -> edge : c2e packet of one cloud step (final hidden +
                         topk + num_accepted)

The machinery (send snapshot, per-channel seqno reorder buffer, lazy
reap, HCCL->stream bridge) is ported from the demo branch's data plane
package (``lwd_comm`` there) with draft/plain wire variants removed: the main
payload is one contiguous bf16 tensor whose size the receiver learns from
the control plane before posting the irecv.

``lwd_channels`` owns the channel groups and streams; the rank layout comes
from ``vllm.distributed.lwd.LwdParallelGroupsSpec``. Design:
docs/design/lwd_data_plane.md in the vLLM repository.
"""

# TODO(lwd): switch the [lwd-comm] log and error prefix of this package to
# [LWD] once it is confirmed that no log analysis tool matches on it.
