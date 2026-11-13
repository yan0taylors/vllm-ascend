# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdDataPlane: singleton entry point of the LWD data plane.

Compute side only ever calls ``submit_send`` / ``submit_recv`` and
consumes ``LwdTransferFuture`` notifications.  ``poll_completions`` is meant
to be invoked from an existing host loop head (worker busy loop);
it drives keepalive reclamation via lazy reap.

Ported (simplified) from the demo branch's ``lwd_comm/service.py``.
"""

from __future__ import annotations

import threading

from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import LwdTransferQueue
from vllm_ascend.distributed.lwd_data_plane.lwd_types import ConnectionId, LwdChannelType, LwdTransferRequest


class LwdDataPlane:
    """Process-wide singleton.  All public methods are thread-safe."""

    _instance: LwdDataPlane | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> LwdDataPlane:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        # HCCL ordering is per physical (channel, direction): send and
        # recv have INDEPENDENT seqno streams (each peer's send order must
        # match the other side's recv-post order), so the FIFO key is
        # (connection, channel, op) — merging directions would make a
        # process's sends and recvs on one channel collide on a shared
        # seqno counter, and merging connections would collide their
        # independent seqno streams.
        self._queues: dict[tuple[ConnectionId, LwdChannelType, str], LwdTransferQueue] = {}
        self._pending_skips: dict[tuple[ConnectionId, LwdChannelType, str], set[int]] = {}
        self._lock = threading.Lock()
        self._shutting_down = False
        self._closed = False
        self._failed_connections: set[ConnectionId] = set()

    # ------------------------------------------------------------------ #
    # Submission                                                          #
    # ------------------------------------------------------------------ #

    def submit_send(self, request: LwdTransferRequest) -> LwdTransferFuture:
        assert request.op == "send", "submit_send requires op='send'"
        return self._submit(request)

    def submit_recv(self, request: LwdTransferRequest) -> LwdTransferFuture:
        """Submit a recv.  The irecv is posted immediately — "early" is a
        property of when you submit, not a separate API."""
        assert request.op == "recv", "submit_recv requires op='recv'"
        return self._submit(request)

    # ------------------------------------------------------------------ #
    # Completion driving                                                  #
    # ------------------------------------------------------------------ #

    def poll_completions(self) -> int:
        """Reap every channel once.  Call from an existing loop head;
        costs one head-of-line ``event.query()`` per channel with
        pending traffic."""
        # TODO(lwd): confirm whether the edge should reap too. On the mm branch
        # only the cloud worker calls this; submit() reaps its own queue only,
        # so an edge UP send snapshot (one prefill batch) stays alive until the
        # next UP send.
        with self._lock:
            queues = list(self._queues.values())
        completed = 0
        for queue in queues:
            completed += len(queue.reap())
        return completed

    def skip_seqno(
        self,
        channel_type: LwdChannelType,
        seqno: int,
        op: str = "recv",
        dp_idx: int = 0,
        edge_id: int = 0,
        cloud_id: int = 0,
        connection_id: ConnectionId | None = None,
    ) -> None:
        """Mark a seqno as never-to-arrive on the given connection FIFO.
        passes op="recv"; the sender-side abort glue passes op="send".  Both
        peers must skip the same seqnos."""
        connection = connection_id or ConnectionId(edge_id, cloud_id, dp_idx)
        key = (connection, channel_type, op)
        with self._lock:
            queue = self._queues.get(key)
            if queue is None:
                # Abort control can arrive before the first data submission.
                # Keep the hole until this FIFO is created instead of losing it.
                self._pending_skips.setdefault(key, set()).add(seqno)
                return
        queue.skip_seqno(seqno)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def abort_connection(self, connection: ConnectionId, error: BaseException) -> None:
        """Isolate one failed endpoint pair without disturbing other FIFOs."""
        with self._lock:
            self._failed_connections.add(connection)
            keys = [key for key in self._queues if key[0] == connection]
            queues = [self._queues.pop(key) for key in keys]
            for key in list(self._pending_skips):
                if key[0] == connection:
                    self._pending_skips.pop(key, None)
        for queue in queues:
            queue.abort(error)
        logger.warning("[LWD] aborted data-plane connection %s: %r", connection, error)

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            self._closed = True
            queues = list(self._queues.values())
        try:
            for queue in queues:
                queue.shutdown(timeout=timeout)
        finally:
            with self._lock:
                self._queues.clear()
                self._pending_skips.clear()
                self._failed_connections.clear()
                self._shutting_down = False
        logger.info("[lwd-comm] service shut down")

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _submit(self, request: LwdTransferRequest) -> LwdTransferFuture:
        with self._lock:
            if self._shutting_down or self._closed:
                raise RuntimeError("lwd-comm service is shut down")
            if request.connection in self._failed_connections:
                raise RuntimeError(f"lwd-comm connection {request.connection} has failed")
            key = (request.connection, request.channel, request.op)
            queue = self._queues.get(key)
            if queue is None:
                queue = LwdTransferQueue(request.connection, request.channel, request.op)
                self._queues[key] = queue
                for seqno in sorted(self._pending_skips.pop(key, ())):
                    queue.skip_seqno(seqno)
        return queue.submit(request)


def lwd_get_data_plane() -> LwdDataPlane:
    return LwdDataPlane.instance()
