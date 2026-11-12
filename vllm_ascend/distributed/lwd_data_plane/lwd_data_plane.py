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

from vllm.config.lwd.topology import LwdConnectionId
from vllm.logger import logger

from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_future import LwdTransferFuture
from vllm_ascend.distributed.lwd_data_plane.lwd_transfer_queue import LwdTransferQueue
from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LWD_DEFAULT_CONNECTION_ID,
    LwdChannelType,
    LwdTransferRequest,
)

_LwdQueueKey = tuple[LwdConnectionId, LwdChannelType, str]


class LwdDataPlane:
    """Process-wide singleton. All public methods are thread-safe."""

    _instance: LwdDataPlane | None = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> LwdDataPlane:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        # Each physical channel has independent send and receive seqno
        # streams. Connections never share FIFO state, even when dp_idx and
        # seqno are equal.
        self._queues: dict[_LwdQueueKey, LwdTransferQueue] = {}
        # A skip can arrive before the first request creates its queue. Keep
        # those holes outside the queue so queue creation cannot erase them.
        self._deferred_skips: dict[_LwdQueueKey, set[int]] = {}
        self._connection_errors: dict[LwdConnectionId, BaseException] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    # ------------------------------------------------------------------ #
    # Submission                                                          #
    # ------------------------------------------------------------------ #

    def submit_send(self, request: LwdTransferRequest) -> LwdTransferFuture:
        assert request.op == "send", "submit_send requires op='send'"
        return self._submit(request)

    def submit_recv(self, request: LwdTransferRequest) -> LwdTransferFuture:
        """Submit a recv. The irecv is posted immediately: early submission
        is a caller decision, not a separate API."""
        assert request.op == "recv", "submit_recv requires op='recv'"
        return self._submit(request)

    # ------------------------------------------------------------------ #
    # Completion driving                                                  #
    # ------------------------------------------------------------------ #

    def poll_completions(self) -> int:
        """Reap every channel once from an existing host loop head."""
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
        *,
        connection_id: LwdConnectionId = LWD_DEFAULT_CONNECTION_ID,
    ) -> None:
        """Skip one seqno only on the selected connection/channel/op FIFO.

        Both peers must skip the same seqno. If the queue does not exist yet,
        the hole is retained by the data plane and applied before the new
        queue becomes visible to submitters.
        """
        key = (connection_id, channel_type, op)
        with self._lock:
            queue = self._queues.get(key)
            if queue is None:
                self._deferred_skips.setdefault(key, set()).add(seqno)
                return
        queue.skip_seqno(seqno)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def abort_connection(self, connection_id: LwdConnectionId, error: BaseException) -> None:
        """Persist a terminal error and reject new transfers on one connection.

        Aborted queues stay registered for ``poll_completions``: posted futures
        report the error immediately, but their handles and send keepalives are
        released only after the device events complete. Other connections are
        independent. Recovery is a process-lifecycle decision.
        """
        with self._lock:
            self._connection_errors[connection_id] = error
            queues = [queue for key, queue in self._queues.items() if key[0] == connection_id]
        for queue in queues:
            queue.abort(error)
        logger.warning("[LWD] aborted data-plane connection %s: %r", connection_id, error)

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            queues = list(self._queues.values())
        try:
            for queue in queues:
                queue.shutdown(timeout=timeout)
        finally:
            with self._lock:
                self._queues.clear()
                self._deferred_skips.clear()
                self._connection_errors.clear()
                self._shutting_down = False
        logger.info("[lwd-comm] service shut down")

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _submit(self, request: LwdTransferRequest) -> LwdTransferFuture:
        key = (request.connection_id, request.channel, request.op)
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("lwd-comm service is shutting down")
            error = self._connection_errors.get(request.connection_id)
            if error is not None:
                raise RuntimeError(f"lwd-comm connection {request.connection_id} has failed: {error!r}") from error
            queue = self._queues.get(key)
            if queue is None:
                queue = LwdTransferQueue(request.connection_id, request.channel, request.op)
                # Apply pre-queue skips before publishing the queue. Another
                # thread therefore cannot observe an unhydrated FIFO.
                for skipped in sorted(self._deferred_skips.pop(key, set())):
                    queue.skip_seqno(skipped)
                self._queues[key] = queue
        return queue.submit(request)


def lwd_get_data_plane() -> LwdDataPlane:
    return LwdDataPlane.instance()
