# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LwdTransferFuture: completion handle for one data plane transfer.

Ported (simplified) from the demo branch's ``lwd_comm/future.py``:
pure-CPU ``done()`` via ``event.query()``, blocking ``wait()`` for the
debug/timeout path, in-place ``_bind`` for deferred (out-of-order
sequenced) futures, and exactly-once ``_finalize`` releasing the send
keepalive.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch

from vllm_ascend.distributed.lwd_data_plane.lwd_types import (
    LwdTransferRequest,
    LwdTransferResult,
    LwdTransferStatus,
)


class LwdTransferFuture:
    """Tracks one posted (or held) channel op.

    Completion means the cross-node P2P op has finished on the device,
    witnessed by ``done_event`` recorded on the channel stream right
    behind the bridged HCCL op.
    """

    def __init__(
        self,
        request: LwdTransferRequest,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: torch.npu.Event | None = None,
    ) -> None:
        self._request = request
        self._handles = handles
        self._done_event = done_event
        self._tensor = tensor
        self._aux_tensor = aux_tensor
        self._keepalive = keepalive
        self._staged_event = staged_event
        self._status = LwdTransferStatus.PENDING
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._done_cond = threading.Condition(self._lock)
        self._finalized = False

    # ------------------------------------------------------------------ #
    # Deferred (out-of-order sequenced) support                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def deferred(cls, request: LwdTransferRequest) -> LwdTransferFuture:
        """A future for a held request; bound in place when its seqno
        turn arrives."""
        return cls(request, [], None)

    def _bind(
        self,
        *,
        handles: list[Any],
        done_event: torch.npu.Event | None,
        tensor: torch.Tensor | None = None,
        keepalive: Any = None,
        aux_tensor: torch.Tensor | None = None,
        staged_event: torch.npu.Event | None = None,
    ) -> None:
        with self._done_cond:
            self._handles = handles
            self._done_event = done_event
            self._tensor = tensor
            self._aux_tensor = aux_tensor
            self._keepalive = keepalive
            self._staged_event = staged_event
            self._done_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Status                                                              #
    # ------------------------------------------------------------------ #

    @property
    def request(self) -> LwdTransferRequest:
        return self._request

    @property
    def _context(self) -> str:
        return f"connection={self._request.connection_id} channel={self._request.channel.value} op={self._request.op}"

    def done(self) -> bool:
        """Pure CPU query; never blocks, never touches the device."""
        with self._lock:
            if self._status is not LwdTransferStatus.PENDING:
                return True
            event = self._done_event
        if event is None:
            return False
        return self._query_completion(event)

    def _query_completion(self, event: torch.npu.Event) -> bool:
        try:
            if not event.query():
                return False
        except Exception as exc:  # device error surfaces as ERROR
            with self._lock:
                if self._status is LwdTransferStatus.PENDING:
                    self._status = LwdTransferStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True
        with self._lock:
            if self._status is LwdTransferStatus.PENDING:
                self._status = LwdTransferStatus.OK
            self._done_cond.notify_all()
        return True

    def _ready_to_reap(self) -> bool:
        """Return whether device-owned resources are safe to release."""
        with self._lock:
            event = self._done_event
            status = self._status
        if event is None:
            return status is not LwdTransferStatus.PENDING
        return self._query_completion(event)

    def _wait_for_device(self, timeout: float | None = None) -> None:
        """Wait for safe reclamation even after a terminal abort error."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._ready_to_reap():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"lwd-comm device wait timed out on {self._context} seqno={self._request.seqno}")
            with self._done_cond:
                self._done_cond.wait(timeout=min(0.001, remaining or 0.001))

    def wait(self, timeout: float | None = None) -> LwdTransferResult:
        """Block until completion (readiness-gate path)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._done_cond:
            while self._status is LwdTransferStatus.PENDING:
                if self._done_event is None:
                    # Deferred future not yet bound; wait for _bind.
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            f"lwd-comm wait timed out (unbound) on {self._context} seqno={self._request.seqno}"
                        )
                    self._done_cond.wait(timeout=remaining or 0.001)
                    continue
                if self._done_event.query():
                    self._status = LwdTransferStatus.OK
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"lwd-comm wait timed out on {self._context} seqno={self._request.seqno}")
                self._done_cond.wait(timeout=min(0.001, remaining or 0.001))
            if self._status is LwdTransferStatus.ERROR:
                raise RuntimeError(
                    f"lwd-comm op failed on {self._context} seqno={self._request.seqno}"
                ) from self._error
        return LwdTransferResult(
            status=self._status,
            tensor=self._tensor,
            aux_tensor=self._aux_tensor,
        )

    def result(self) -> LwdTransferResult:
        # TODO(lwd): confirm the behavior for a held future that is not bound
        # yet: it returns tensor=None, and wait_on_current_stream() is a no-op
        # before _bind.
        with self._lock:
            status, tensor, aux_tensor, error = (
                self._status,
                self._tensor,
                self._aux_tensor,
                self._error,
            )
        if status is LwdTransferStatus.ERROR:
            raise RuntimeError(f"lwd-comm op failed on {self._context} seqno={self._request.seqno}") from error
        return LwdTransferResult(status=status, tensor=tensor, aux_tensor=aux_tensor)

    # ------------------------------------------------------------------ #
    # Host staging                                                        #
    # ------------------------------------------------------------------ #

    def staged_done(self) -> bool:
        """Query D2H completion without blocking the communication thread."""
        event = self._staged_event
        if event is None:
            return True
        try:
            return bool(event.query())
        except Exception as exc:
            with self._lock:
                if self._status is not LwdTransferStatus.ERROR:
                    self._status = LwdTransferStatus.ERROR
                    self._error = exc
                    self._done_cond.notify_all()
            return True

    def release_device_tensor(self) -> None:
        """Release a staged receive's device slot, not a normal recv result."""
        if self._staged_event is None:
            return
        with self._done_cond:
            self._tensor = None

    # ------------------------------------------------------------------ #
    # Device-side ordering for consumers                                  #
    # ------------------------------------------------------------------ #

    def wait_on_current_stream(self) -> None:
        """Order the *current* stream after the channel-stream completion
        event (pure device-side; CPU returns immediately)."""
        event = self._done_event
        if event is not None:
            torch.npu.current_stream().wait_event(event)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def _fail(self, error: BaseException) -> None:
        """Report a terminal error without releasing in-flight device state."""
        with self._done_cond:
            if self._finalized:
                return
            self._status = LwdTransferStatus.ERROR
            self._error = error
            self._done_cond.notify_all()

    def _finalize(self, error: BaseException | None = None) -> None:
        """Exactly-once terminal bookkeeping: release the send keepalive.
        With ``error`` the future completes in ERROR state immediately
        (abort/skip path) so waiters fail promptly instead of timing out."""
        with self._done_cond:
            if self._finalized:
                return
            self._finalized = True
            self._keepalive = None
            self._handles = []
            if error is not None:
                self._status = LwdTransferStatus.ERROR
                self._error = error
                self._done_cond.notify_all()
                return
            if self._status is LwdTransferStatus.PENDING:
                self._status = (
                    LwdTransferStatus.OK
                    if (self._done_event is not None and self._done_event.query())
                    else self._status
                )
