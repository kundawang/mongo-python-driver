# Copyright 2022-present MongoDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal helpers for lock and condition coordination primitives."""

from __future__ import annotations

import asyncio
import collections
import os
import sys
import threading
import weakref
from asyncio import wait_for
from typing import Any, Optional, TypeVar

import pymongo._asyncio_lock

_HAS_REGISTER_AT_FORK = hasattr(os, "register_at_fork")

# References to instances of _create_lock
_forkable_locks: weakref.WeakSet[threading.Lock] = weakref.WeakSet()

_T = TypeVar("_T")

# Needed to support 3.13 asyncio fixes (https://github.com/python/cpython/issues/112202)
# in older versions of Python
if sys.version_info >= (3, 13):
    Lock = asyncio.Lock
    Condition = asyncio.Condition
else:
    Lock = pymongo._asyncio_lock.Lock
    Condition = pymongo._asyncio_lock.Condition


def _create_lock() -> threading.Lock:
    """Represents a lock that is tracked upon instantiation using a WeakSet and
    reset by pymongo upon forking.
    """
    lock = threading.Lock()
    if _HAS_REGISTER_AT_FORK:
        _forkable_locks.add(lock)
    return lock


def _async_create_lock() -> Lock:
    """Represents an asyncio.Lock."""
    return Lock()


def _create_condition(
    lock: threading.Lock, condition_class: Optional[Any] = None
) -> threading.Condition:
    """Represents a threading.Condition."""
    if condition_class:
        return condition_class(lock)
    return threading.Condition(lock)


def _async_create_condition(lock: Lock, condition_class: Optional[Any] = None) -> Condition:
    """Represents an asyncio.Condition."""
    if condition_class:
        return condition_class(lock)
    return Condition(lock)


class _Condition:
    """Synchronous condition variable used by the synchronous driver.

    Wraps :class:`threading.Condition` (or a compatible ``condition_class``,
    e.g. a gevent/eventlet condition) and mirrors the API of
    :class:`_ACondition` so that shared call sites can be converted with
    ``just synchro``.
    """

    def __init__(
        self, lock: Optional[threading.Lock] = None, condition_class: Optional[Any] = None
    ) -> None:
        if lock is None:
            lock = _create_lock()
        if condition_class is not None:
            self._cond: threading.Condition = condition_class(lock)
        else:
            self._cond = threading.Condition(lock)

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    def acquire(self) -> bool:
        """Acquire the underlying lock."""
        return self._cond.acquire()

    def release(self) -> None:
        """Release the underlying lock."""
        self._cond.release()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until notified or timed out.

        Returns True if notified, False on timeout.  The caller must hold
        the underlying lock; it is released while waiting and re-acquired
        before returning.
        """
        return self._cond.wait(timeout)

    def notify(self, n: int = 1) -> None:
        """Wake up to n waiters."""
        self._cond.notify(n)

    def notify_all(self) -> None:
        """Wake up all waiters."""
        self._cond.notify_all()


class _ACondition:
    """Asyncio condition variable with a cancellation-safe wait().

    Provides the same guarantees as :class:`asyncio.Condition` on Python
    3.13+ (see https://github.com/python/cpython/issues/112202), on all
    supported Python versions:

    - A task cancelled while waiting releases the underlying lock and wakes
      the next waiter before propagating CancelledError, so a notification
      is never swallowed by a cancelled waiter.
    - A woken task always re-acquires the underlying lock before returning
      or propagating an error, even if it is cancelled while re-acquiring.
    """

    def __init__(self, lock: Optional[Lock] = None, condition_class: Optional[Any] = None) -> None:
        # condition_class is accepted for API compatibility with _Condition
        # (used by the synchronous driver, e.g. for gevent); the async driver
        # does not support custom condition classes.
        if lock is None:
            lock = _async_create_lock()
        self._lock = lock
        self._waiters: collections.deque[asyncio.Future[None]] = collections.deque()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    def locked(self) -> bool:
        """Return True if the underlying lock is acquired."""
        return self._lock.locked()

    async def acquire(self) -> bool:
        """Acquire the underlying lock."""
        return await self._lock.acquire()

    def release(self) -> None:
        """Release the underlying lock."""
        self._lock.release()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until notified or timed out.

        Returns True if notified, False on timeout.  The caller must hold
        the underlying lock; it is released while waiting and always
        re-acquired before returning or raising, even on cancellation.
        """
        if not self.locked():
            raise RuntimeError("cannot wait on un-acquired lock")

        self.release()
        try:
            try:
                fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._waiters.append(fut)
                try:
                    if timeout is None:
                        await fut
                    else:
                        await wait_for(fut, timeout)
                    return True
                except asyncio.TimeoutError:
                    return False
                finally:
                    self._waiters.remove(fut)
            finally:
                # Must re-acquire the lock even if the wait was cancelled.
                # We only catch CancelledError here, since we don't want any
                # other (fatal) errors with the future to cause us to spin.
                err = None
                while True:
                    try:
                        await self.acquire()
                        break
                    except asyncio.CancelledError as e:
                        err = e

                if err is not None:
                    try:
                        raise err  # Re-raise most recent exception instance.
                    finally:
                        err = None  # Break reference cycles.
        except BaseException:
            # Any error raised out of here _may_ have occurred after this task
            # believed to have been successfully notified.  Make sure to
            # notify another task instead.  This may result in a "spurious
            # wakeup", which is allowed as part of the condition variable
            # protocol.
            self._notify(1)
            raise

    def notify(self, n: int = 1) -> None:
        """Wake up to n waiters.  The caller must hold the underlying lock."""
        if not self.locked():
            raise RuntimeError("cannot notify on un-acquired lock")
        self._notify(n)

    def _notify(self, n: int) -> None:
        idx = 0
        for fut in self._waiters:
            if idx >= n:
                break
            if not fut.done():
                idx += 1
                fut.set_result(None)

    def notify_all(self) -> None:
        """Wake up all waiters.  The caller must hold the underlying lock."""
        self.notify(len(self._waiters))


def _release_locks() -> None:
    # Completed the fork, reset all the locks in the child.
    for lock in _forkable_locks:
        if lock.locked():
            lock.release()


async def _async_cond_wait(condition: Condition, timeout: Optional[float]) -> bool:
    try:
        return await wait_for(condition.wait(), timeout)
    except asyncio.TimeoutError:
        return False


def _cond_wait(condition: threading.Condition, timeout: Optional[float]) -> bool:
    return condition.wait(timeout)
