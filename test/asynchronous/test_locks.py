# Copyright 2024-present MongoDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lock.py"""

from __future__ import annotations

import asyncio
import sys
import unittest

from pymongo.lock import _ACondition, _async_create_condition, _async_create_lock

sys.path[0:0] = [""]


class TestACondition(unittest.IsolatedAsyncioTestCase):
    async def test_wait_notify(self):
        cond = _ACondition()
        result = []

        async def waiter():
            async with cond:
                if await cond.wait():
                    result.append(1)

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        self.assertFalse(cond.locked())

        async with cond:
            cond.notify()
        await asyncio.sleep(0)
        await task
        self.assertEqual([1], result)
        self.assertFalse(cond.locked())

    async def test_wait_timeout(self):
        cond = _ACondition()
        async with cond:
            self.assertFalse(await cond.wait(0.01))
            # The lock is re-acquired after a timeout.
            self.assertTrue(cond.locked())

    async def test_wait_unacquired(self):
        cond = _ACondition()
        with self.assertRaises(RuntimeError):
            await cond.wait()

    async def test_notify_unacquired(self):
        cond = _ACondition()
        with self.assertRaises(RuntimeError):
            cond.notify()

    async def test_context_manager(self):
        cond = _ACondition()
        self.assertFalse(cond.locked())
        async with cond:
            self.assertTrue(cond.locked())
        self.assertFalse(cond.locked())

    async def test_context_manager_releases_on_cancel(self):
        cond = _ACondition()

        async def waiter():
            async with cond:
                await cond.wait()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        self.assertFalse(cond.locked())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        # The cancelled waiter re-acquired the lock before propagating
        # CancelledError, and the context manager released it.
        self.assertFalse(cond.locked())
        self.assertFalse(cond._waiters)

        # The lock is usable by other tasks afterwards.
        async with cond:
            self.assertTrue(cond.locked())

    async def test_wait_cancel_reacquires_lock(self):
        # A cancelled wait() re-acquires the lock before raising, even
        # without a context manager.
        cond = _ACondition()
        await cond.acquire()

        wait = asyncio.create_task(cond.wait())
        await asyncio.sleep(0)
        self.assertFalse(cond.locked())
        wait.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await wait
        self.assertFalse(cond._waiters)
        # The cancelled task holds the lock again; release it.
        self.assertTrue(cond.locked())
        cond.release()
        self.assertFalse(cond.locked())

    async def test_cancelled_waiter_wakes_next(self):
        # A notification consumed by a cancelled waiter is passed on to
        # the next waiter instead of being lost.
        cond = _ACondition()
        result = []

        async def waiter(i):
            async with cond:
                if await cond.wait():
                    result.append(i)

        t1 = asyncio.create_task(waiter(1))
        t2 = asyncio.create_task(waiter(2))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(len(cond._waiters), 2)

        # Notify t1, but cancel it before it can re-acquire the lock.
        await cond.acquire()
        cond.notify(1)
        t1.cancel()
        cond.release()

        with self.assertRaises(asyncio.CancelledError):
            await t1
        # t1's cancellation wakes t2, which completes normally.
        await asyncio.wait_for(t2, 5)
        self.assertEqual([2], result)
        self.assertFalse(cond.locked())

    async def test_cancelled_while_reacquiring(self):
        # Cancellation delivered while re-acquiring the lock is retried
        # until the lock is held, then re-raised.
        cond = _ACondition()
        await cond.acquire()

        wait = asyncio.create_task(cond.wait())
        await asyncio.sleep(0)
        self.assertFalse(cond.locked())

        # Re-acquire so we can notify and contest the lock.
        await cond.acquire()
        cond.notify()
        await asyncio.sleep(0)
        # The waiter is now blocked re-acquiring the lock; cancel it there.
        wait.cancel()
        cond.release()
        with self.assertRaises(asyncio.CancelledError):
            await wait
        # The lock was still re-acquired before the CancelledError escaped.
        self.assertTrue(cond.locked())
        cond.release()

    async def test_notify_all_with_cancelled_waiters(self):
        cond = _ACondition()
        result = []

        async def waiter(i):
            async with cond:
                if await cond.wait():
                    result.append(i)

        tasks = [asyncio.create_task(waiter(i)) for i in range(3)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        tasks[0].cancel()
        with self.assertRaises(asyncio.CancelledError):
            await tasks[0]

        async with cond:
            cond.notify_all()
        await asyncio.gather(*tasks[1:])
        self.assertEqual(sorted(result), [1, 2])
        self.assertFalse(cond.locked())


if sys.version_info < (3, 13):
    # Tests adapted from: https://github.com/python/cpython/blob/v3.13.0rc2/Lib/test/test_asyncio/test_locks.py
    # Includes tests for:
    # - https://github.com/python/cpython/issues/111693
    # - https://github.com/python/cpython/issues/112202
    class TestConditionStdlib(unittest.IsolatedAsyncioTestCase):
        async def test_wait(self):
            cond = _async_create_condition(_async_create_lock())
            result = []

            async def c1(result):
                await cond.acquire()
                if await cond.wait():
                    result.append(1)
                return True

            async def c2(result):
                await cond.acquire()
                if await cond.wait():
                    result.append(2)
                return True

            async def c3(result):
                await cond.acquire()
                if await cond.wait():
                    result.append(3)
                return True

            t1 = asyncio.create_task(c1(result))
            t2 = asyncio.create_task(c2(result))
            t3 = asyncio.create_task(c3(result))

            await asyncio.sleep(0)
            self.assertEqual([], result)
            self.assertFalse(cond.locked())

            self.assertTrue(await cond.acquire())
            cond.notify()
            await asyncio.sleep(0)
            self.assertEqual([], result)
            self.assertTrue(cond.locked())

            cond.release()
            await asyncio.sleep(0)
            self.assertEqual([1], result)
            self.assertTrue(cond.locked())

            cond.notify(2)
            await asyncio.sleep(0)
            self.assertEqual([1], result)
            self.assertTrue(cond.locked())

            cond.release()
            await asyncio.sleep(0)
            self.assertEqual([1, 2], result)
            self.assertTrue(cond.locked())

            cond.release()
            await asyncio.sleep(0)
            self.assertEqual([1, 2, 3], result)
            self.assertTrue(cond.locked())

            self.assertTrue(t1.done())
            self.assertTrue(t1.result())
            self.assertTrue(t2.done())
            self.assertTrue(t2.result())
            self.assertTrue(t3.done())
            self.assertTrue(t3.result())

        async def test_wait_cancel(self):
            cond = _async_create_condition(_async_create_lock())
            await cond.acquire()

            wait = asyncio.create_task(cond.wait())
            asyncio.get_running_loop().call_soon(wait.cancel)
            with self.assertRaises(asyncio.CancelledError):
                await wait
            self.assertFalse(cond._waiters)
            self.assertTrue(cond.locked())

        async def test_wait_cancel_contested(self):
            cond = _async_create_condition(_async_create_lock())

            await cond.acquire()
            self.assertTrue(cond.locked())

            wait_task = asyncio.create_task(cond.wait())
            await asyncio.sleep(0)
            self.assertFalse(cond.locked())

            # Notify, but contest the lock before cancelling
            await cond.acquire()
            self.assertTrue(cond.locked())
            cond.notify()
            asyncio.get_running_loop().call_soon(wait_task.cancel)
            asyncio.get_running_loop().call_soon(cond.release)

            try:
                await wait_task
            except asyncio.CancelledError:
                # Should not happen, since no cancellation points
                pass

            self.assertTrue(cond.locked())

        async def test_wait_cancel_after_notify(self):
            # See bpo-32841
            waited = False

            cond = _async_create_condition(_async_create_lock())

            async def wait_on_cond():
                nonlocal waited
                async with cond:
                    waited = True  # Make sure this area was reached
                    await cond.wait()

            waiter = asyncio.create_task(wait_on_cond())
            await asyncio.sleep(0)  # Start waiting

            await cond.acquire()
            cond.notify()
            await asyncio.sleep(0)  # Get to acquire()
            waiter.cancel()
            await asyncio.sleep(0)  # Activate cancellation
            cond.release()
            await asyncio.sleep(0)  # Cancellation should occur

            self.assertTrue(waiter.cancelled())
            self.assertTrue(waited)

        async def test_wait_unacquired(self):
            cond = _async_create_condition(_async_create_lock())
            with self.assertRaises(RuntimeError):
                await cond.wait()

        async def test_wait_for(self):
            cond = _async_create_condition(_async_create_lock())
            presult = False

            def predicate():
                return presult

            result = []

            async def c1(result):
                await cond.acquire()
                if await cond.wait_for(predicate):
                    result.append(1)
                    cond.release()
                return True

            t = asyncio.create_task(c1(result))

            await asyncio.sleep(0)
            self.assertEqual([], result)

            await cond.acquire()
            cond.notify()
            cond.release()
            await asyncio.sleep(0)
            self.assertEqual([], result)

            presult = True
            await cond.acquire()
            cond.notify()
            cond.release()
            await asyncio.sleep(0)
            self.assertEqual([1], result)

            self.assertTrue(t.done())
            self.assertTrue(t.result())

        async def test_wait_for_unacquired(self):
            cond = _async_create_condition(_async_create_lock())

            # predicate can return true immediately
            res = await cond.wait_for(lambda: [1, 2, 3])
            self.assertEqual([1, 2, 3], res)

            with self.assertRaises(RuntimeError):
                await cond.wait_for(lambda: False)

        async def test_notify(self):
            cond = _async_create_condition(_async_create_lock())
            result = []

            async def c1(result):
                async with cond:
                    if await cond.wait():
                        result.append(1)
                    return True

            async def c2(result):
                async with cond:
                    if await cond.wait():
                        result.append(2)
                    return True

            async def c3(result):
                async with cond:
                    if await cond.wait():
                        result.append(3)
                    return True

            t1 = asyncio.create_task(c1(result))
            t2 = asyncio.create_task(c2(result))
            t3 = asyncio.create_task(c3(result))

            await asyncio.sleep(0)
            self.assertEqual([], result)

            async with cond:
                cond.notify(1)
            await asyncio.sleep(1)
            self.assertEqual([1], result)

            async with cond:
                cond.notify(1)
                cond.notify(2048)
            await asyncio.sleep(1)
            self.assertEqual([1, 2, 3], result)

            self.assertTrue(t1.done())
            self.assertTrue(t1.result())
            self.assertTrue(t2.done())
            self.assertTrue(t2.result())
            self.assertTrue(t3.done())
            self.assertTrue(t3.result())

        async def test_notify_all(self):
            cond = _async_create_condition(_async_create_lock())

            result = []

            async def c1(result):
                async with cond:
                    if await cond.wait():
                        result.append(1)
                    return True

            async def c2(result):
                async with cond:
                    if await cond.wait():
                        result.append(2)
                    return True

            t1 = asyncio.create_task(c1(result))
            t2 = asyncio.create_task(c2(result))

            await asyncio.sleep(0)
            self.assertEqual([], result)

            async with cond:
                cond.notify_all()
            await asyncio.sleep(1)
            self.assertEqual([1, 2], result)

            self.assertTrue(t1.done())
            self.assertTrue(t1.result())
            self.assertTrue(t2.done())
            self.assertTrue(t2.result())

        async def test_context_manager(self):
            cond = _async_create_condition(_async_create_lock())
            self.assertFalse(cond.locked())
            async with cond:
                self.assertTrue(cond.locked())
            self.assertFalse(cond.locked())

        async def test_timeout_in_block(self):
            condition = _async_create_condition(_async_create_lock())
            async with condition:
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(condition.wait(), timeout=0.5)

        @unittest.skipIf(
            sys.version_info < (3, 11), "raising the same cancelled error requires Python>=3.11"
        )
        async def test_cancelled_error_wakeup(self):
            # Test that a cancelled error, received when awaiting wakeup,
            # will be re-raised un-modified.
            wake = False
            raised = None
            cond = _async_create_condition(_async_create_lock())

            async def func():
                nonlocal raised
                async with cond:
                    with self.assertRaises(asyncio.CancelledError) as err:
                        await cond.wait_for(lambda: wake)
                    raised = err.exception
                    raise raised

            task = asyncio.create_task(func())
            await asyncio.sleep(0)
            # Task is waiting on the condition, cancel it there.
            task.cancel(msg="foo")  # type: ignore[call-arg]
            with self.assertRaises(asyncio.CancelledError) as err:
                await task
            self.assertEqual(err.exception.args, ("foo",))
            # We should have got the _same_ exception instance as the one
            # originally raised.
            self.assertIs(err.exception, raised)

        @unittest.skipIf(
            sys.version_info < (3, 11), "raising the same cancelled error requires Python>=3.11"
        )
        async def test_cancelled_error_re_aquire(self):
            # Test that a cancelled error, received when re-aquiring lock,
            # will be re-raised un-modified.
            wake = False
            raised = None
            cond = _async_create_condition(_async_create_lock())

            async def func():
                nonlocal raised
                async with cond:
                    with self.assertRaises(asyncio.CancelledError) as err:
                        await cond.wait_for(lambda: wake)
                    raised = err.exception
                    raise raised

            task = asyncio.create_task(func())
            await asyncio.sleep(0)
            # Task is waiting on the condition
            await cond.acquire()
            wake = True
            cond.notify()
            await asyncio.sleep(0)
            # Task is now trying to re-acquire the lock, cancel it there.
            task.cancel(msg="foo")  # type: ignore[call-arg]
            cond.release()
            with self.assertRaises(asyncio.CancelledError) as err:
                await task
            self.assertEqual(err.exception.args, ("foo",))
            # We should have got the _same_ exception instance as the one
            # originally raised.
            self.assertIs(err.exception, raised)

        @unittest.skipIf(sys.version_info < (3, 11), "asyncio.timeout requires Python>=3.11")
        async def test_cancelled_wakeup(self):
            # Test that a task cancelled at the "same" time as it is woken
            # up as part of a Condition.notify() does not result in a lost wakeup.
            # This test simulates a cancel while the target task is awaiting initial
            # wakeup on the wakeup queue.
            condition = _async_create_condition(_async_create_lock())
            state = 0

            async def consumer():
                nonlocal state
                async with condition:
                    while True:
                        await condition.wait_for(lambda: state != 0)
                        if state < 0:
                            return
                        state -= 1

            # create two consumers
            c = [asyncio.create_task(consumer()) for _ in range(2)]
            # wait for them to settle
            await asyncio.sleep(0.1)
            async with condition:
                # produce one item and wake up one
                state += 1
                condition.notify(1)

                # Cancel it while it is awaiting to be run.
                # This cancellation could come from the outside
                c[0].cancel()

                # now wait for the item to be consumed
                # if it doesn't means that our "notify" didn"t take hold.
                # because it raced with a cancel()
                try:
                    async with asyncio.timeout(1):
                        await condition.wait_for(lambda: state == 0)
                except TimeoutError:
                    pass
                self.assertEqual(state, 0)

                # clean up
                state = -1
                condition.notify_all()
            await c[1]

        @unittest.skipIf(sys.version_info < (3, 11), "asyncio.timeout requires Python>=3.11")
        async def test_cancelled_wakeup_relock(self):
            # Test that a task cancelled at the "same" time as it is woken
            # up as part of a Condition.notify() does not result in a lost wakeup.
            # This test simulates a cancel while the target task is acquiring the lock
            # again.
            condition = _async_create_condition(_async_create_lock())
            state = 0

            async def consumer():
                nonlocal state
                async with condition:
                    while True:
                        await condition.wait_for(lambda: state != 0)
                        if state < 0:
                            return
                        state -= 1

            # create two consumers
            c = [asyncio.create_task(consumer()) for _ in range(2)]
            # wait for them to settle
            await asyncio.sleep(0.1)
            async with condition:
                # produce one item and wake up one
                state += 1
                condition.notify(1)

                # now we sleep for a bit.  This allows the target task to wake up and
                # settle on re-aquiring the lock
                await asyncio.sleep(0)

                # Cancel it while awaiting the lock
                # This cancel could come the outside.
                c[0].cancel()

                # now wait for the item to be consumed
                # if it doesn't means that our "notify" didn"t take hold.
                # because it raced with a cancel()
                try:
                    async with asyncio.timeout(1):
                        await condition.wait_for(lambda: state == 0)
                except TimeoutError:
                    pass
                self.assertEqual(state, 0)

                # clean up
                state = -1
                condition.notify_all()
            await c[1]

    if __name__ == "__main__":
        unittest.main()
