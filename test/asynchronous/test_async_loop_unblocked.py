# Copyright 2025-present MongoDB, Inc.
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

"""Test that the asynchronous API does not block the event loop."""

from __future__ import annotations

import asyncio
import socket
import time
from unittest import mock

from pymongo.asynchronous.auth import _canonicalize_hostname
from pymongo.asynchronous.helpers import _getaddrinfo, _getnameinfo
from pymongo.errors import ServerSelectionTimeoutError
from test.asynchronous import AsyncIntegrationTest, AsyncUnitTest


class TestClientLoopUnblocked(AsyncIntegrationTest):
    async def test_client_does_not_block_loop(self):
        # Use an unreachable TEST-NET host to ensure that the client times out attempting to create a connection.
        client = self.simple_client("192.0.2.1", serverSelectionTimeoutMS=500)
        latencies = []

        # If the loop is being blocked, at least one iteration will have a latency much more than 0.1 seconds
        async def background_task():
            start = time.monotonic()
            try:
                while True:
                    start = time.monotonic()
                    await asyncio.sleep(0.1)
                    latencies.append(time.monotonic() - start)
            except asyncio.CancelledError:
                latencies.append(time.monotonic() - start)
                raise

        t = asyncio.create_task(background_task())

        with self.assertRaisesRegex(ServerSelectionTimeoutError, "No servers found yet"):
            await client.admin.command("ping")

        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t

        self.assertLessEqual(
            sorted(latencies, reverse=True)[0],
            1.0,
            "Background task was blocked from running",
        )


class TestDNSResolutionLoopUnblocked(AsyncUnitTest):
    """DNS resolution must be offloaded to an executor and not block the event loop."""

    # How long the patched blocking socket calls sleep.
    RESOLUTION_DELAY = 0.5
    # Maximum tolerable latency for a background task while resolution runs.
    MAX_LATENCY = 0.4

    async def assert_loop_unblocked(self, awaitable):
        """Run awaitable while a background task ticks; fail if the loop stalls."""
        ticks = []

        async def background_task():
            try:
                while True:
                    await asyncio.sleep(0.05)
                    ticks.append(time.monotonic())
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(background_task())
        try:
            result = await awaitable
            # Give the background task a chance to observe any stall that
            # happened while the awaitable was running.
            await asyncio.sleep(0.2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertGreaterEqual(len(ticks), 2, "Background task never ran")
        gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:])]
        self.assertLessEqual(
            max(gaps),
            self.MAX_LATENCY,
            "Background task was blocked from running during DNS resolution",
        )
        return result

    def blocking_getaddrinfo(self, host, port, *args, **kwargs):
        time.sleep(self.RESOLUTION_DELAY)
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "canonical.example.com",
                ("127.0.0.1", 27017),
            )
        ]

    def blocking_getnameinfo(self, sockaddr, flags):
        time.sleep(self.RESOLUTION_DELAY)
        return ("reverse.example.com", "27017")

    async def test_getaddrinfo_does_not_block_loop(self):
        with mock.patch("socket.getaddrinfo", side_effect=self.blocking_getaddrinfo):
            result = await self.assert_loop_unblocked(
                _getaddrinfo("localhost", 27017, family=socket.AF_INET, type=socket.SOCK_STREAM)
            )
        self.assertEqual(result[0][4], ("127.0.0.1", 27017))

    async def test_getnameinfo_does_not_block_loop(self):
        with mock.patch("socket.getnameinfo", side_effect=self.blocking_getnameinfo):
            result = await self.assert_loop_unblocked(
                _getnameinfo(("127.0.0.1", 27017), socket.NI_NAMEREQD)
            )
        self.assertEqual(result, ("reverse.example.com", "27017"))

    async def test_canonicalize_hostname_does_not_block_loop(self):
        with mock.patch("socket.getaddrinfo", side_effect=self.blocking_getaddrinfo):
            with mock.patch("socket.getnameinfo", side_effect=self.blocking_getnameinfo):
                result = await self.assert_loop_unblocked(
                    _canonicalize_hostname("example.com", "forwardAndReverse")
                )
        self.assertEqual(result, "reverse.example.com")
