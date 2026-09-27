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
from pymongo.asynchronous.helpers import _getaddrinfo
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
    async def _run_with_loop_monitor(self, awaitable):
        latencies = []

        # If the loop is being blocked, at least one iteration will have a
        # latency much more than 0.05 seconds.
        async def background_task():
            try:
                while True:
                    start = time.monotonic()
                    await asyncio.sleep(0.05)
                    latencies.append(time.monotonic() - start)
            except asyncio.CancelledError:
                latencies.append(time.monotonic() - start)
                raise

        task = asyncio.create_task(background_task())
        try:
            await awaitable
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        return latencies

    async def test_getaddrinfo_does_not_block_loop(self):
        response = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 27017))]

        def slow_getaddrinfo(*args, **kwargs):
            # Simulate a slow DNS resolver.  Must not block the event loop.
            time.sleep(0.75)
            return response

        with mock.patch("socket.getaddrinfo", side_effect=slow_getaddrinfo):
            latencies = await self._run_with_loop_monitor(
                _getaddrinfo("localhost", 27017, type=socket.SOCK_STREAM)
            )

        self.assertLessEqual(max(latencies), 0.5, "getaddrinfo blocked the event loop")

    async def test_getnameinfo_does_not_block_loop(self):
        def slow_getnameinfo(*args, **kwargs):
            # Simulate a slow reverse DNS lookup.  Must not block the event loop.
            time.sleep(0.75)
            return ("localhost", "0")

        with mock.patch("socket.getnameinfo", side_effect=slow_getnameinfo):
            latencies = await self._run_with_loop_monitor(_canonicalize_hostname("localhost", True))

        self.assertLessEqual(max(latencies), 0.5, "getnameinfo blocked the event loop")
