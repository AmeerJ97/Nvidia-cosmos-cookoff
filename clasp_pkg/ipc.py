"""
CLASP Agent IPC — Redis pub/sub message bus.
Provides broadcast, direct message, and request/reply patterns.
See docs/AGENT_IPC_ONBOARDING.md for usage.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Awaitable, Callable

import redis.asyncio as aioredis

log = logging.getLogger("clasp.ipc")

BROADCAST_CHANNEL = "agents:all"
AGENT_CHANNEL_PREFIX = "agent:"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class AgentBus:
    """
    Per-agent message bus handle.
    Obtained via setup_ipc("agent_name").
    """

    def __init__(self, name: str, redis: aioredis.Redis):
        self.name = name
        self._redis = redis
        self._pubsub: aioredis.client.PubSub | None = None
        self._pending_replies: dict[str, asyncio.Future] = {}

    # ── Sending ──────────────────────────────────────────────────────────────

    def _envelope(self, msg_type: str, payload: dict, reply_to: str | None = None) -> str:
        env = {
            "sender": self.name,
            "type": msg_type,
            "payload": payload,
            "ts": time.time(),
            "id": str(uuid.uuid4()),
        }
        if reply_to:
            env["reply_to"] = reply_to
        return json.dumps(env)

    async def broadcast(self, payload: dict) -> None:
        """Send a message to all agents on agents:all."""
        await self._redis.publish(BROADCAST_CHANNEL, self._envelope("broadcast", payload))

    async def send(self, target: str, payload: dict) -> None:
        """Send a direct message to a named agent."""
        channel = f"{AGENT_CHANNEL_PREFIX}{target}"
        await self._redis.publish(channel, self._envelope("direct", payload))

    async def reply(self, original_msg: dict, payload: dict) -> None:
        """Reply to a request message."""
        reply_to = original_msg.get("reply_to")
        if not reply_to:
            log.warning("reply() called but original message has no reply_to field")
            return
        channel = f"{AGENT_CHANNEL_PREFIX}{original_msg['sender']}"
        env = self._envelope("reply", payload)
        # Inject correlation id so requester can match
        env_dict = json.loads(env)
        env_dict["correlation_id"] = original_msg["id"]
        await self._redis.publish(channel, json.dumps(env_dict))

    async def request(
        self,
        target: str,
        payload: dict,
        timeout: float = 5.0,
    ) -> dict | None:
        """
        Send a request and wait for a reply.
        Returns the reply envelope or None on timeout.
        Requires listen_default() to be running.
        """
        msg_id = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_replies[msg_id] = fut

        channel = f"{AGENT_CHANNEL_PREFIX}{target}"
        env = {
            "sender": self.name,
            "type": "request",
            "payload": payload,
            "ts": time.time(),
            "id": msg_id,
            "reply_to": self.name,
        }
        await self._redis.publish(channel, json.dumps(env))

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("Request to %s timed out after %.1fs", target, timeout)
            return None
        finally:
            self._pending_replies.pop(msg_id, None)

    # ── Receiving ─────────────────────────────────────────────────────────────

    async def subscribe(
        self,
        channels: list[str],
        handler: Callable[[dict], Awaitable[None]],
    ) -> asyncio.Task:
        """Subscribe to specific channels and dispatch to handler in background."""
        ps = self._redis.pubsub()
        await ps.subscribe(*channels)

        async def _loop():
            async for raw in ps.listen():
                if raw["type"] != "message":
                    continue
                try:
                    msg = json.loads(raw["data"])
                except (json.JSONDecodeError, TypeError):
                    continue

                # Resolve pending request/reply futures
                if msg.get("type") == "reply" and "correlation_id" in msg:
                    fut = self._pending_replies.get(msg["correlation_id"])
                    if fut and not fut.done():
                        fut.set_result(msg)
                    continue

                try:
                    await handler(msg)
                except Exception as e:
                    log.error("IPC handler error: %s", e, exc_info=True)

        task = asyncio.create_task(_loop(), name=f"ipc-{self.name}")
        log.info("IPC agent '%s' subscribed to: %s", self.name, channels)
        return task

    async def listen_default(
        self,
        handler: Callable[[dict], Awaitable[None]],
    ) -> asyncio.Task:
        """Subscribe to broadcast + own channel."""
        channels = [BROADCAST_CHANNEL, f"{AGENT_CHANNEL_PREFIX}{self.name}"]
        return await self.subscribe(channels, handler)

    async def close(self) -> None:
        await self._redis.aclose()


# ── Factory ───────────────────────────────────────────────────────────────────

async def setup_ipc(
    agent_name: str,
    redis_url: str = DEFAULT_REDIS_URL,
) -> AgentBus:
    """Create and return a connected AgentBus for the named agent."""
    r = await aioredis.from_url(redis_url, decode_responses=True)
    bus = AgentBus(agent_name, r)
    # Announce presence
    await bus.broadcast({"status": "online", "agent": agent_name})
    log.info("IPC online: agent='%s' redis=%s", agent_name, redis_url)
    return bus
