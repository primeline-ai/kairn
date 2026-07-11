"""Async event bus for Kairn.

Honesty note (weakness-audit rank 95): no in-tree code registers a listener
as of v0.2 - the ~21 .emit() call sites across the engines are extension
points, and every emission is an awaited no-op until a subscriber calls
.on()/.on_all() (tests and external embedders do). Do not assume any side
effect happens via events today.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from kairn.events.types import EventType

logger = logging.getLogger(__name__)

Listener = Callable[[EventType, dict[str, Any]], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async pub/sub event bus."""

    def __init__(self) -> None:
        self._listeners: dict[EventType, list[Listener]] = defaultdict(list)
        self._global_listeners: list[Listener] = []

    def on(self, event_type: EventType, listener: Listener) -> None:
        """Register a listener for a specific event type."""
        self._listeners[event_type].append(listener)

    def on_all(self, listener: Listener) -> None:
        """Register a listener for all events."""
        self._global_listeners.append(listener)

    def off(self, event_type: EventType, listener: Listener) -> None:
        """Remove a listener."""
        if listener in self._listeners[event_type]:
            self._listeners[event_type].remove(listener)

    async def emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """Emit an event to all registered listeners."""
        data = data or {}
        listeners = self._listeners.get(event_type, []) + self._global_listeners

        for listener in listeners:
            try:
                await listener(event_type, data)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in event listener for %s", event_type)

    def clear(self) -> None:
        """Remove all listeners."""
        self._listeners.clear()
        self._global_listeners.clear()
