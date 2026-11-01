"""GhostBarge event handling and cancellation tokens for real-time interruptions (USP 2).

Ensures that when a user speaks while the assistant is responding, ongoing LLM calls,
task executions, and audio playback are cancelled immediately within < 50ms.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger("phantomflow.ghostbarge")


class CancellationToken:
    """Async-aware cancellation token for halting active streaming tasks."""

    def __init__(self, turn_id: str):
        self.turn_id = turn_id
        self._is_cancelled = False
        self._cancellation_time_ns: Optional[int] = None
        self._reason: Optional[str] = None
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def cancel(self, reason: str = "ghostbarge_interruption") -> None:
        """Cancel this token."""
        if not self._is_cancelled:
            self._is_cancelled = True
            self._cancellation_time_ns = time.perf_counter_ns()
            self._reason = reason
            self._event.set()
            logger.info("Turn %s cancelled: %s", self.turn_id, reason)

    async def wait_for_cancellation(self) -> None:
        """Wait until token is cancelled."""
        await self._event.wait()


class GhostBargeController:
    """Coordinates real-time interruption events and task cancellation."""

    def __init__(self):
        self._tokens: Dict[str, CancellationToken] = {}
        self._active_asyncio_tasks: Dict[str, asyncio.Task] = {}

    def create_token(self, turn_id: str) -> CancellationToken:
        """Create and track a cancellation token for a turn."""
        token = CancellationToken(turn_id=turn_id)
        self._tokens[turn_id] = token
        return token

    def attach_task(self, turn_id: str, task: asyncio.Task) -> None:
        """Associate an asyncio task with the active turn for direct cancellation."""
        self._active_asyncio_tasks[turn_id] = task

    def interrupt_turn(self, turn_id: str, reason: str = "user_barge_in") -> bool:
        """Trigger immediate cancellation of an active turn."""
        token = self._tokens.get(turn_id)
        if token and not token.is_cancelled:
            token.cancel(reason=reason)

            # Cancel running asyncio task immediately if registered
            if task := self._active_asyncio_tasks.pop(turn_id, None):
                if not task.done():
                    task.cancel()
                    logger.info("Cancelled running asyncio task for turn %s", turn_id)
            return True
        return False

    def cleanup_turn(self, turn_id: str) -> None:
        """Cleanup completed turn references."""
        self._tokens.pop(turn_id, None)
        self._active_asyncio_tasks.pop(turn_id, None)
