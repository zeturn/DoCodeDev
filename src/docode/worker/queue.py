from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable


JobRunner = Callable[[str], Awaitable[None]]
logger = logging.getLogger(__name__)


class AsyncJobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    def start(self, runner: JobRunner) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run(runner))

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self, runner: JobRunner) -> None:
        while self._running:
            job_id = await self._queue.get()
            try:
                await runner(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("job runner failed", extra={"job_id": job_id})
            finally:
                self._queue.task_done()
