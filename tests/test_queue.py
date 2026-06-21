from __future__ import annotations

import asyncio
import unittest

from docode.worker.queue import AsyncJobQueue


class AsyncJobQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_continues_after_runner_exception(self) -> None:
        queue = AsyncJobQueue()
        calls: list[str] = []
        completed = asyncio.Event()

        async def runner(job_id: str) -> None:
            calls.append(job_id)
            if job_id == "bad-job":
                raise RuntimeError("job failed unexpectedly")
            if job_id == "good-job":
                completed.set()

        queue.start(runner)
        try:
            with self.assertLogs("docode.worker.queue", level="ERROR") as logs:
                await queue.enqueue("bad-job")
                await queue.enqueue("good-job")

                await asyncio.wait_for(completed.wait(), timeout=1)
                await asyncio.wait_for(queue._queue.join(), timeout=1)
            self.assertTrue(any("job runner failed" in entry for entry in logs.output))
        finally:
            await queue.stop()

        self.assertEqual(calls, ["bad-job", "good-job"])


if __name__ == "__main__":
    unittest.main()
