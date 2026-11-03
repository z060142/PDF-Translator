"""Economy mode: Anthropic requests sent through the Message Batches API (50% of the price).

Callers simply `await submit(params)` and get the Message back, so the pipeline (translate ->
condense -> check -> fix) runs unchanged; each of its rounds just becomes a batch. Requests are
collected until no new one arrives for IDLE seconds (or MAX_WAIT has passed since the first),
then submitted together; the batch is polled until it ends and every caller's future is resolved
from the results. Batches still running when the job stops are cancelled, so requests that have
not been processed yet are not billed.
"""
from __future__ import annotations

import asyncio
from typing import Callable

import anthropic


class BatchItemError(Exception):
    """A request inside a batch that ended as errored / canceled / expired."""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind          # API error type ("invalid_request_error", ...) or "canceled" / "expired"


class MessageBatcher:
    IDLE = 3.0              # submit once no new request arrived for this long
    MAX_WAIT = 30.0         # ... or at the latest this long after the first queued request
    MAX_REQUESTS = 10000    # per batch (API limit: 100,000 requests / 256 MB)
    POLL = 20.0             # seconds between status checks
    REPORT_EVERY = 6        # polls between progress log lines

    def __init__(self, client: anthropic.AsyncAnthropic, log: Callable[[str, str], None] | None = None):
        self._client = client
        self._log = log or (lambda msg, level="info": None)
        self._pending: list[tuple[str, dict, asyncio.Future]] = []
        self._first = self._last = 0.0
        self._seq = 0
        self._flusher: asyncio.Task | None = None
        self._runs: set[asyncio.Task] = set()
        self._active: set[str] = set()

    async def submit(self, params: dict):
        """Queue one Messages request; returns its Message when the batch has ended."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._seq += 1
        now = loop.time()
        if not self._pending:
            self._first = now
        self._last = now
        self._pending.append((f"r{self._seq}", params, fut))
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.create_task(self._flush_loop())
        return await fut

    async def _flush_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._pending:
            now = loop.time()
            if (len(self._pending) >= self.MAX_REQUESTS or now - self._last >= self.IDLE
                    or now - self._first >= self.MAX_WAIT):
                items = self._pending[:self.MAX_REQUESTS]
                del self._pending[:self.MAX_REQUESTS]
                self._first = now
                items = [it for it in items if not it[2].done()]   # callers that were cancelled meanwhile
                if items:
                    task = asyncio.create_task(self._run(items))
                    self._runs.add(task)
                    task.add_done_callback(self._runs.discard)
            else:
                await asyncio.sleep(0.5)

    async def _run(self, items: list[tuple[str, dict, asyncio.Future]]) -> None:
        futs = {cid: fut for cid, _, fut in items}

        def fail_all(exc: BaseException) -> None:
            for f in futs.values():
                if not f.done():
                    f.set_exception(exc)

        try:
            batch = await self._client.messages.batches.create(
                requests=[{"custom_id": cid, "params": params} for cid, params, _ in items])
        except Exception as e:
            fail_all(e)
            return
        bid = batch.id
        self._active.add(bid)
        self._log(f"已送出批次 {bid}（{len(items)} 個請求，5 折計價），等待結果…", "info")
        try:
            polls = errors = 0
            while batch.processing_status != "ended":
                await asyncio.sleep(self.POLL)
                try:
                    batch = await self._client.messages.batches.retrieve(bid)
                    errors = 0
                except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) as e:
                    errors += 1
                    if errors >= 15:
                        raise
                    self._log(f"查詢批次 {bid} 失敗（{type(e).__name__}），稍後重試", "warn")
                    continue
                polls += 1
                if polls % self.REPORT_EVERY == 0 and batch.processing_status != "ended":
                    c = batch.request_counts
                    self._log(f"批次 {bid} 處理中：完成 {c.succeeded + c.errored}/{len(items)}", "info")
            async for r in await self._client.messages.batches.results(bid):
                fut = futs.get(r.custom_id)
                if fut is None or fut.done():
                    continue
                res = r.result
                if res.type == "succeeded":
                    fut.set_result(res.message)
                elif res.type == "errored":
                    err = res.error.error
                    fut.set_exception(BatchItemError(getattr(err, "type", "api_error"), getattr(err, "message", "")))
                else:
                    fut.set_exception(BatchItemError(res.type, "batch request was not processed"))
            fail_all(BatchItemError("missing", "result not found in batch output"))
        except asyncio.CancelledError:
            for f in futs.values():
                f.cancel()
            raise
        except Exception as e:
            fail_all(e)
        finally:
            self._active.discard(bid)

    async def close(self) -> None:
        """Cancel unfinished batches (unprocessed requests are not billed) and pending work."""
        for bid in list(self._active):              # server side first: the run tasks clear _active
            try:
                await self._client.messages.batches.cancel(bid)
                self._log(f"已取消未完成的批次 {bid}", "warn")
            except Exception:
                pass
        for f in (self._flusher, *self._runs):
            if f is not None and not f.done():
                f.cancel()
        self._active.clear()
        for _, _, fut in self._pending:
            if not fut.done():
                fut.cancel()
        self._pending.clear()
