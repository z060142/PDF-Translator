"""Anthropic prompt caching: layered breakpoints and cache-aware request scheduling.

Every request is laid out as stable-to-volatile layers, each ending in a cache breakpoint:

    system prompt   - same for every page of the document (and every book of a batch)
    page context    - same for all batch / condense / fix requests of one page
    task            - unique per request, never cached

A write costs 1.25x (2x for 1h) and only pays off when the prefix is read again, so the page
context is marked only when the caller expects it to be shared (Prompt.reuse >= 2: several
batches of one page, several fix ranges) or when it is already warm (marking it is then a cheap
read). A page checked once is never written.

A cache entry becomes readable only once the response that writes it starts streaming, so
requests fired in parallel over the same new prefix would all pay the write premium and read
nothing. The coordinator fixes that: the first request over a layer that is not warm primes it
alone; requests sharing that layer wait until the primer's first stream event and then read
the cache. Warm layers are tracked by their TTL (every read refreshes it), and layers that
are too short to be cached are never gated.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field

TTL_SECONDS = {"5m": 300, "1h": 3600}
WRITE_MULT = {"5m": 1.25, "1h": 2.0}           # cache write price relative to base input
_DEFAULT_READ_MULT = 0.1                         # cache read price relative to base input
_READ_MULT = {                                   # models whose read price differs from 0.1x
    "claude-opus-5-5": 0.05,                     # $0.20 / $4 per MTok
    "claude-fable-5-1": 0.025, "claude-mythos-5-1": 0.025,
}
# minimum cacheable prefix (tokens); shorter prefixes are silently not cached
_MIN_TOKENS = {
    **dict.fromkeys(("claude-opus-5-5", "claude-opus-5", "claude-fable-5", "claude-fable-5-1",
                     "claude-mythos-5", "claude-mythos-5-1", "claude-sonnet-5-5", "claude-haiku-5-5"), 512),
    **dict.fromkeys(("claude-opus-4-8", "claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5",
                     "claude-opus-4-1", "claude-opus-4"), 1024),
    **dict.fromkeys(("claude-opus-4-7", "claude-haiku-3-5"), 2048),
    **dict.fromkeys(("claude-opus-4-6", "claude-opus-4-5", "claude-haiku-4-5"), 4096),
}
PRIME_WAIT = 90.0          # longest wait for another request to prime a layer
TTL_MARGIN = 20.0          # treat an entry as expired a little early
_CJK = re.compile(r"[⺀-鿿가-힯豈-﫿＀-￯]")


def read_mult(model: str) -> float:
    return _READ_MULT.get(model, _DEFAULT_READ_MULT)


def min_tokens(model: str) -> int:
    return _MIN_TOKENS.get(model, 1024)


def estimate_tokens(text: str) -> int:
    """Rough token count: ~1 per CJK character, ~3.5 characters per token otherwise."""
    cjk = len(_CJK.findall(text))
    return int(cjk + (len(text) - cjk) / 3.5)


def cache_control(ttl: str) -> dict | None:
    if ttl not in TTL_SECONDS:
        return None
    return {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": "1h"}


@dataclass
class _Layer:
    warm_until: float = 0.0
    priming: asyncio.Event | None = None
    uncacheable: bool = False


@dataclass
class Lease:
    """Layers of one request; `mine` are the layers this request primes."""
    keys: list[str]
    mine: list[str] = field(default_factory=list)
    sent: float = 0.0


class CacheCoordinator:
    """Per job (one event loop): shared by the translator and checker clients."""

    def __init__(self) -> None:
        self._layers: dict[str, _Layer] = {}
        self.waits = 0          # requests that waited for a primer instead of writing again

    def layers(self, model: str, signature: str, parts: list[str], overhead: int = 0,
               minimum: int | None = None) -> list[tuple[str, bool]]:
        """(cache key, long enough to be cached) of every layer, cumulative in render order.

        overhead: tokens the API adds in front (structured-output schema). The size test is
        liberal on purpose: gating a layer that turns out uncacheable costs one short wait (it is
        then marked uncacheable), missing a cacheable one costs a write per parallel request."""
        out: list[tuple[str, bool]] = []
        h = hashlib.sha1(f"{model}\x00{signature}".encode("utf-8"))
        total = overhead
        need = 0.8 * (minimum if minimum is not None else min_tokens(model))
        for part in parts:
            h.update(b"\x00" + part.encode("utf-8"))
            total += estimate_tokens(part)
            out.append((h.hexdigest(), total >= need))
        return out

    def is_warm(self, key: str) -> bool:
        layer = self._layers.get(key)
        return layer is not None and not layer.uncacheable and (
            layer.warm_until > time.monotonic() or layer.priming is not None)

    async def acquire(self, keys: list[str]) -> Lease:
        """Wait until every layer is warm or primed by this request."""
        lease = Lease(keys)
        waited = False
        for key in keys:
            layer = self._layers.setdefault(key, _Layer())
            while not (layer.uncacheable or layer.warm_until > time.monotonic()):
                if layer.priming is None:
                    layer.priming = asyncio.Event()
                    lease.mine.append(key)
                    break
                waited = True
                try:
                    await asyncio.wait_for(layer.priming.wait(), PRIME_WAIT)
                except asyncio.TimeoutError:
                    break
        if waited:
            self.waits += 1
        lease.sent = time.monotonic()
        return lease

    def started(self, lease: Lease, ttl: str, cached: bool, ttl_seconds: int | None = None) -> None:
        """The response started streaming: its prefix is now readable (or proved uncacheable)."""
        until = lease.sent + (ttl_seconds if ttl_seconds is not None else TTL_SECONDS.get(ttl, 0)) - TTL_MARGIN
        for key in lease.keys:
            layer = self._layers[key]
            if cached:
                layer.warm_until = max(layer.warm_until, until)
            else:
                layer.uncacheable = True
        self.release(lease)

    def release(self, lease: Lease) -> None:
        for key in lease.mine:
            layer = self._layers.get(key)
            if layer is not None and layer.priming is not None:
                layer.priming.set()
                layer.priming = None
        lease.mine = []
