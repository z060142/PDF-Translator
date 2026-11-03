"""Unified async JSON-completion client for Anthropic and OpenAI-compatible providers."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import anthropic
import openai

from .config import FALLBACK_MODELS, ModelConfig
from .msgbatch import BatchItemError, MessageBatcher
from .oa_cache import dialect_for, parse_usage, request_extras
from .prompt_cache import WRITE_MULT, CacheCoordinator, cache_control, estimate_tokens, read_mult

BATCH_DISCOUNT = 0.5          # Message Batches API price factor


class LLMError(Exception):
    """Error after which retrying the same request is pointless (bad config, refusal...)."""

    def __init__(self, msg: str, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal  # True -> configuration problem, job should pause


class RetryableError(Exception):
    pass


@dataclass
class Usage:
    input_tokens: int = 0          # uncached input (full price)
    output_tokens: int = 0
    cache_read_tokens: int = 0
    requests: int = 0
    cache_write_tokens: int = 0
    billed_input: float = 0.0      # input at list price in base-input token units (writes / reads weighted)
    paid_input: float = 0.0        # billed_input after the batch discount
    paid_output: float = 0.0       # output tokens after the batch discount
    batch_requests: int = 0

    def add(self, other: "Usage") -> None:
        for k, v in vars(other).items():
            setattr(self, k, getattr(self, k) + v)


def cache_saving(u: dict) -> float | None:
    """Share of the input cost saved by caching (None when nothing was cached)."""
    total = u.get("input_tokens", 0) + u.get("cache_write_tokens", 0) + u.get("cache_read_tokens", 0)
    if not total or not (u.get("cache_read_tokens") or u.get("cache_write_tokens")):
        return None
    return 1 - u.get("billed_input", total) / total


@dataclass
class Prompt:
    system: str           # stable for the whole document          -> cache layer 1
    context: str          # stable for one page (full page text)     -> cache layer 2; "" = none
    task: str             # request-specific part, never cached
    schema: dict = field(default_factory=dict)
    reuse: int = 1        # requests expected to share this context soon (>= 2 -> worth a cache write)
    cache_group: str = "" # requests that share a prefix (the page); routing key on OpenAI


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise RetryableError(f"模型輸出不是有效 JSON: {text[:200]!r}")


def _schema_overhead(schema: dict) -> int:
    """Tokens structured outputs add to the cached prefix (measured: ~200 + the schema itself)."""
    return 200 + estimate_tokens(json.dumps(schema)) if schema else 0


class LLMClient:
    def __init__(self, cfg: ModelConfig, cache_ttl: str = "5m", cache: CacheCoordinator | None = None,
                 batch: bool = False, log: Callable[[str, str], None] | None = None, role: str = ""):
        self.cfg = cfg
        self.cache = cache
        self.role = role
        self._log = log or (lambda msg, level="info": None)
        self._oa_cache_ok = True            # cleared if the endpoint rejects the caching fields
        self._oa_stream_ok = True           # cleared if the endpoint rejects streaming usage
        key = os.environ.get(cfg.key_env(), "")
        if not key:
            raise LLMError(f"環境變數 {cfg.key_env()} 未設定", fatal=True)
        self.batcher: MessageBatcher | None = None
        if cfg.provider == "anthropic":
            self._a = anthropic.AsyncAnthropic(api_key=key, max_retries=2)
            if batch:
                self.batcher = MessageBatcher(self._a, log)
                # requests of a batch run in any order and the batch may take longer than 5 minutes
                if cache_ttl == "5m":
                    cache_ttl = "1h"
        elif cfg.provider == "openai":
            self._o = openai.AsyncOpenAI(api_key=key, base_url=cfg.base_url.strip() or None, max_retries=2)
        else:
            raise LLMError(f"未知的 provider: {cfg.provider}", fatal=True)
        self.cache_ttl = cache_ttl          # "5m" | "1h" | "off" (Anthropic only)

    async def close(self) -> None:
        try:
            if self.batcher is not None:
                await self.batcher.close()
            if self.cfg.provider == "anthropic":
                await self._a.close()
            else:
                await self._o.close()
        except Exception:
            pass

    async def complete_json(self, p: Prompt) -> tuple[dict, Usage]:
        try:
            if self.cfg.provider == "anthropic":
                return await self._anthropic(p)
            return await self._openai(p)
        except BatchItemError as e:
            if e.kind in ("invalid_request_error", "authentication_error", "permission_error", "not_found_error"):
                raise LLMError(f"批次請求被拒絕（{e}）", fatal=True) from e
            raise RetryableError(f"批次請求失敗（{e}）") from e
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise LLMError(f"{type(e).__name__}: {e}", fatal=True) from e
        except (anthropic.BadRequestError, openai.BadRequestError) as e:
            raise LLMError(f"請求參數錯誤 (可能是模型不支援此思考程度/參數): {e}", fatal=True) from e
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError,
                openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        except (anthropic.APIStatusError, openai.APIStatusError) as e:
            if getattr(e, "status_code", 0) >= 500:
                raise RetryableError(f"{type(e).__name__}: {e}") from e
            raise LLMError(f"{type(e).__name__}: {e}") from e

    # ---------------- Anthropic ----------------
    async def _anthropic(self, p: Prompt) -> tuple[dict, Usage]:
        cfg = self.cfg
        ttl = self.cache_ttl
        cc = cache_control(ttl)
        output_config: dict = {}
        if p.schema:
            output_config["format"] = {"type": "json_schema", "schema": p.schema}
        thinking = None
        if cfg.effort and cfg.effort != "default":
            output_config["effort"] = cfg.effort
            thinking = {"type": "adaptive"}

        # cache layers: system (shared by every page) -> page context (shared within a page).
        # A write only pays off if the prefix is read again: the page context is marked only when
        # several requests will share it, or when it is already warm (marking it is then a read).
        has_context = bool(p.context)
        mark_system = bool(cc)
        mark_context = bool(cc) and has_context and p.reuse >= 2
        gate: list[str] = []
        if cc and self.cache is not None:
            # thinking / effort / output format are part of the cached prefix
            signature = json.dumps({"thinking": thinking, "output_config": output_config}, sort_keys=True)
            parts = [p.system] + ([p.context] if has_context else [])
            layers = self.cache.layers(cfg.model, signature, parts, _schema_overhead(p.schema))
            if has_context:
                mark_context = mark_context or self.cache.is_warm(layers[1][0])
            marked = [True, mark_context]
            gate = [k for (k, ok), m in zip(layers, marked) if ok and m]

        def block(text: str, marked: bool) -> dict:
            return {"type": "text", "text": text, **({"cache_control": cc} if marked else {})}

        # stable -> volatile: [system | bp] [page context | bp] [task]
        content = ([block(p.context, mark_context)] if has_context else []) + [block(p.task, False)]
        kwargs: dict = dict(
            model=cfg.model,
            max_tokens=int(cfg.max_tokens),
            system=[block(p.system, mark_system)],
            messages=[{"role": "user", "content": content}],
        )
        if thinking:
            kwargs["thinking"] = thinking
        if output_config:
            kwargs["output_config"] = output_config

        if self.batcher is not None:
            msg = await self.batcher.submit(kwargs)          # no streaming, no gating, no fallbacks
        else:
            msg = await self._stream(kwargs, gate, ttl)

        usage = self._anthropic_usage(msg.usage, ttl, batch=self.batcher is not None)
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise LLMError(f"模型拒絕回應 (refusal, category={cat})")
        if msg.stop_reason == "max_tokens":
            raise RetryableError("輸出被 max_tokens 截斷，請調高 max_tokens 或降低批次大小")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return _extract_json(text), usage

    async def _stream(self, kwargs: dict, gate: list[str], ttl: str):
        """Streaming request; with a gate, waits for / primes the cache layers (see prompt_cache)."""
        cfg = self.cfg
        if cfg.refusal_fallback and cfg.model in FALLBACK_MODELS:
            manager = self._a.beta.messages.stream(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            manager = self._a.messages.stream(**kwargs)
        lease = await self.cache.acquire(gate) if gate else None
        try:
            async with manager as stream:
                try:
                    first = await stream.__anext__()     # message_start: prefill done, cache readable
                except StopAsyncIteration:
                    raise RetryableError("API 回應串流為空") from None
                if lease is not None:
                    fu = getattr(getattr(first, "message", None), "usage", None)
                    cached = fu is None or bool((getattr(fu, "cache_creation_input_tokens", 0) or 0)
                                                + (getattr(fu, "cache_read_input_tokens", 0) or 0))
                    self.cache.started(lease, ttl, cached)
                return await stream.get_final_message()
        finally:
            if lease is not None:
                self.cache.release(lease)               # no-op unless the request failed early

    def _anthropic_usage(self, u, ttl: str, batch: bool = False) -> Usage:
        """Token usage with the input weighted by its actual price (writes 1.25x / 2x, reads ~0.1x)."""
        inp = u.input_tokens or 0
        out = u.output_tokens or 0
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        write = getattr(u, "cache_creation_input_tokens", 0) or 0
        detail = getattr(u, "cache_creation", None)
        if detail is not None:
            w5 = getattr(detail, "ephemeral_5m_input_tokens", 0) or 0
            w1 = getattr(detail, "ephemeral_1h_input_tokens", 0) or 0
        else:
            w5, w1 = (0, write) if ttl == "1h" else (write, 0)
        billed = inp + w5 * WRITE_MULT["5m"] + w1 * WRITE_MULT["1h"] + read * read_mult(self.cfg.model)
        f = BATCH_DISCOUNT if batch else 1.0
        return Usage(input_tokens=inp, output_tokens=out, cache_read_tokens=read, requests=1,
                     cache_write_tokens=write, billed_input=billed, paid_input=billed * f,
                     paid_output=out * f, batch_requests=int(batch))

    # ---------------- OpenAI-compatible ----------------
    async def _openai(self, p: Prompt) -> tuple[dict, Usage]:
        cfg = self.cfg
        ttl = self.cache_ttl
        d = dialect_for(cfg.base_url, cfg.model, ttl)
        caching = ttl != "off" and self._oa_cache_ok
        system = p.system
        if p.schema:
            system += ("\n\nRespond with a single JSON object matching this JSON schema, no other text:\n"
                       + json.dumps(p.schema, ensure_ascii=False))
        has_context = bool(p.context)
        bp = d.breakpoint(ttl) if (caching and d.marker) else None
        # same write policy as the Anthropic path: the page context only when it will be shared
        mark_context = has_context and p.reuse >= 2
        gate: list[str] = []
        if caching and self.cache is not None:
            signature = json.dumps({"effort": cfg.effort, "json": cfg.json_mode}, sort_keys=True)
            layers = self.cache.layers(cfg.model, signature, [system] + ([p.context] if has_context else []),
                                       minimum=d.min_tokens)
            if has_context:
                mark_context = mark_context or self.cache.is_warm(layers[1][0])
            if d.gate and self._oa_stream_ok:
                gate = [k for (k, ok), m in zip(layers, [True, mark_context]) if ok and m]

        def part(text: str, marked: bool) -> dict:
            return {"type": "text", "text": text, **({d.marker: bp} if (bp and marked) else {})}

        if bp:      # explicit breakpoints need content parts
            messages = [{"role": "system", "content": [part(system, True)]},
                        {"role": "user", "content": ([part(p.context, mark_context)] if has_context else [])
                         + [part(p.task, False)]}]
        else:       # plain strings: accepted by every OpenAI-compatible server
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": "\n\n".join(x for x in (p.context, p.task) if x)}]
        kwargs: dict = dict(model=cfg.model, messages=messages)
        if cfg.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if cfg.effort and cfg.effort != "default":
            kwargs["reasoning_effort"] = cfg.effort
        if cfg.max_tokens:
            kwargs["max_completion_tokens"] = int(cfg.max_tokens)
        extras = request_extras(d, ttl, cfg.model, system, p.cache_group, self.role) if self._oa_cache_ok else {}
        if extras:
            kwargs["extra_body"] = extras

        try:
            text, finish, u = await self._openai_send(kwargs, gate, d)
        except openai.BadRequestError as e:
            msg = str(e).lower()
            if self._oa_cache_ok and ("prompt_cache" in msg or "session_id" in msg or "cache_control" in msg):
                # the endpoint does not know a caching field: drop caching for this client and retry
                self._oa_cache_ok = False
                self._log(f"[{self.role}] 端點不支援快取參數，改為不帶快取參數送出：{e}", "warn")
                return await self._openai(p)
            if self._oa_stream_ok and gate and ("stream" in msg):
                self._oa_stream_ok = False
                return await self._openai(p)
            raise
        uncached, read, write, out = parse_usage(u) if u is not None else (0, 0, 0, 0)
        billed = uncached + write * d.write_mult + read * d.read_mult
        usage = Usage(input_tokens=uncached, output_tokens=out, cache_read_tokens=read, requests=1,
                      cache_write_tokens=write, billed_input=billed, paid_input=billed, paid_output=out)
        if finish == "length":
            raise RetryableError("輸出被長度上限截斷，請調高 max_tokens 或降低批次大小")
        return _extract_json(text), usage

    async def _openai_send(self, kwargs: dict, gate: list[str], d) -> tuple[str, str | None, object]:
        """Returns (text, finish_reason, usage). With a gate, streams so that the first token
        (prefill done, prefix cached) can release the requests waiting for this prefix."""
        if not gate:
            resp = await self._o.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            return choice.message.content or "", choice.finish_reason, resp.usage
        lease = await self.cache.acquire(gate)
        try:
            stream = await self._o.chat.completions.create(
                **kwargs, stream=True, stream_options={"include_usage": True})
            parts: list[str] = []
            finish = usage = None
            primed = False
            async for chunk in stream:
                if not primed:                                   # first token: prefix is cached
                    self.cache.started(lease, "", True, d.ttl_seconds)
                    primed = True
                if chunk.choices:
                    c = chunk.choices[0]
                    if c.delta is not None and c.delta.content:
                        parts.append(c.delta.content)
                    if c.finish_reason:
                        finish = c.finish_reason
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
            return "".join(parts), finish, usage
        finally:
            self.cache.release(lease)
