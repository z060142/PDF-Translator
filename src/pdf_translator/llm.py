"""Unified async JSON-completion client for Anthropic and OpenAI-compatible providers."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import anthropic
import openai

from .config import FALLBACK_MODELS, ModelConfig


class LLMError(Exception):
    """Error after which retrying the same request is pointless (bad config, refusal...)."""

    def __init__(self, msg: str, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal  # True -> configuration problem, job should pause


class RetryableError(Exception):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    requests: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.requests += other.requests


@dataclass
class Prompt:
    system: str
    context: str          # large, stable part (full page text) -> cached when possible
    task: str             # request-specific part
    schema: dict = field(default_factory=dict)


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


class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        key = os.environ.get(cfg.key_env(), "")
        if not key:
            raise LLMError(f"環境變數 {cfg.key_env()} 未設定", fatal=True)
        if cfg.provider == "anthropic":
            self._a = anthropic.AsyncAnthropic(api_key=key, max_retries=2)
        elif cfg.provider == "openai":
            self._o = openai.AsyncOpenAI(api_key=key, base_url=cfg.base_url.strip() or None, max_retries=2)
        else:
            raise LLMError(f"未知的 provider: {cfg.provider}", fatal=True)

    async def close(self) -> None:
        try:
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
        output_config: dict = {}
        if p.schema:
            output_config["format"] = {"type": "json_schema", "schema": p.schema}
        kwargs: dict = dict(
            model=cfg.model,
            max_tokens=int(cfg.max_tokens),
            system=p.system,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": p.context, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": p.task},
                ],
            }],
        )
        if cfg.effort and cfg.effort != "default":
            output_config["effort"] = cfg.effort
            kwargs["thinking"] = {"type": "adaptive"}
        if output_config:
            kwargs["output_config"] = output_config

        if cfg.refusal_fallback and cfg.model in FALLBACK_MODELS:
            async with self._a.beta.messages.stream(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            ) as stream:
                msg = await stream.get_final_message()
        else:
            async with self._a.messages.stream(**kwargs) as stream:
                msg = await stream.get_final_message()

        u = msg.usage
        usage = Usage(u.input_tokens or 0, u.output_tokens or 0, getattr(u, "cache_read_input_tokens", 0) or 0, 1)
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise LLMError(f"模型拒絕回應 (refusal, category={cat})")
        if msg.stop_reason == "max_tokens":
            raise RetryableError("輸出被 max_tokens 截斷，請調高 max_tokens 或降低批次大小")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return _extract_json(text), usage

    # ---------------- OpenAI-compatible ----------------
    async def _openai(self, p: Prompt) -> tuple[dict, Usage]:
        cfg = self.cfg
        system = p.system
        if p.schema:
            system += ("\n\nRespond with a single JSON object matching this JSON schema, no other text:\n"
                       + json.dumps(p.schema, ensure_ascii=False))
        kwargs: dict = dict(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": p.context + "\n\n" + p.task},
            ],
        )
        if cfg.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if cfg.effort and cfg.effort != "default":
            kwargs["reasoning_effort"] = cfg.effort
        if cfg.max_tokens:
            kwargs["max_completion_tokens"] = int(cfg.max_tokens)
        resp = await self._o.chat.completions.create(**kwargs)
        u = resp.usage
        usage = Usage(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0, 0, 1)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise RetryableError("輸出被長度上限截斷，請調高 max_tokens 或降低批次大小")
        return _extract_json(choice.message.content or ""), usage
