"""Prompt caching for OpenAI-compatible endpoints: one "dialect" per service / model family.

The request layout is the same as on the Anthropic path (system -> page context -> task, stable
first), and so are the rules: write only what will be read again, and let the first request over
a shared prefix prime the cache while the others wait for its first streamed token. What differs
per service is how a breakpoint is written, how requests are kept on the machine / provider that
holds the cache, and what reads and writes cost:

  OpenAI GPT-5.6+     explicit `prompt_cache_breakpoint` on content parts + `prompt_cache_options`
                      {"mode": "explicit"} (no implicit breakpoint on the unique tail); writes 1.25x
  OpenAI older        automatic, free writes; `prompt_cache_key` routes a page's requests together
  OpenRouter          Anthropic / Qwen models: `cache_control` breakpoints; OpenAI GPT-5.6+: as
                      above; everything else automatic. `session_id` keeps the job on one provider
  DeepSeek            automatic disk cache; hits reported as prompt_cache_hit_tokens
  other servers       nothing sent; hits read from usage when reported
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .prompt_cache import min_tokens as anthropic_min_tokens
from .prompt_cache import read_mult as anthropic_read_mult


def family(base_url: str) -> str:
    u = (base_url or "").strip().lower()
    if not u or "api.openai.com" in u:
        return "openai"
    if "openrouter.ai" in u:
        return "openrouter"
    if "deepseek.com" in u:
        return "deepseek"
    return "other"


def _gpt_version(model: str) -> tuple[int, int] | None:
    m = re.search(r"gpt-(\d+)(?:\.(\d+))?", model.lower())
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def _openai_explicit(model: str) -> bool:
    v = _gpt_version(model)
    return v is not None and v >= (5, 6)


def _openai_read_mult(model: str) -> float:
    v = _gpt_version(model)
    if v is None:
        return 0.5
    if v >= (5, 0):
        return 0.1
    return 0.25 if "4.1" in model else 0.5


@dataclass
class Dialect:
    family: str
    marker: str | None          # content-part field marking a breakpoint, None = automatic caching
    gate: bool                  # coordinate parallel requests (needs streaming)
    min_tokens: int
    read_mult: float            # price of a cached read relative to base input
    write_mult: float           # price of a reported cache write relative to base input
    ttl_seconds: int            # how long a primed prefix is assumed warm

    def breakpoint(self, ttl: str) -> dict | None:
        if self.marker == "prompt_cache_breakpoint":
            return {"mode": "explicit", "ttl": "30m"}
        if self.marker == "cache_control":
            return {"type": "ephemeral", "ttl": "1h"} if ttl == "1h" else {"type": "ephemeral"}
        return None


def dialect_for(base_url: str, model: str, ttl: str) -> Dialect:
    fam = family(base_url)
    name = model.lower()
    if fam == "openrouter" and "/" in name:
        vendor, bare = name.split("/", 1)
    else:
        vendor, bare = ("openai" if fam == "openai" else fam), name
    if vendor == "anthropic":
        canonical = bare.replace(".", "-")
        return Dialect(fam, "cache_control", True, anthropic_min_tokens(canonical),
                       anthropic_read_mult(canonical), 2.0 if ttl == "1h" else 1.25, 3600 if ttl == "1h" else 300)
    if vendor == "qwen":
        return Dialect(fam, "cache_control", True, 1024, 0.1, 1.25, 300)
    if vendor == "openai":
        if _openai_explicit(bare):
            return Dialect(fam, "prompt_cache_breakpoint", True, 1024, 0.1, 1.25, 1800)
        return Dialect(fam, None, fam in ("openai", "openrouter"), 1024, _openai_read_mult(bare), 1.0, 300)
    if vendor == "google":
        return Dialect(fam, None, True, 4096 if "pro" in bare else 1024, 0.25, 1.0, 300)
    if vendor == "deepseek":
        return Dialect(fam, None, False, 1024, 0.1, 1.0, 300)
    if vendor in ("x-ai", "moonshotai"):
        return Dialect(fam, None, fam == "openrouter", 1024, 0.25, 1.0, 300)
    return Dialect(fam, None, fam == "openrouter", 1024, 0.5, 1.0, 300)


def request_extras(d: Dialect, ttl: str, model: str, system: str, group: str, role: str) -> dict:
    """Top-level request fields (sent through extra_body / typed params) for this dialect."""
    out: dict = {}
    if d.marker == "prompt_cache_breakpoint":
        # explicit mode: no implicit breakpoint on the unique tail; "off" -> no writes at all
        out["prompt_cache_options"] = {"mode": "explicit"}
    if ttl == "off":
        return out
    digest = hashlib.sha1(f"{model}\x00{system}".encode("utf-8")).hexdigest()[:16]
    if d.family == "openai":
        # routes a page's requests to the machine holding its prefix (keep ~15 requests/min per key)
        out["prompt_cache_key"] = f"pdftr-{digest}-{group}" if group else f"pdftr-{digest}"
    elif d.family == "openrouter":
        # sticky provider routing for the whole job, so every request can read the same cache
        out["session_id"] = f"pdftr-{role}-{digest}"
    return out


def parse_usage(u) -> tuple[int, int, int, int]:
    """(uncached input, cached read, cache write, output) from an OpenAI-compatible usage object."""
    prompt = getattr(u, "prompt_tokens", 0) or 0
    out = getattr(u, "completion_tokens", 0) or 0
    det = getattr(u, "prompt_tokens_details", None)
    read = (getattr(det, "cached_tokens", 0) or 0) if det is not None else 0
    write = (getattr(det, "cache_write_tokens", 0) or 0) if det is not None else 0
    if not read:
        read = getattr(u, "prompt_cache_hit_tokens", 0) or 0          # DeepSeek
    return max(0, prompt - read - write), read, write, out
