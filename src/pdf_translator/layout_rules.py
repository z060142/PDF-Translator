"""Layout rules learned from the document while it is being translated.

1. Class scale: every text style class (original size + bold) gets one scale for the whole
   document, taken as a low percentile of the scales at which its translated units fit.
   Same-style text therefore has the same size on every page; only real outliers shrink further.
2. Length ratio: median translation/source length, used to predict which segments will
   overflow so the translator can be told their space limit up front.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from collections import defaultdict
from typing import Iterable

from .markup import plain
from .pdf_layout import FontPair, capacity_chars, fit_scale, prepare, style_key

CLASS_PERCENTILE = 0.2      # 80% of a class fits at the class scale without extra shrinking
MIN_RATIO_SAMPLES = 10
_CJK_TARGET = re.compile(r"中文|漢|汉|日本|日語|日语|韓|韩|한국|Chinese|Japanese|Korean|繁體|繁体|简体", re.I)


def target_char_em(target_lang: str) -> float:
    """Average character width in em for the target language."""
    return 1.0 if _CJK_TARGET.search(target_lang) else 0.5


def _percentile(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    if len(vals) < 5:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def unit_fit(u: dict, fonts: FontPair, lo: float, hi: float) -> float:
    """Fit scale of a translated unit, cached in the unit (invalidated by text/font/limits)."""
    text = u["translation"].strip()
    marks = ",".join(u.get("marks") or ())
    key = hashlib.md5(f"{fonts.signature}|{lo:.3f}|{hi:.3f}|{marks}|{u.get('italic')}|{text}"
                      .encode("utf-8")).hexdigest()[:16]
    if u.get("fit_key") != key:
        u["fit"] = round(fit_scale(u, prepare(u, text, fonts), lo, hi, fonts.spacing), 4)
        u["fit_key"] = key
    return u["fit"]


def learn(units_lists: Iterable[list[dict]], fonts: FontPair, min_scale: float, max_scale: float) -> dict:
    samples: dict[str, list[float]] = defaultdict(list)
    ratios: list[float] = []
    for units in units_lists:
        for u in units:
            if u.get("skip") or not u.get("translation"):
                continue
            samples[style_key(u)].append(unit_fit(u, fonts, min_scale, max_scale))
            src = len(plain(u, "text"))
            if src >= 8:
                ratios.append(len(plain(u, "translation")) / src)
    scales = {k: round(max(min_scale, min(max_scale, _percentile(v, CLASS_PERCENTILE))), 3)
              for k, v in samples.items()}
    return {
        "scales": scales,
        "counts": {k: len(v) for k, v in samples.items()},
        "ratio": round(statistics.median(ratios), 3) if len(ratios) >= MIN_RATIO_SAMPLES else None,
    }


def length_limits(units: list[dict], rules: dict, max_scale: float, char_em: float,
                  use_current: bool = False) -> dict[str, int]:
    """Space limits for segments likely to overflow at full size.

    use_current: judge by the existing translation instead of the predicted length.
    """
    ratio = rules.get("ratio") or 1.0
    limits: dict[str, int] = {}
    for u in units:
        src = len(plain(u, "text"))
        if src < 20 or u["pitch"] is None:   # single lines hug the text, a limit would mislead
            continue
        cap = capacity_chars(u, max_scale, char_em)
        expected = len(plain(u, "translation")) if (use_current and u.get("translation")) else ratio * src
        if expected > 0.8 * cap:
            limits[u["id"]] = cap
    return limits


def summary(rules: dict) -> str:
    parts = [f"{k} ×{v:.2f}（{rules['counts'][k]} 段）"
             for k, v in sorted(rules["scales"].items(), key=lambda kv: -rules["counts"][kv[0]])[:6]]
    ratio = rules.get("ratio")
    tail = f"；譯文/原文長度比 {ratio:.2f}" if ratio else ""
    return "、".join(parts) + tail
