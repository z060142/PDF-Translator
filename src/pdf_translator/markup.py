"""Inline formatting carried through translation as a small, allowlisted Markdown subset.

Source segments mark inline styles (a bold word inside a regular paragraph, an underlined
phrase, a footnote superscript...) with the markers below; the translator must put the same
markers around the corresponding translated words. Styles of a whole segment (a bold heading)
stay unit-level properties and are not marked.

Only allowlisted markers are recognised. For each segment:
- allowed  = marker types its source uses; these are rendered.
- foreign  = other allowlisted types whose delimiter does not occur literally in the source;
             if the translation pairs them up they are model-added formatting and are removed.
- anything else (types whose delimiter is literal source text, other Markdown, HTML) is text.
In segments that use markers, literal marker characters are backslash-escaped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Mark:
    key: str
    delim: str
    rule: str


# longest delimiters first: the tokenizer tries them in this order
ALLOWLIST: tuple[Mark, ...] = (
    Mark("bold", "**", "**text** = bold"),
    Mark("underline", "++", "++text++ = underline"),
    Mark("strike", "~~", "~~text~~ = strikethrough"),
    Mark("highlight", "==", "==text== = highlighted"),
    Mark("italic", "*", "*text* = italic"),
    Mark("sup", "^", "^text^ = superscript (e.g. footnote numbers)"),
    Mark("sub", "~", "~text~ = subscript"),
)
BY_KEY = {m.key: m for m in ALLOWLIST}
NEST_ORDER = ("highlight", "underline", "strike", "bold", "italic", "sup", "sub")  # outermost first
_ESCAPABLE = set("".join(m.delim for m in ALLOWLIST)) | {"\\"}
_HTML_TAG = re.compile(r"</?(?:b|strong|i|em|u|ins|s|del|strike|mark|sup|sub|span|font)\b[^>]*>", re.I)

Styles = frozenset
_EMPTY: frozenset = frozenset()


# ---------------------------------------------------------------- writing

def _settle_whitespace(text: str, styles: list[frozenset]) -> list[frozenset]:
    """Whitespace takes only the styles shared by its neighbours, so markers hug words
    ("**a** b", not "**a **b") while underline still runs across the space in "++a b++"."""
    out = list(styles)
    n = len(text)
    prev = [_EMPTY] * n
    last = _EMPTY
    for i in range(n):
        prev[i] = last
        if not text[i].isspace():
            last = styles[i]
    nxt = _EMPTY
    for i in range(n - 1, -1, -1):
        if text[i].isspace():
            out[i] = styles[i] & prev[i] & nxt
        else:
            nxt = styles[i]
    return out


def to_markup(text: str, styles: list[frozenset], keys: set[str] | frozenset | None = None) -> str:
    """Serialize plain text + per-character styles into marker text. Literal characters of the
    `keys` delimiters (default: the styles used) are escaped; no keys -> plain text."""
    styles = _settle_whitespace(text, styles)
    used = set().union(*styles) if styles else set()
    keys = used if keys is None else set(keys) | used
    if not keys:
        return text
    escape = {c for k in keys for c in BY_KEY[k].delim} | {"\\"}
    out: list[str] = []
    stack: list[str] = []
    for ch, st in zip(text, styles):
        keep = 0
        while keep < len(stack) and stack[keep] in st:
            keep += 1
        for k in reversed(stack[keep:]):
            out.append(BY_KEY[k].delim)
        del stack[keep:]
        for k in NEST_ORDER:
            if k in st and k not in stack:
                out.append(BY_KEY[k].delim)
                stack.append(k)
        out.append("\\" + ch if ch in escape else ch)
    for k in reversed(stack):
        out.append(BY_KEY[k].delim)
    return "".join(out)


# ---------------------------------------------------------------- reading

@dataclass
class Parsed:
    text: str                    # plain text
    styles: list[frozenset]      # per character
    used: set[str]               # marker types with at least one pair
    unbalanced: set[str]         # marker types with an unpaired delimiter


def parse(text: str, keys: set[str] | frozenset, escapes: bool,
          literal_unpaired: set[str] | frozenset = frozenset()) -> Parsed:
    """Parse marker text. Only `keys` are recognised; an unpaired delimiter is dropped, or
    kept as literal text for types in `literal_unpaired`."""
    marks = [m for m in ALLOWLIST if m.key in keys]
    toks: list[tuple[str, str]] = []           # ("c", char) | ("m", key)
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if escapes and ch == "\\" and i + 1 < n and text[i + 1] in _ESCAPABLE:
            toks.append(("c", text[i + 1]))
            i += 2
            continue
        for m in marks:
            if text.startswith(m.delim, i):
                toks.append(("m", m.key))
                i += len(m.delim)
                break
        else:
            toks.append(("c", ch))
            i += 1
    count: dict[str, int] = {}
    for kind, v in toks:
        if kind == "m":
            count[v] = count.get(v, 0) + 1
    unbalanced = {k for k, c in count.items() if c % 2}
    seen: dict[str, int] = {}
    active: set[str] = set()
    chars: list[str] = []
    styles: list[frozenset] = []
    cur: frozenset = _EMPTY
    for kind, v in toks:
        if kind == "c":
            chars.append(v)
            styles.append(cur)
            continue
        seen[v] = seen.get(v, 0) + 1
        if seen[v] == count[v] and count[v] % 2:          # the unpaired last delimiter
            if v in literal_unpaired:
                for c in BY_KEY[v].delim:
                    chars.append(c)
                    styles.append(cur)
            continue
        active ^= {v}
        cur = frozenset(active)
    used = {k for k, c in count.items() if c >= 2}
    return Parsed("".join(chars), styles, used, unbalanced)


def unit_marks(u: dict) -> frozenset:
    return frozenset(u.get("marks") or ())


def plain(u: dict, field: str = "text") -> str:
    """Text of a unit field without markers."""
    text = u.get(field) or ""
    marks = unit_marks(u)
    if not marks:
        return text
    return parse(text, marks, True).text


def styled(u: dict, field: str = "translation", inline: bool = True) -> tuple[str, list[frozenset]]:
    """Plain text and per-character styles of a unit field, including unit-level bold / italic."""
    return styled_text(u, u.get(field) or "", inline)


def styled_text(u: dict, text: str, inline: bool = True) -> tuple[str, list[frozenset]]:
    """Plain text and per-character styles of `text` (marker text of unit u)."""
    marks = unit_marks(u)
    base = frozenset(k for k in ("bold", "italic") if u.get(k))
    if not marks:
        return text, [base] * len(text)
    p = parse(text, marks, True)
    if not inline:
        return p.text, [base] * len(p.text)
    return p.text, [base | s for s in p.styles]


def normalize_translation(u: dict, text: str) -> tuple[str, list[str]]:
    """Validate a returned translation against its source markers.

    Returns (canonical text, problems). The canonical text keeps only the source's marker
    types, balanced and escaped; problems describe what the model did wrong."""
    allowed = unit_marks(u)
    src_plain = plain(u, "text")
    problems: list[str] = []
    tags = {t.lower() for t in _HTML_TAG.findall(text)} - {t.lower() for t in _HTML_TAG.findall(src_plain)}
    if tags:
        problems.append("HTML tags are not allowed: " + " ".join(sorted(tags)))
        text = _HTML_TAG.sub(lambda m: m.group(0) if m.group(0).lower() not in tags else "", text)
    foreign = {m.key for m in ALLOWLIST if m.key not in allowed and m.delim not in src_plain}
    p = parse(text, allowed | foreign, escapes=bool(allowed), literal_unpaired=foreign)
    for k in sorted(p.used & foreign, key=NEST_ORDER.index):
        problems.append(f"marker {BY_KEY[k].delim} is not allowed in this segment")
    for k in sorted(p.unbalanced & allowed, key=NEST_ORDER.index):
        problems.append(f"unbalanced {BY_KEY[k].delim} markers")
    for k in sorted(allowed - p.used, key=NEST_ORDER.index):
        problems.append(f"missing {BY_KEY[k].rule.split(' = ')[0]} ({k}) used in the source")
    styles = [s & allowed for s in p.styles]
    return (to_markup(p.text, styles, allowed) if allowed else p.text), problems


def rules_for(units: list[dict]) -> str:
    """Formatting rules for one request: only the marker types its segments use."""
    keys = set().union(*(unit_marks(u) for u in units)) if units else set()
    if not keys:
        return ""
    escapes = any("\\" in (u.get("text") or "") for u in units if unit_marks(u))
    lines = [f"  {BY_KEY[k].rule}" for k in NEST_ORDER if k in keys]
    text = ("Inline formatting: some segments mark inline styles with these Markdown markers "
            "(the complete allowlist for this request):\n" + "\n".join(lines) + "\n"
            "- In each segment, put the same markers around the translated words that correspond to the "
            "marked source words. Every marker type a segment uses must appear in its translation, with "
            "matching opening and closing markers. Word order may change; the markers follow the words.\n"
            "- Use no other markers, Markdown or HTML, and add no markers to segments whose source has none.\n")
    if escapes:
        text += "- A backslash before a marker character (e.g. \\*) is a literal character: keep the backslash.\n"
    return text + "\n"
