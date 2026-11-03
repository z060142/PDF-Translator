"""PDF text extraction into translatable units, and re-rendering translated text in place.

Only text is touched: original glyphs are removed with redactions (no fill, images and
vector graphics kept), then the translation is written into the same box with the same
baseline, size and colour, using one normalised font family.

Inline styles (bold / italic words, underline, strikethrough, highlight, super/subscript) are
read per character and carried through translation as allowlisted Markdown markers (see
markup.py). Underline / strikethrough lines and highlight annotations of translated text are
removed and redrawn under the translated words.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass

import pymupdf

from .markup import NEST_ORDER, styled_text, to_markup

# Characters that may be broken anywhere when wrapping (CJK ideographs, kana, fullwidth forms).
_BREAK_ANY = r"⺀-⿿　-ヿ㄀-ㇿ㐀-䶿一-鿿豈-﫿︰-﹏＀-￯"
_TOKEN_RE = re.compile(rf"[^\s{_BREAK_ANY}]+\s*|\s+|.", re.S)
_CJK_CHAR = re.compile(rf"[{_BREAK_ANY}]")
# Line starts that open a new list item: "1.", "2)", "(3)", "①", "•", "-", "a." ...
LIST_RE = re.compile(r"^(\d{1,3}[.)．、](?!\d)|\(\d{1,3}\)|[①-⑳❶-❿➀-➓]|[•●▪■◆○◦·\-–—]\s|[a-zA-Z][.)]\s)")
NO_LINE_END = set("（「『【〔〈《‘“([{")
NO_LINE_START = set("，。、；：？！）」』】〕〉》’”…‧·,.;:?!)]}%")
FLAG_SUPERSCRIPT, FLAG_ITALIC, FLAG_SERIF, FLAG_BOLD = 1, 2, 4, 16

SCRIPT_SCALE = 0.65          # size of super/subscript text relative to the line
ITALIC_SHEAR = 0.21          # slant used to draw italic text (no italic CJK fonts exist)
DEFAULT_HIGHLIGHT = 0xFFEB3B


# ---------------------------------------------------------------- decorations

class _Decorations:
    """Underline / strikethrough candidates (thin horizontal vector lines) and text-markup
    annotations of one page, matched against characters during extraction.

    A drawing path is only treated as decoration (removed and redrawn) when every one of its
    segments lies under text that covers most of its length, so table rules stay untouched."""

    def __init__(self, page: pymupdf.Page):
        self.segs: list[tuple[float, float, float, float, int]] = []    # x0, x1, y, thickness, path id
        self.covered: list[float] = []
        self.path_segs: dict[int, list[int]] = {}
        self.path_rect: dict[int, list[float]] = {}
        self.buckets: dict[int, list[int]] = defaultdict(list)
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        for pid, p in enumerate(drawings):
            segs = []
            for it in p.get("items", ()):
                if it[0] == "l":
                    a, b = it[1], it[2]
                    if abs(a.y - b.y) > 0.5:
                        break
                    segs.append((min(a.x, b.x), max(a.x, b.x), (a.y + b.y) / 2, float(p.get("width") or 1.0)))
                elif it[0] == "re":
                    r = it[1]
                    if r.height > 3.0 or r.width < 2 * r.height:
                        break
                    segs.append((r.x0, r.x1, (r.y0 + r.y1) / 2, r.height))
                else:
                    break
            else:
                if segs:
                    self.path_segs[pid] = []
                    self.path_rect[pid] = list(p["rect"])
                    for s in segs:
                        sid = len(self.segs)
                        self.segs.append((*s, pid))
                        self.covered.append(0.0)
                        self.path_segs[pid].append(sid)
                        self.buckets[int(s[2] // 4)].append(sid)
        self.annots: list[tuple[list[pymupdf.Rect], str, int]] = []
        self.annot_color: dict[int, int] = {}
        kinds = {pymupdf.PDF_ANNOT_HIGHLIGHT: "highlight", pymupdf.PDF_ANNOT_UNDERLINE: "underline",
                 pymupdf.PDF_ANNOT_SQUIGGLY: "underline", pymupdf.PDF_ANNOT_STRIKE_OUT: "strike"}
        try:
            for a in page.annots(types=list(kinds)):
                vs = a.vertices or []
                rects = [pymupdf.Quad(vs[i:i + 4]).rect for i in range(0, len(vs) - 3, 4)] or [a.rect]
                self.annots.append((rects, kinds[a.type[0]], a.xref))
                c = (a.colors or {}).get("stroke")
                if c and len(c) == 3:
                    self.annot_color[a.xref] = (int(c[0] * 255) << 16) | (int(c[1] * 255) << 8) | int(c[2] * 255)
        except Exception:
            pass
        self._ok: dict[int, bool] | None = None

    def match(self, bbox, baseline: float, size: float, st: set) -> None:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        if self.segs:
            lo, hi = baseline - 0.6 * size, baseline + 0.35 * size
            for b in range(int(lo // 4), int(hi // 4) + 1):
                for sid in self.buckets.get(b, ()):
                    x0, x1, y, th, _ = self.segs[sid]
                    if not (x0 - 0.5 <= cx <= x1 + 0.5) or th > max(2.5, 0.15 * size):
                        continue
                    dy = y - baseline
                    if -0.05 * size <= dy <= 0.35 * size:
                        st.add(f"u:{sid}")
                    elif -0.6 * size <= dy < -0.15 * size:
                        st.add(f"s:{sid}")
                    else:
                        continue
                    self.covered[sid] += w
        for rects, kind, xref in self.annots:
            if any(r.x0 - 0.5 <= cx <= r.x1 + 0.5 and r.y0 - 0.5 <= cy <= r.y1 + 0.5 for r in rects):
                st.add(kind)
                st.add(f"x:{xref}")

    def path_ok(self, sid: int) -> int | None:
        """Path id of a segment if the whole path is a text decoration, else None."""
        if self._ok is None:
            seg_ok = [c >= 0.6 * max(0.5, s[1] - s[0]) for s, c in zip(self.segs, self.covered)]
            self._ok = {pid: all(seg_ok[i] for i in sids) for pid, sids in self.path_segs.items()}
        pid = self.segs[sid][4]
        return pid if self._ok.get(pid) else None


# ---------------------------------------------------------------- extraction

@dataclass
class _Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    baseline: float
    color: int
    bold: bool
    serif: bool
    rects: list
    styles: list            # per character: set of style keys + pending decoration refs


def _line_info(spans: list[dict], deco: _Decorations | None = None) -> _Line:
    text = "".join(s["text"] for s in spans)
    xs0 = [s["bbox"][0] for s in spans if s["text"].strip()]
    xs1 = [s["bbox"][2] for s in spans if s["text"].strip()]
    ys0 = [s["bbox"][1] for s in spans if s["text"].strip()]
    ys1 = [s["bbox"][3] for s in spans if s["text"].strip()]
    # dominant span by character count, ignoring superscripts
    main = None
    best = -1
    for s in spans:
        n = len(s["text"].strip())
        if s["flags"] & FLAG_SUPERSCRIPT:
            n = n * 0.1
        if n > best:
            best, main = n, s
    bold_chars = sum(len(s["text"].strip()) for s in spans
                     if s["flags"] & FLAG_BOLD or "bold" in s["font"].lower())
    total = sum(len(s["text"].strip()) for s in spans) or 1
    size, baseline = float(main["size"]), float(main["origin"][1])

    styles: list[set] = []
    for s in spans:
        font = s["font"].lower()
        st0: set = set()
        if s["flags"] & FLAG_BOLD or "bold" in font:
            st0.add("bold")
        if s["flags"] & FLAG_ITALIC or "italic" in font or "oblique" in font:
            st0.add("italic")
        if s is not main:
            ssz, oy = float(s["size"]), float(s["origin"][1])
            if s["flags"] & FLAG_SUPERSCRIPT or (ssz < 0.85 * size and oy < baseline - 0.2 * size):
                st0.add("sup")
            elif ssz < 0.85 * size and oy > baseline + 0.1 * size:
                st0.add("sub")
        chars = s.get("chars")
        if chars is None or len(chars) != len(s["text"]):
            styles += [set(st0) for _ in s["text"]]
            continue
        for c in chars:
            st = set(st0)
            if deco is not None:
                deco.match(c["bbox"], baseline, size, st)
            styles.append(st)

    return _Line(
        text=text,
        x0=min(xs0), y0=min(ys0), x1=max(xs1), y1=max(ys1),
        size=size,
        baseline=baseline,
        color=int(main["color"]),
        bold=bold_chars / total > 0.6,
        serif=bool(main["flags"] & FLAG_SERIF),
        rects=[list(s["bbox"]) for s in spans if s["text"].strip()],
        styles=styles,
    )


def _strip_span(text: str) -> tuple[int, int]:
    return len(text) - len(text.lstrip()), len(text.rstrip())


def _merge_fragments(a: _Line, b: _Line) -> _Line:
    """Join two pieces of one visual line (justified text is often split at wide gaps)."""
    main = a if len(a.text.strip()) >= len(b.text.strip()) else b
    ae = len(a.text.rstrip())
    bs = _strip_span(b.text)[0]
    return _Line(
        text=a.text.rstrip() + " " + b.text.lstrip(),
        x0=min(a.x0, b.x0), y0=min(a.y0, b.y0), x1=max(a.x1, b.x1), y1=max(a.y1, b.y1),
        size=main.size, baseline=main.baseline, color=main.color,
        bold=a.bold and b.bold, serif=main.serif, rects=a.rects + b.rects,
        styles=a.styles[:ae] + [set()] + b.styles[bs:],
    )


def _same_row(a: _Line, b: _Line) -> bool:
    size = max(a.size, b.size)
    return abs(a.baseline - b.baseline) < 0.3 * size and -0.2 * size < b.x0 - a.x1 < 1.4 * size  # table cells are further apart


def _first_word_width(line: _Line) -> float:
    t = line.text.strip()
    word = t.split()[0] if t.split() else t
    if word and _CJK_CHAR.match(word[0]):
        word = word[0]
    return sum(line.size * (1.0 if ord(ch) > 0x2E80 else 0.55) for ch in word)


def _coarse_break(prev: _Line, cur: _Line) -> bool:
    """Lines that can never belong to the same text region."""
    size = max(cur.size, prev.size)
    if abs(cur.size - prev.size) > 0.15 * size:
        return True
    if cur.baseline <= prev.baseline + 0.3 * prev.size:      # same row / going up -> separate cell
        return True
    if cur.y0 - prev.y1 > 0.9 * size:                         # paragraph gap
        return True
    if cur.bold != prev.bold:
        return True
    overlap = min(cur.x1, prev.x1) - max(cur.x0, prev.x0)
    return overlap < 0.5 * min(cur.x1 - cur.x0, prev.x1 - prev.x0)  # different column


_NUM_MARK = re.compile(r"^\(?(\d{1,3}|[a-zA-Z])[.)．、]?")


def _list_item(text: str, last: dict) -> bool:
    """True if the line starts a list item. Numbered markers must continue a sequence (or start
    at 1/a), so "(3) times a week" wrapped to a line start is not mistaken for an item."""
    t = text.strip()
    m = LIST_RE.match(t)
    if not m:
        return False
    nm = _NUM_MARK.match(t)
    if not nm or not t[0].isalnum() and t[0] != "(":
        return True                                            # bullet / circled number
    tok = nm.group(1)
    style = ("paren" if t[0] == "(" else "plain") + ("num" if tok.isdigit() else "alpha")
    n = int(tok) if tok.isdigit() else ord(tok.lower()) - 96
    ok = n == 1 or last.get(style) == n - 1
    if ok:
        last[style] = n
    return ok


def _paragraph_break(prev: _Line, cur: _Line, col_right: float, first: _Line, last: dict) -> bool:
    size = max(cur.size, prev.size)
    if _list_item(cur.text, last):                             # a new list item
        return True
    if cur.x0 > prev.x0 + max(1.5, 0.8 * size):
        # hanging indent under a list item continues it; otherwise an indented line starts a paragraph
        return not (prev is first and LIST_RE.match(first.text.strip()))
    # the next line's first word would have fitted on the previous line -> deliberate line break
    return prev.x1 + 0.3 * size + _first_word_width(cur) < col_right - 0.5 * size


def _join(lines: list[_Line]) -> tuple[str, list[set]]:
    """Join the lines of a paragraph (de-hyphenation, no space between CJK lines) with styles."""
    a, b = _strip_span(lines[0].text)
    out = list(lines[0].text[a:b])
    st = list(lines[0].styles[a:b])
    for ln in lines[1:]:
        a, b = _strip_span(ln.text)
        nxt, nst = ln.text[a:b], ln.styles[a:b]
        if not nxt:
            continue
        if out and out[-1] == "-" and len(out) > 1 and out[-2].isalpha() and nxt[:1].islower():
            out.pop()
            st.pop()
        elif not (out and _CJK_CHAR.match(out[-1]) and _CJK_CHAR.match(nxt[:1])):
            out.append(" ")
            st.append(set())
        out += nxt
        st += nst
    return "".join(out), st


def _spread(vals: list[float]) -> float:
    return max(vals) - min(vals) if vals else 0.0


def _make_unit(lines: list[_Line], page_rect: pymupdf.Rect, uid: str,
               deco: _Decorations | None = None, markup: bool = True) -> dict:
    text, raw_styles = _join(lines)
    chars = [max(1, len(l.text.strip())) for l in lines]
    size = sum(l.size * c for l, c in zip(lines, chars)) / sum(chars)
    color = max(set(l.color for l in lines), key=lambda c: sum(ch for l, ch in zip(lines, chars) if l.color == c))
    x0, y0 = min(l.x0 for l in lines), min(l.y0 for l in lines)
    x1, y1 = max(l.x1 for l in lines), max(l.y1 for l in lines)
    pitch = None
    if len(lines) > 1:
        diffs = [b.baseline - a.baseline for a, b in zip(lines, lines[1:]) if b.baseline > a.baseline]
        if diffs:
            pitch = statistics.median(diffs)

    tol = max(1.5, 0.5 * size)
    lefts = [l.x0 for l in lines]
    rights = [l.x1 for l in lines]
    centers = [(l.x0 + l.x1) / 2 for l in lines]
    indent = hang = 0.0        # x offsets of the first line / following lines from bbox x0
    if len(lines) >= 2:
        rest_aligned = len(lines) >= 3 and _spread(lefts[1:]) < tol
        first_indented = lefts[0] - min(lefts[1:]) > tol and (
            rest_aligned or (len(lines) == 2 and abs(rights[0] - rights[1]) < tol))
        hanging = min(lefts[1:]) - lefts[0] > tol and (rest_aligned or LIST_RE.match(lines[0].text.strip()))
        if _spread(lefts) < tol:
            align = "left"
        elif first_indented:
            align = "left"
            indent = lefts[0] - min(lefts[1:])
        elif hanging:
            align = "left"
            hang = min(lefts[1:]) - lefts[0]
        elif _spread(centers) < tol:
            align = "center"
        elif _spread(rights) < tol:
            align = "right"
        else:
            align = "left"
    else:
        pw = page_rect.width
        cx = (x0 + x1) / 2
        align = "center" if abs(cx - pw / 2) < 0.03 * pw and x0 > 0.12 * pw else "left"

    # ---- inline styles ----
    styles: list[set] = []
    deco_paths: set[int] = set()
    xrefs: set[int] = set()
    for st in raw_styles:
        s2 = set()
        for k in st:
            if ":" not in k:
                s2.add(k)
                continue
            kind, ref = k.split(":", 1)
            if kind == "x":
                xrefs.add(int(ref))
            elif deco is not None:
                pid = deco.path_ok(int(ref))
                if pid is not None:
                    s2.add("underline" if kind == "u" else "strike")
                    deco_paths.add(pid)
        styles.append(s2)
    solid = [i for i, ch in enumerate(text) if not ch.isspace()] or [0]
    bold = sum(c for l, c in zip(lines, chars) if l.bold) / sum(chars) > 0.5
    italic = bool(text) and sum(1 for i in solid if i < len(styles) and "italic" in styles[i]) > 0.5 * len(solid)
    inline: list[frozenset] = []
    for st in styles:
        if bold or not markup:
            st.discard("bold")
        if italic or not markup:
            st.discard("italic")
        inline.append(frozenset(st) if markup else frozenset())
    marks = sorted(set().union(*inline), key=NEST_ORDER.index) if inline else []

    unit = {
        "id": uid,
        "text": to_markup(text, inline) if marks else text,
        "bbox": [round(v, 2) for v in (x0, y0, x1, y1)],
        "size": round(size, 2),
        "pitch": round(pitch, 2) if pitch else None,
        "first_baseline": round(lines[0].baseline, 2),
        "color": color,
        "bold": bold,
        "italic": italic,
        "serif": lines[0].serif,
        "align": align,
        "indent": round(indent, 2),
        "hang": round(hang, 2),
        "rects": [r for l in lines for r in l.rects],
        "skip": not any(ch.isalpha() for ch in text),
        "translation": None,
    }
    if marks:
        unit["marks"] = marks
    if markup and deco is not None:
        if deco_paths:
            unit["deco_rects"] = [[round(v, 2) for v in deco.path_rect[p]] for p in sorted(deco_paths)]
        if xrefs:
            unit["annots"] = sorted(xrefs)
            for x in unit["annots"]:
                if x in deco.annot_color:
                    unit["hl_color"] = deco.annot_color[x]
                    break
    return unit


def extract_units(page: pymupdf.Page, markup: bool = True) -> list[dict]:
    """Split a page into translatable units. markup=True also records inline styles."""
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    d = page.get_text("rawdict", flags=flags)
    deco = _Decorations(page) if markup else None
    # 1) collect horizontal lines in reading order; None marks a hard separator
    seq: list[_Line | None] = []
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            for s in ln["spans"]:
                s["text"] = "".join(c["c"] for c in s.get("chars", ()))
            dx, dy = ln["dir"]
            spans = [s for s in ln["spans"] if s["text"]]
            if abs(dy) > 0.01 or dx < 0 or not "".join(s["text"] for s in spans).strip():
                seq.append(None)    # non-horizontal / empty line: left untouched
                continue
            info = _line_info(spans, deco)
            prev = seq[-1] if seq else None
            if prev is not None and _same_row(prev, info):
                seq[-1] = _merge_fragments(prev, info)
            else:
                seq.append(info)
    # 2) coarse regions, then split each region into paragraphs using its right edge
    regions: list[list[_Line]] = []
    cur: list[_Line] = []
    for info in seq:
        if info is None or (cur and _coarse_break(cur[-1], info)):
            if cur:
                regions.append(cur)
            cur = []
        if info is not None:
            cur.append(info)
    if cur:
        regions.append(cur)

    groups: list[list[_Line]] = []
    last_marks: dict = {}                    # last list number seen per marker style
    for reg in regions:
        right = max(l.x1 for l in reg)
        _list_item(reg[0].text, last_marks)  # region starts count for numbering too
        g = [reg[0]]
        for prev, line in zip(reg, reg[1:]):
            if _paragraph_break(prev, line, right, g[0], last_marks):
                groups.append(g)
                g = []
            g.append(line)
        groups.append(g)
    units = [_make_unit(g, page.rect, f"S{i}", deco, markup) for i, g in enumerate(groups, 1)]
    # a single line that merely sits near the page centre but shares its left edge with a
    # neighbouring unit is left-aligned text, not a centred heading
    for i, u in enumerate(units):
        if u["align"] == "center" and u["pitch"] is None:
            for j in (i - 2, i - 1, i + 1, i + 2):
                if 0 <= j < len(units) and abs(units[j]["bbox"][0] - u["bbox"][0]) < 0.5 * u["size"]:
                    u["align"] = "left"
                    break
    return units


# ---------------------------------------------------------------- rendering

class FontPair:
    """Regular + bold font used for all translated text (font normalisation).

    spacing: extra letter spacing in em (negative = tighter), applied to every glyph.
    markup: render inline styles (False = plain text, original decorations left alone).
    """

    def __init__(self, regular_path: str, bold_path: str, spacing: float = 0.0, markup: bool = True):
        self.regular_path, self.bold_path = regular_path, bold_path
        self.spacing = float(spacing or 0.0)
        self.markup = bool(markup)
        self.regular = pymupdf.Font(fontfile=regular_path) if regular_path else pymupdf.Font("cjk")
        if bold_path and bold_path != regular_path:
            self.bold = pymupdf.Font(fontfile=bold_path)
        else:
            self.bold = None   # emulate bold with stroke
        self._w1: dict[tuple[str, bool], float] = {}

    @property
    def signature(self) -> str:
        return f"{self.regular_path}|{self.bold_path}|{self.spacing:.3f}|{int(self.markup)}"

    def font(self, bold: bool) -> pymupdf.Font:
        return self.bold if (bold and self.bold is not None) else self.regular

    def font_for(self, u: dict) -> pymupdf.Font:
        return self.font(u["bold"])

    def width1(self, ch: str, bold: bool) -> float:
        """Advance width of one character at size 1."""
        key = (ch, bold and self.bold is not None)
        w = self._w1.get(key)
        if w is None:
            w = self._w1[key] = self.font(bold).text_length(ch, 1.0)
        return w

    def install(self, page: pymupdf.Page) -> None:
        if self.regular_path:
            page.insert_font(fontname="TRreg", fontfile=self.regular_path)
        else:
            page.insert_font(fontname="TRreg", fontbuffer=self.regular.buffer)
        if self.bold is not None:
            page.insert_font(fontname="TRbold", fontfile=self.bold_path)


def text_width(font: pymupdf.Font, text: str, size: float, cs: float = 0.0) -> float:
    """Advance width of a line including letter spacing cs (em) between glyphs."""
    w = font.text_length(text, size)
    if cs and len(text) > 1:
        w += cs * size * (len(text) - 1)
    return w


@dataclass
class Rich:
    """Translated text of a unit ready for layout: plain characters, styles, unit widths."""
    text: str
    styles: list
    w1: list            # advance width per character at size 1 (super/subscript already scaled)


def prepare(u: dict, text: str, fonts: FontPair) -> Rich:
    plain, styles = styled_text(u, text, fonts.markup)
    w1 = []
    for ch, st in zip(plain, styles):
        w = fonts.width1(ch, "bold" in st)
        if "sup" in st or "sub" in st:
            w *= SCRIPT_SCALE
        w1.append(w)
    return Rich(plain, styles, w1)


def _measure(rt: Rich, idx: list[int], size: float, cs: float) -> float:
    j = len(idx)
    while j and rt.text[idx[j - 1]].isspace():
        j -= 1
    if not j:
        return 0.0
    w = size * sum(rt.w1[k] for k in idx[:j])
    return w + cs * size * (j - 1) if cs and j > 1 else w


def _rstrip(rt: Rich, idx: list[int]) -> list[int]:
    j = len(idx)
    while j and rt.text[idx[j - 1]].isspace():
        j -= 1
    return idx[:j]


def _lstrip(rt: Rich, idx: list[int]) -> list[int]:
    j = 0
    while j < len(idx) and rt.text[idx[j]].isspace():
        j += 1
    return idx[j:]


def _wrap(rt: Rich, size: float, width: float, first_width: float | None = None,
          cs: float = 0.0) -> list[list[int]]:
    """Greedy line breaking over character indices; CJK may break between any characters,
    other scripts at spaces. Returns the character indices of every line."""
    lines: list[list[int]] = []
    text = rt.text

    def limit() -> float:
        return first_width if (first_width is not None and not lines) else width

    def measure(idx: list[int]) -> float:
        return _measure(rt, idx, size, cs)

    start = 0
    ends = [i for i, ch in enumerate(text) if ch == "\n"] + [len(text)]
    for end in ends:
        cur: list[int] = []
        for m in _TOKEN_RE.finditer(text, start, end):
            tok = list(range(*m.span()))
            cand = cur + tok
            if not _rstrip(rt, cur) or measure(cand) <= limit():
                cur = cand
            elif text[tok[0]] in NO_LINE_START and measure(cand) <= limit() + 1.05 * size:
                cur = cand                                    # keep closing punctuation on the line
                continue
            else:
                carry: list[int] = []
                cur = _rstrip(rt, cur)
                while len(cur) > 1 and text[cur[-1]] in NO_LINE_END:
                    carry = [cur[-1]] + carry                 # opening brackets move to the next line
                    cur = _rstrip(rt, cur[:-1])
                lines.append(cur)
                cur = carry + _lstrip(rt, tok)
            # the current line is a single token wider than the box: hard-split by characters
            while (measure(cur) > limit() and len(_rstrip(rt, cur)) > 1
                   and [text[k] for k in _rstrip(rt, cur)] == [text[k] for k in _rstrip(rt, _lstrip(rt, tok))]):
                cut = len(cur)
                while cut > 1 and measure(cur[:cut]) > limit():
                    cut -= 1
                lines.append(cur[:cut])
                cur = tok = cur[cut:]
        lines.append(_rstrip(rt, cur))
        start = end + 1
    while len(lines) > 1 and not lines[-1]:
        lines.pop()
    return lines or [[]]


def _rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255


def _geometry(u: dict) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = u["bbox"]
    size0 = u["size"]
    width = max(x1 - x0, size0)
    pitch0 = min(max(u["pitch"] or size0 * 1.25, size0 * 1.0), size0 * 2.5)
    asc = max(u["first_baseline"] - y0, size0 * 0.6)
    avail = (y1 - y0) + 0.25 * size0
    return width, pitch0, asc, avail


def _try_scale(u: dict, rt: Rich, scale: float, cs: float = 0.0) -> tuple[bool, list[list[int]]]:
    width, pitch0, asc, avail = _geometry(u)
    s = u["size"] * scale
    if u["align"] == "left":
        lines = _wrap(rt, s, width - u.get("hang", 0.0), width - u["indent"], cs)
    else:
        lines = _wrap(rt, s, width, cs=cs)
    need = asc * scale + (len(lines) - 1) * pitch0 * scale + 0.2 * s
    return need <= avail, lines


def fit_scale(u: dict, rt: Rich, lo: float, hi: float, cs: float = 0.0) -> float:
    """Largest scale in [lo, hi] at which the text fits its box (lo if it never fits)."""
    if _try_scale(u, rt, hi, cs)[0]:
        return hi
    if not _try_scale(u, rt, lo, cs)[0]:
        return lo
    a, b = lo, hi
    for _ in range(7):
        m = (a + b) / 2
        if _try_scale(u, rt, m, cs)[0]:
            a = m
        else:
            b = m
    return a


def capacity_chars(u: dict, scale: float, char_em: float) -> int:
    """Rough number of characters (each char_em wide) that fit the box at the given scale."""
    width, pitch0, asc, avail = _geometry(u)
    s = u["size"] * scale
    n_lines = max(1, int((avail - asc * scale - 0.2 * s) / (pitch0 * scale)) + 1)
    per_line = width / (char_em * s)
    return max(1, int(n_lines * per_line - (u["indent"] + u.get("hang", 0.0) * (n_lines - 1)) / (char_em * s)))


def style_key(u: dict) -> str:
    """Text style class: original size (0.5pt buckets) + bold."""
    return f"{round(u['size'] * 2) / 2:g}pt" + (" bold" if u["bold"] else "")


def layout_unit(u: dict, rt: Rich, min_scale: float, start_scale: float = 1.0, cs: float = 0.0):
    """Use the learned class scale; shrink further only if this unit does not fit at it."""
    width, pitch0, asc, avail = _geometry(u)
    start_scale = max(start_scale, min_scale)
    scale = fit_scale(u, rt, min_scale, start_scale, cs)
    lines = _try_scale(u, rt, scale, cs)[1]
    pitch = pitch0 * scale
    # still overflowing at the minimum scale: tighten line spacing as a last resort
    s = u["size"] * scale
    need = asc * scale + (len(lines) - 1) * pitch + 0.2 * s
    if need > avail and len(lines) > 1:
        pitch = max(s * 1.05, (avail - asc * scale - 0.2 * s) / (len(lines) - 1))
    return scale, lines, pitch, asc * scale, width


def _draw_line(page: pymupdf.Page, rt: Rich, idx: list[int], x: float, y: float, s: float, fx: float,
               fonts: FontPair, color: tuple, hl_color: tuple, cs: float) -> None:
    """Draw one laid-out line run by run (font, slant, script size, decorations)."""
    runs: list[tuple[frozenset, list[int]]] = []
    for k in idx:
        st = rt.styles[k]
        if runs and runs[-1][0] == st:
            runs[-1][1].append(k)
        else:
            runs.append((st, [k]))
    placed = []                 # (styles, indices, x start, x end) in unscaled coordinates
    cx = x
    for st, ks in runs:
        w = s * sum(rt.w1[k] for k in ks) + cs * s * len(ks)
        placed.append((st, ks, cx, cx + w - (cs * s if cs else 0.0)))
        cx += w

    def X(v: float) -> float:   # horizontal condensing around the line start
        return x + fx * (v - x)

    for st, ks, a, b in placed:
        if "highlight" in st and any(not rt.text[k].isspace() for k in ks):
            page.draw_rect(pymupdf.Rect(X(a), y - 0.9 * s, X(b), y + 0.25 * s), color=None,
                           fill=hl_color, fill_opacity=0.45, overlay=True)
    for st, ks, a, _ in placed:
        txt = "".join(rt.text[k] for k in ks)
        if not txt.strip():
            continue
        size, dy = s, 0.0
        if "sup" in st:
            size, dy = s * SCRIPT_SCALE, -0.35 * s
        elif "sub" in st:
            size, dy = s * SCRIPT_SCALE, 0.15 * s
        bold = "bold" in st
        real_bold = bold and fonts.bold is not None
        fake_bold = bold and fonts.bold is None
        style = dict(fontsize=size, fontname="TRbold" if real_bold else "TRreg", color=color,
                     render_mode=2 if fake_bold else 0,
                     border_width=0.03 if fake_bold else 0.05,
                     fill=color if fake_bold else None)
        mat = pymupdf.Matrix(fx, 1)
        if "italic" in st:
            mat = mat * pymupdf.Matrix(1, 0, ITALIC_SHEAR, 1, 0, 0)   # PDF space (y up): tops lean right
        morph = None if (fx == 1 and "italic" not in st) else (pymupdf.Point(x, y + dy), mat)
        if not cs:
            page.insert_text((a, y + dy), txt, morph=morph, **style)
        else:
            # letter spacing: place glyph by glyph (the morph pivot stays at the line start)
            gx = a
            for k in ks:
                ch = rt.text[k]
                if not ch.isspace():
                    page.insert_text((gx, y + dy), ch, morph=morph, **style)
                gx += rt.w1[k] * s + cs * s
    lw = max(0.4, 0.06 * s)
    for st, ks, a, b in placed:
        for key, off in (("underline", 0.13 * s), ("strike", -0.3 * s)):
            if key in st:
                page.draw_line((X(a), y + off), (X(b), y + off), color=color, width=lw, overlay=True)


def _remove_decorations(page: pymupdf.Page, units: list[dict]) -> None:
    """Remove underline / strikethrough lines and text-markup annotations of translated units;
    they are redrawn under the translated words."""
    rects = {tuple(r) for u in units for r in u.get("deco_rects", ())}
    if rects:
        for r in rects:
            rect = pymupdf.Rect(r)
            rect.x0 -= 0.6
            rect.y0 -= 0.6
            rect.x1 += 0.6
            rect.y1 += 0.6
            page.add_redact_annot(rect, fill=False)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                              text=pymupdf.PDF_REDACT_TEXT_NONE)
    for xref in sorted({x for u in units for x in u.get("annots", ())}):
        try:
            annot = page.load_annot(xref)
            if annot is not None:
                page.delete_annot(annot)
        except Exception:
            pass


def render_page(page: pymupdf.Page, units: list[dict], fonts: FontPair, min_scale: float,
                class_scales: dict[str, float] | None = None, default_scale: float = 1.0) -> dict:
    """Draw translations; returns layout stats {units, shrunk, condensed}."""
    stats = {"units": 0, "shrunk": 0, "condensed": 0}
    todo = [u for u in units if not u.get("skip") and u.get("translation")]
    if not todo:
        return stats
    if fonts.markup:
        _remove_decorations(page, todo)
    for u in todo:
        for r in u["rects"]:
            rect = pymupdf.Rect(r)
            h = rect.height
            rect.y0 += h * 0.2
            rect.y1 -= h * 0.2
            if rect.is_empty:
                continue
            page.add_redact_annot(rect, fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    fonts.install(page)
    cs = fonts.spacing

    for u in todo:
        rt = prepare(u, u["translation"].strip(), fonts)
        start = (class_scales or {}).get(style_key(u), default_scale)
        scale, lines, pitch, asc, width = layout_unit(u, rt, min_scale, start, cs)
        stats["units"] += 1
        if scale < max(start, min_scale) - 0.005:
            stats["shrunk"] += 1
        condensed = False
        s = u["size"] * scale
        x0, y0, x1, _ = u["bbox"]
        y = y0 + asc
        color = _rgb(u["color"])
        hl = _rgb(u.get("hl_color", DEFAULT_HIGHLIGHT))
        for i, line in enumerate(lines):
            if line:
                lw = _measure(rt, line, s, cs)
                left = x0
                if u["align"] == "left":
                    left += u["indent"] if i == 0 else u.get("hang", 0.0)
                avail_w = x1 - left
                if u["align"] == "center":
                    x = x0 + (width - lw) / 2
                elif u["align"] == "right":
                    x = x1 - lw
                else:
                    x = left
                fx = 1.0
                if lw > avail_w + 0.5:                       # condense horizontally instead of overflowing
                    fx = max(0.5, avail_w / lw)
                    if u["align"] != "left":
                        x = x0
                    condensed = True
                _draw_line(page, rt, line, x, y, s, fx, fonts, color, hl, cs)
            y += pitch
        if condensed:
            stats["condensed"] += 1
    return stats


def render_document(src_path: str, out_path: str, pages_units: dict[int, list[dict]],
                    fonts: FontPair, min_scale: float,
                    class_scales: dict[str, float] | None = None, default_scale: float = 1.0) -> dict:
    total = {"units": 0, "shrunk": 0, "condensed": 0}
    doc = pymupdf.open(src_path)
    try:
        for pno, units in pages_units.items():
            st = render_page(doc[pno], units, fonts, min_scale, class_scales, default_scale)
            for k in total:
                total[k] += st[k]
        try:
            doc.subset_fonts(fallback=True)   # fontTools subsetting (handles OTF/CFF)
        except Exception:
            pass
        doc.save(out_path, garbage=3, deflate=True)
    finally:
        doc.close()
    return total
