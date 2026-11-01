"""PDF text extraction into translatable units, and re-rendering translated text in place.

Only text is touched: original glyphs are removed with redactions (no fill, images and
vector graphics kept), then the translation is written into the same box with the same
baseline, size and colour, using one normalised font family.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import pymupdf

# Characters that may be broken anywhere when wrapping (CJK ideographs, kana, fullwidth forms).
_BREAK_ANY = r"⺀-⿿　-ヿ㄀-ㇿ㐀-䶿一-鿿豈-﫿︰-﹏＀-￯"
_TOKEN_RE = re.compile(rf"[^\s{_BREAK_ANY}]+\s*|\s+|.", re.S)
_CJK_CHAR = re.compile(rf"[{_BREAK_ANY}]")
# Line starts that open a new list item: "1.", "2)", "(3)", "①", "•", "-", "a." ...
LIST_RE = re.compile(r"^(\d{1,3}[.)．、](?!\d)|\(\d{1,3}\)|[①-⑳❶-❿➀-➓]|[•●▪■◆○◦·\-–—]\s|[a-zA-Z][.)]\s)")
NO_LINE_END = set("（「『【〔〈《‘“([{")
NO_LINE_START = set("，。、；：？！）」』】〕〉》’”…‧·,.;:?!)]}%")
FLAG_SUPERSCRIPT, FLAG_ITALIC, FLAG_SERIF, FLAG_BOLD = 1, 2, 4, 16


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


def _line_info(spans: list[dict]) -> _Line:
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
    return _Line(
        text=text,
        x0=min(xs0), y0=min(ys0), x1=max(xs1), y1=max(ys1),
        size=float(main["size"]),
        baseline=float(main["origin"][1]),
        color=int(main["color"]),
        bold=bold_chars / total > 0.6,
        serif=bool(main["flags"] & FLAG_SERIF),
        rects=[list(s["bbox"]) for s in spans if s["text"].strip()],
    )


def _merge_fragments(a: _Line, b: _Line) -> _Line:
    """Join two pieces of one visual line (justified text is often split at wide gaps)."""
    main = a if len(a.text.strip()) >= len(b.text.strip()) else b
    return _Line(
        text=a.text.rstrip() + " " + b.text.lstrip(),
        x0=min(a.x0, b.x0), y0=min(a.y0, b.y0), x1=max(a.x1, b.x1), y1=max(a.y1, b.y1),
        size=main.size, baseline=main.baseline, color=main.color,
        bold=a.bold and b.bold, serif=main.serif, rects=a.rects + b.rects,
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


def _join(lines: list[_Line]) -> str:
    out = lines[0].text.strip()
    for ln in lines[1:]:
        nxt = ln.text.strip()
        if not nxt:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and nxt[:1].islower():
            out = out[:-1] + nxt
        elif _CJK_CHAR.match(out[-1:] or " ") and _CJK_CHAR.match(nxt[:1]):
            out += nxt
        else:
            out += " " + nxt
    return out


def _spread(vals: list[float]) -> float:
    return max(vals) - min(vals) if vals else 0.0


def _make_unit(lines: list[_Line], page_rect: pymupdf.Rect, uid: str) -> dict:
    text = _join(lines)
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

    return {
        "id": uid,
        "text": text,
        "bbox": [round(v, 2) for v in (x0, y0, x1, y1)],
        "size": round(size, 2),
        "pitch": round(pitch, 2) if pitch else None,
        "first_baseline": round(lines[0].baseline, 2),
        "color": color,
        "bold": sum(c for l, c in zip(lines, chars) if l.bold) / sum(chars) > 0.5,
        "serif": lines[0].serif,
        "align": align,
        "indent": round(indent, 2),
        "hang": round(hang, 2),
        "rects": [r for l in lines for r in l.rects],
        "skip": not any(ch.isalpha() for ch in text),
        "translation": None,
    }


def extract_units(page: pymupdf.Page) -> list[dict]:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    d = page.get_text("dict", flags=flags)
    # 1) collect horizontal lines in reading order; None marks a hard separator
    seq: list[_Line | None] = []
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            dx, dy = ln["dir"]
            spans = [s for s in ln["spans"] if s["text"]]
            if abs(dy) > 0.01 or dx < 0 or not "".join(s["text"] for s in spans).strip():
                seq.append(None)    # non-horizontal / empty line: left untouched
                continue
            info = _line_info(spans)
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
    units = [_make_unit(g, page.rect, f"S{i}") for i, g in enumerate(groups, 1)]
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
    """

    def __init__(self, regular_path: str, bold_path: str, spacing: float = 0.0):
        self.regular_path, self.bold_path = regular_path, bold_path
        self.spacing = float(spacing or 0.0)
        self.regular = pymupdf.Font(fontfile=regular_path) if regular_path else pymupdf.Font("cjk")
        if bold_path and bold_path != regular_path:
            self.bold = pymupdf.Font(fontfile=bold_path)
        else:
            self.bold = None   # emulate bold with stroke

    @property
    def signature(self) -> str:
        return f"{self.regular_path}|{self.bold_path}|{self.spacing:.3f}"

    def font_for(self, u: dict) -> pymupdf.Font:
        return self.bold if (u["bold"] and self.bold is not None) else self.regular

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


def _wrap(text: str, font: pymupdf.Font, size: float, width: float, first_width: float | None = None,
          cs: float = 0.0) -> list[str]:
    """Greedy line breaking; CJK may break between any characters, other scripts at spaces."""
    lines: list[str] = []

    def limit() -> float:
        return first_width if (first_width is not None and not lines) else width

    def measure(t: str) -> float:
        return text_width(font, t.rstrip(), size, cs)

    for para in text.split("\n"):
        cur = ""
        for tok in _TOKEN_RE.findall(para):
            cand = cur + tok
            if not cur.strip() or measure(cand) <= limit():
                cur = cand
            elif tok[0] in NO_LINE_START and measure(cand) <= limit() + 1.05 * size:
                cur = cand                                    # keep closing punctuation on the line
                continue
            else:
                carry = ""
                while cur.rstrip() and cur.rstrip()[-1] in NO_LINE_END and len(cur.rstrip()) > 1:
                    carry = cur.rstrip()[-1] + carry          # opening brackets move to the next line
                    cur = cur.rstrip()[:-1]
                lines.append(cur.rstrip())
                cur = carry + tok.lstrip()
            # the current line is a single token wider than the box: hard-split by characters
            while measure(cur) > limit() and len(cur.rstrip()) > 1 and cur.rstrip() == tok.strip():
                cut = len(cur)
                while cut > 1 and text_width(font, cur[:cut], size, cs) > limit():
                    cut -= 1
                lines.append(cur[:cut])
                cur = tok = cur[cut:]
        lines.append(cur.rstrip())
    while len(lines) > 1 and not lines[-1]:
        lines.pop()
    return lines or [""]


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


def _try_scale(u: dict, text: str, font: pymupdf.Font, scale: float, cs: float = 0.0) -> tuple[bool, list[str]]:
    width, pitch0, asc, avail = _geometry(u)
    s = u["size"] * scale
    if u["align"] == "left":
        lines = _wrap(text, font, s, width - u.get("hang", 0.0), width - u["indent"], cs)
    else:
        lines = _wrap(text, font, s, width, cs=cs)
    need = asc * scale + (len(lines) - 1) * pitch0 * scale + 0.2 * s
    return need <= avail, lines


def fit_scale(u: dict, text: str, font: pymupdf.Font, lo: float, hi: float, cs: float = 0.0) -> float:
    """Largest scale in [lo, hi] at which the text fits its box (lo if it never fits)."""
    if _try_scale(u, text, font, hi, cs)[0]:
        return hi
    if not _try_scale(u, text, font, lo, cs)[0]:
        return lo
    a, b = lo, hi
    for _ in range(7):
        m = (a + b) / 2
        if _try_scale(u, text, font, m, cs)[0]:
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


def layout_unit(u: dict, text: str, font: pymupdf.Font, min_scale: float, start_scale: float = 1.0,
                cs: float = 0.0):
    """Use the learned class scale; shrink further only if this unit does not fit at it."""
    width, pitch0, asc, avail = _geometry(u)
    start_scale = max(start_scale, min_scale)
    scale = fit_scale(u, text, font, min_scale, start_scale, cs)
    lines = _try_scale(u, text, font, scale, cs)[1]
    pitch = pitch0 * scale
    # still overflowing at the minimum scale: tighten line spacing as a last resort
    s = u["size"] * scale
    need = asc * scale + (len(lines) - 1) * pitch + 0.2 * s
    if need > avail and len(lines) > 1:
        pitch = max(s * 1.05, (avail - asc * scale - 0.2 * s) / (len(lines) - 1))
    return scale, lines, pitch, asc * scale, width


def render_page(page: pymupdf.Page, units: list[dict], fonts: FontPair, min_scale: float,
                class_scales: dict[str, float] | None = None, default_scale: float = 1.0) -> dict:
    """Draw translations; returns layout stats {units, shrunk, condensed}."""
    stats = {"units": 0, "shrunk": 0, "condensed": 0}
    todo = [u for u in units if not u.get("skip") and u.get("translation")]
    if not todo:
        return stats
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
        text = u["translation"].strip()
        use_bold_font = u["bold"] and fonts.bold is not None
        font = fonts.font_for(u)
        fname = "TRbold" if use_bold_font else "TRreg"
        fake_bold = u["bold"] and fonts.bold is None
        start = (class_scales or {}).get(style_key(u), default_scale)
        scale, lines, pitch, asc, width = layout_unit(u, text, font, min_scale, start, cs)
        stats["units"] += 1
        if scale < max(start, min_scale) - 0.005:
            stats["shrunk"] += 1
        condensed = False
        s = u["size"] * scale
        x0, y0, x1, _ = u["bbox"]
        y = y0 + asc
        color = _rgb(u["color"])
        style = dict(fontsize=s, fontname=fname, color=color,
                     render_mode=2 if fake_bold else 0,
                     border_width=0.03 if fake_bold else 0.05,
                     fill=color if fake_bold else None)
        for i, line in enumerate(lines):
            if line:
                lw = text_width(font, line, s, cs)
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
                morph = None
                if lw > avail_w + 0.5:                       # condense horizontally instead of overflowing
                    fx = max(0.5, avail_w / lw)
                    if u["align"] != "left":
                        x = x0
                    morph = (pymupdf.Point(x, y), pymupdf.Matrix(fx, 1))
                    condensed = True
                if not cs:
                    page.insert_text((x, y), line, morph=morph, **style)
                else:
                    # letter spacing: place glyph by glyph (the morph pivot stays at the line start)
                    cx = x
                    for ch in line:
                        if not ch.isspace():
                            page.insert_text((cx, y), ch, morph=morph, **style)
                        cx += font.text_length(ch, s) + cs * s
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
