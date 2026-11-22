"""EPUB support: every spine document (XHTML chapter) is one "page" of the pipeline.

Units are the leaf block elements of a chapter (paragraphs, headings, list items, table cells...).
Inline formatting (b/strong, i/em, u, s/del, mark, sup, sub) is carried through translation with
the same markers as PDF text; on output the element's text is replaced by the translation and the
markers are written back as inline tags. Elements without text inside a unit (images, empty
anchors) are kept. Everything else in the book (CSS, images, OPF, navigation) is copied unchanged.
"""
from __future__ import annotations

import posixpath
import zipfile
from urllib.parse import unquote

from lxml import etree

from .markup import NEST_ORDER, styled_text, to_markup

_CONTAINER = "META-INF/container.xml"
_OPF_NS = "{http://www.idpf.org/2007/opf}"
_XHTML_NS = "http://www.w3.org/1999/xhtml"

BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "td", "th", "caption",
          "figcaption", "blockquote", "div", "section", "article", "aside", "header", "footer",
          "nav", "main", "figure", "ul", "ol", "dl", "table", "thead", "tbody", "tfoot", "tr",
          "hr", "address", "details", "summary", "body"}
SKIP = {"script", "style", "pre", "code", "svg", "math", "head", "title", "kbd", "samp", "rt", "rp"}
INLINE_STYLE = {"b": "bold", "strong": "bold", "i": "italic", "em": "italic", "cite": "italic",
                "dfn": "italic", "var": "italic", "u": "underline", "ins": "underline",
                "s": "strike", "strike": "strike", "del": "strike", "mark": "highlight",
                "sup": "sup", "sub": "sub"}
STYLE_TAG = {"bold": "b", "italic": "i", "underline": "u", "strike": "s", "highlight": "mark",
             "sup": "sup", "sub": "sub"}


def is_epub(path: str) -> bool:
    return str(path).lower().endswith(".epub")


def _local(el) -> str:
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _parser() -> etree.XMLParser:
    return etree.XMLParser(resolve_entities=False, no_network=True, recover=True, remove_blank_text=False)


def spine_docs(zf: zipfile.ZipFile) -> list[str]:
    """Zip paths of the spine's (X)HTML documents in reading order."""
    cont = etree.fromstring(zf.read(_CONTAINER), _parser())
    rf = next(el for el in cont.iter() if _local(el) == "rootfile")
    opf_path = rf.get("full-path")
    opf = etree.fromstring(zf.read(opf_path), _parser())
    base = posixpath.dirname(opf_path)
    manifest = {}
    for it in opf.iter(f"{_OPF_NS}item"):
        manifest[it.get("id")] = (it.get("href", ""), it.get("media-type", ""))
    docs = []
    names = set(zf.namelist())
    for ref in opf.iter(f"{_OPF_NS}itemref"):
        href, mt = manifest.get(ref.get("idref"), ("", ""))
        if "html" not in mt:
            continue
        p = posixpath.normpath(posixpath.join(base, unquote(href)))
        if p in names and p not in docs:
            docs.append(p)
    return docs


def page_count(path: str) -> int:
    with zipfile.ZipFile(path) as zf:
        return len(spine_docs(zf))


def _has_block_child(el) -> bool:
    return any(_local(c) in BLOCKS for c in el.iterchildren())


def _has_text(el) -> bool:
    return bool("".join(t for t in el.itertext() if t).strip())


def _leaf_blocks(root) -> list:
    body = next((el for el in root.iter() if _local(el) == "body"), root)
    out = []

    def walk(el):
        name = _local(el)
        if name in SKIP or not name:
            return
        if _has_block_child(el):
            for c in el.iterchildren():
                walk(c)
        elif name in BLOCKS and _has_text(el):
            out.append(el)

    walk(body)
    return out


def _collect(el, cur: frozenset, chars: list[str], styles: list[frozenset]) -> None:
    def add(text):
        if text:
            for ch in text.replace("\r", "").replace("\n", " ").replace("\t", " "):
                chars.append(ch)
                styles.append(cur)

    add(el.text)
    for c in el.iterchildren():
        name = _local(c)
        if name == "br":
            chars.append("\n")
            styles.append(frozenset())
        elif name and name not in SKIP:
            _collect(c, cur | ({INLINE_STYLE[name]} if name in INLINE_STYLE else set()), chars, styles)
        if c.tail:
            for ch in c.tail.replace("\r", "").replace("\n", " ").replace("\t", " "):
                chars.append(ch)
                styles.append(cur)


def _normalize_ws(chars: list[str], styles: list[frozenset]) -> tuple[str, list[frozenset]]:
    out_c, out_s = [], []
    for ch, st in zip(chars, styles):
        if ch == " " and (not out_c or out_c[-1] in " \n"):
            continue
        if ch == "\n" and out_c and out_c[-1] == " ":
            out_c.pop()
            out_s.pop()
        out_c.append(ch)
        out_s.append(st)
    while out_c and out_c[-1] in " \n":
        out_c.pop()
        out_s.pop()
    return "".join(out_c), out_s


def _make_unit(el, uid: str, tree, markup: bool) -> dict:
    chars: list[str] = []
    styles: list[frozenset] = []
    _collect(el, frozenset(), chars, styles)
    text, styles = _normalize_ws(chars, styles)
    if not markup:
        styles = [frozenset()] * len(text)
    used = set().union(*styles) if styles else set()
    marks = sorted(used, key=NEST_ORDER.index)
    unit = {
        "id": uid,
        "text": to_markup(text, styles) if marks else text,
        "xpath": tree.getpath(el),
        "tag": _local(el),
        "bold": False, "italic": False,
        "pitch": None, "size": 0,
        "skip": not any(ch.isalpha() for ch in text),
        "translation": None,
    }
    if marks:
        unit["marks"] = marks
    return unit


def _parse_doc(data: bytes):
    return etree.ElementTree(etree.fromstring(data, _parser()))


def extract_units(path: str, index: int, markup: bool = True) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = spine_docs(zf)[index]
        tree = _parse_doc(zf.read(name))
    return [_make_unit(el, f"S{i}", tree, markup) for i, el in enumerate(_leaf_blocks(tree.getroot()), 1)]


def _write_unit(el, u: dict, markup: bool) -> None:
    text, styles = styled_text(u, u["translation"].strip(), inline=markup)
    ns = el.tag.split("}")[0][1:] if el.tag.startswith("{") else None
    q = (lambda t: f"{{{ns}}}{t}") if ns else (lambda t: t)
    # keep children without text (images, empty anchors...) at the start of the element
    keep = [c for c in el.iterchildren() if isinstance(c.tag, str) and _local(c) != "br" and not _has_text(c)]
    for c in list(el):
        el.remove(c)
    el.text = None
    for c in keep:
        c.tail = None
        el.append(c)

    def append_text(parent, s):
        last = parent[-1] if len(parent) else None
        if last is None:
            parent.text = (parent.text or "") + s
        else:
            last.tail = (last.tail or "") + s

    # stack of (style key, element); el itself is the bottom
    stack: list[tuple[str, object]] = []
    for ch, st in zip(text, styles):
        want = [k for k in NEST_ORDER if k in st]
        n = 0
        while n < len(stack) and n < len(want) and stack[n][0] == want[n]:
            n += 1
        del stack[n:]
        for k in want[n:]:
            parent = stack[-1][1] if stack else el
            child = etree.SubElement(parent, q(STYLE_TAG[k]))
            stack.append((k, child))
        parent = stack[-1][1] if stack else el
        if ch == "\n":
            etree.SubElement(parent, q("br"))
        else:
            append_text(parent, ch)


def render_document(src_path: str, out_path: str, pages_units: dict[int, list[dict]],
                    markup: bool = True, target_lang: str = "") -> dict:
    total = {"units": 0, "shrunk": 0, "condensed": 0}
    with zipfile.ZipFile(src_path) as zin:
        docs = spine_docs(zin)
        replaced: dict[str, bytes] = {}
        for pno, units in pages_units.items():
            if pno >= len(docs):
                continue
            todo = [u for u in units if not u.get("skip") and u.get("translation")]
            if not todo:
                continue
            name = docs[pno]
            tree = _parse_doc(zin.read(name))
            for u in todo:
                found = tree.xpath(u["xpath"])
                if not found:
                    continue
                _write_unit(found[0], u, markup)
                total["units"] += 1
            replaced[name] = etree.tostring(tree, xml_declaration=True, encoding="utf-8")
        with zipfile.ZipFile(out_path, "w") as zout:
            infos = zin.infolist()
            infos.sort(key=lambda i: i.filename != "mimetype")   # mimetype must come first, stored
            for info in infos:
                data = replaced.get(info.filename, zin.read(info.filename))
                ct = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
                zout.writestr(info.filename, data, compress_type=ct)
    return total
