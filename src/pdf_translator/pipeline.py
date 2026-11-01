"""Translation job: concurrency, pause/resume, persistent state and the check-and-fix loop."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import pymupdf

from .config import WORK_DIR, Settings, resolve_fonts
from .llm import LLMClient, LLMError, Prompt, RetryableError, Usage
from .layout_rules import learn, length_limits, summary, target_char_em, unit_fit
from .pdf_layout import FontPair, extract_units, render_document

STATE_VERSION = 1

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def check_schema(with_reason: bool) -> dict:
    item_props = {"batch": {"type": "integer"}, "start": {"type": "string"}, "end": {"type": "string"}}
    if with_reason:
        item_props["reason"] = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "pass": {"type": "boolean"},
            "rejected": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": list(item_props),
                    "additionalProperties": False,
                },
            },
        },
        "required": ["pass", "rejected"],
        "additionalProperties": False,
    }


# ------------------------------------------------------------------ helpers

def parse_page_range(spec: str, page_count: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(page_count))
    pages: set[int] = set()
    for part in re.split(r"[,\s，]+", spec):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a.strip() else 1
            end = int(b) if b.strip() else page_count
            pages.update(range(max(1, start), min(page_count, end) + 1))
        else:
            n = int(part)
            if 1 <= n <= page_count:
                pages.add(n)
    return sorted(p - 1 for p in pages)


def work_dir_for(pdf_path: str, target_lang: str) -> Path:
    h = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()[:10]
    t = hashlib.sha1(target_lang.encode("utf-8")).hexdigest()[:6]
    return WORK_DIR / f"{Path(pdf_path).stem}_{h}_{t}"


def default_output_path(pdf_path: str) -> str:
    p = Path(pdf_path)
    return str(p.with_name(f"{p.stem}_translated.pdf"))


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
        return st if st.get("version") == STATE_VERSION else None
    except Exception:
        return None


def make_fonts(settings: Settings) -> FontPair:
    return FontPair(*resolve_fonts(settings), settings.char_spacing)


def render_from_state(settings: Settings, state: dict, out_path: str) -> tuple[dict, dict]:
    """Render all translated pages from saved translations (no API calls).

    Class scales are re-learned from the whole document with the current font / spacing /
    scale settings. Returns (rules, layout stats).
    """
    fonts = make_fonts(settings)
    pages_units = {int(k): v["units"] for k, v in state["pages"].items()}
    rules = learn(pages_units.values(), fonts, settings.min_font_scale, settings.max_font_scale)
    stats = render_document(state["source"], out_path, pages_units, fonts, settings.min_font_scale,
                            rules["scales"], settings.max_font_scale)
    return rules, stats


def layout_report(stats: dict) -> str:
    return (f"排版 {stats['units']} 段：{stats['shrunk']} 段低於樣式比例再縮小、"
            f"{stats['condensed']} 段水平壓縮")


class DynamicLimiter:
    """Semaphore whose limit is read from settings each time (changes apply on the fly)."""

    def __init__(self, get_limit: Callable[[], int]):
        self.get_limit = get_limit
        self.active = 0
        self.cond = asyncio.Condition()

    async def __aenter__(self):
        async with self.cond:
            await self.cond.wait_for(lambda: self.active < max(1, int(self.get_limit())))
            self.active += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.active -= 1
            self.cond.notify_all()

    async def poke(self):
        async with self.cond:
            self.cond.notify_all()


def _seg_lines(units: list[dict], field: str = "text") -> str:
    return "\n".join(f"[{u['id']}] {u.get(field) or ''}" for u in units)


def _by_batch(units: list[dict], field: str) -> str:
    out: list[str] = []
    batches: dict[int, list[dict]] = {}
    for u in units:
        batches.setdefault(u["batch"], []).append(u)
    for b, us in sorted(batches.items()):
        out.append(f"=== Batch {b} ({us[0]['id']}–{us[-1]['id']}) ===")
        out.append(_seg_lines(us, field))
    return "\n".join(out)


_ID_PREFIX = re.compile(r"^\s*\[S\d+\]\s*")


# ------------------------------------------------------------------ job

class Job:
    def __init__(self, settings: Settings, emit: Callable[[dict], None]):
        self.settings = settings.copy()
        self.emit = emit
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.state: dict = {}
        self.state_path: Path | None = None
        self.log_path: Path | None = None
        self.usage = {"translator": Usage(), "checker": Usage()}
        self.paused = False
        self._clients: dict[str, LLMClient] = {}
        self._all_clients: list[LLMClient] = []
        self._last_save = 0.0
        self._save_pending = False
        self._fatal_msgs: set[str] = set()
        self._tasks: list[asyncio.Task] = []

    # ============ thread-facing API (called from the UI thread) ============
    def start(self) -> None:
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def pause(self) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self._do_pause, "使用者暫停")

    def resume(self, new_settings: Settings | None = None) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self._do_resume, new_settings.copy() if new_settings else None)

    def stop(self) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self._do_stop)

    def render_now(self, layout_settings: Settings | None = None) -> None:
        """Re-layout and write the output now; layout_settings lets a paused job preview new
        font / spacing / scale settings before they are applied with resume()."""
        if self.loop:
            s = None
            if layout_settings is not None:
                s = layout_settings.copy()
                for k in ("input_pdf", "output_pdf", "page_range", "target_lang"):
                    setattr(s, k, getattr(self.settings, k))
            asyncio.run_coroutine_threadsafe(self._render("重新排版輸出", s), self.loop)

    def is_alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    # ============ internals ============
    def log(self, msg: str, level: str = "info") -> None:
        line = time.strftime("%H:%M:%S ") + msg
        self.emit({"type": "log", "msg": line, "level": level})
        if self.log_path:
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as e:  # pragma: no cover - last resort
            self.log(f"未預期錯誤: {e}\n{traceback.format_exc()}", "error")
            self.emit({"type": "state", "state": "error"})
        finally:
            try:
                self.loop.run_until_complete(self._close_clients())
            finally:
                self.loop.close()

    def _do_pause(self, reason: str) -> None:
        if not self.paused:
            self.paused = True
            self._gate.clear()
            self.log(f"已暫停（{reason}）。進行中的請求會跑完，新請求暫停送出。", "warn")
            self.emit({"type": "state", "state": "paused"})

    def _do_resume(self, new_settings: Settings | None) -> None:
        if new_settings is not None:
            # document-level fields stay fixed during a job
            for k in ("input_pdf", "output_pdf", "page_range", "target_lang"):
                setattr(new_settings, k, getattr(self.settings, k))
            self.settings = new_settings
            self._clients.clear()          # rebuilt lazily with new model settings
            self._fonts = make_fonts(self.settings)
            self._relearn()
            self._fatal_msgs.clear()
            self.log("已套用新設定。")
        self.paused = False
        self._gate.set()
        asyncio.ensure_future(self._limiter.poke())
        asyncio.ensure_future(self._page_limiter.poke())
        self.emit({"type": "state", "state": "running"})

    def _do_stop(self) -> None:
        self.log("停止中…進度已保存，下次按「開始」即可從中斷處繼續。", "warn")
        self._stopping = True
        self._gate.set()
        for t in self._tasks:
            t.cancel()

    async def _close_clients(self) -> None:
        for c in self._all_clients:
            await c.close()

    def _client(self, role: str) -> LLMClient:
        c = self._clients.get(role)
        if c is None:
            c = LLMClient(getattr(self.settings, role))
            self._clients[role] = c
            self._all_clients.append(c)
        return c

    # ---------- persistence ----------
    def _save(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_save < 1.5:
            if not self._save_pending:
                self._save_pending = True
                self.loop.call_later(1.6, self._save_deferred)
            return
        self._last_save = now
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _save_deferred(self) -> None:
        self._save_pending = False
        self._save(force=True)

    # ---------- main ----------
    async def _main(self) -> None:
        s = self.settings
        self._gate = asyncio.Event()
        self._gate.set()
        self._stopping = False
        self._limiter = DynamicLimiter(lambda: self.settings.request_concurrency)
        self._fonts = make_fonts(s)
        self._rules: dict = {"scales": {}, "counts": {}, "ratio": None}
        self._rules_text = ""
        self._page_limiter = DynamicLimiter(lambda: self.settings.page_concurrency)

        if not s.input_pdf or not Path(s.input_pdf).exists():
            self.log("找不到輸入 PDF。", "error")
            self.emit({"type": "state", "state": "finished"})
            return
        wd = work_dir_for(s.input_pdf, s.target_lang)
        wd.mkdir(parents=True, exist_ok=True)
        self.state_path = wd / "state.json"
        self.log_path = wd / "log.txt"
        out_path = s.output_pdf.strip() or default_output_path(s.input_pdf)
        self.settings.output_pdf = out_path

        doc = pymupdf.open(s.input_pdf)
        st = load_state(self.state_path)
        if st is None:
            st = {"version": STATE_VERSION, "source": str(Path(s.input_pdf).resolve()),
                  "target_lang": s.target_lang, "page_count": doc.page_count, "pages": {}}
        else:
            st["source"] = str(Path(s.input_pdf).resolve())
        self.state = st
        try:
            pages = parse_page_range(s.page_range, doc.page_count)
        except ValueError:
            self.log(f"頁碼範圍格式錯誤: {s.page_range!r}", "error")
            self.emit({"type": "state", "state": "finished"})
            return
        for pno in pages:
            if str(pno) not in st["pages"]:
                st["pages"][str(pno)] = {"status": "pending", "units": extract_units(doc[pno]),
                                         "fix_rounds": 0, "warnings": [], "error": None}
        doc.close()
        self._relearn()
        self._save(force=True)

        done = sum(1 for p in pages if st["pages"][str(p)]["status"] == "done")
        self._pages = pages
        self.log(f"工作目錄: {wd}")
        if done:
            self.log(f"從中斷處恢復：{done}/{len(pages)} 頁已完成。")
        self.emit({"type": "state", "state": "running"})
        self._emit_progress()

        self._tasks = [asyncio.create_task(self._process_page(p)) for p in pages
                       if st["pages"][str(p)]["status"] != "done"]
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._save(force=True)

        if self._stopping:
            self.emit({"type": "state", "state": "stopped"})
            return
        failed = [p + 1 for p in pages if st["pages"][str(p)]["status"] != "done"]
        await self._render("輸出最終結果")
        if failed:
            self.log(f"完成，但以下頁面失敗（再按「開始」會重試）: {failed}", "warn")
        else:
            self.log("全部完成。")
        warn_pages = [p + 1 for p in pages if st["pages"][str(p)].get("warnings")]
        if warn_pages:
            self.log(f"檢查未完全通過的頁面: {warn_pages}（詳見 log.txt）", "warn")
        self.emit({"type": "state", "state": "finished"})

    async def _render(self, label: str, settings: Settings | None = None) -> None:
        settings = settings or self.settings
        out = settings.output_pdf
        try:
            rules, stats = await self.loop.run_in_executor(None, render_from_state, settings, self.state, out)
            self.log(f"{label}: {out}（{layout_report(stats)}）")
            self.emit({"type": "rules", "text": summary(rules)})
            self.emit({"type": "output", "path": out})
        except Exception as e:
            self.log(f"輸出 PDF 失敗: {e}", "error")

    def _emit_progress(self) -> None:
        pages = self._pages
        sts = [self.state["pages"][str(p)]["status"] for p in pages]
        self.emit({"type": "progress", "done": sts.count("done"), "total": len(pages),
                   "error": sts.count("error"),
                   "usage": {k: vars(v).copy() for k, v in self.usage.items()}})

    def _set_status(self, pno: int, status: str) -> None:
        self.state["pages"][str(pno)]["status"] = status
        self.emit({"type": "page", "page": pno + 1, "status": status})
        self._save()
        self._emit_progress()

    # ---------- API call with retry / pause ----------
    async def _call(self, role: str, prompt_fn: Callable[[], Prompt], label: str) -> dict:
        attempt = 0
        while True:
            await self._gate.wait()
            if self._stopping:
                raise asyncio.CancelledError
            async with self._limiter:
                await self._gate.wait()
                if self._stopping:
                    raise asyncio.CancelledError
                try:
                    client = self._client(role)
                    data, usage = await client.complete_json(prompt_fn())
                except LLMError as e:
                    if e.fatal:
                        msg = f"[{role}] {e}"
                        if msg not in self._fatal_msgs:
                            self._fatal_msgs.add(msg)
                            self.log(f"{msg} — 請調整設定後按「繼續」。", "error")
                        self._do_pause("設定錯誤")
                        continue
                    raise
                except RetryableError as e:
                    attempt += 1
                    if attempt >= max(1, self.settings.max_retries):
                        raise
                    wait = min(60, 3 * 2 ** (attempt - 1))
                    self.log(f"{label} 失敗（第 {attempt} 次）：{e}；{wait}s 後重試", "warn")
                else:
                    self.usage[role].add(usage)
                    self._emit_progress()
                    return data
            await asyncio.sleep(wait)

    # ---------- prompts ----------
    def _src_lang(self) -> str:
        sl = self.settings.source_lang.strip()
        return "the source language (detect it automatically)" if sl in ("", "自動偵測", "auto") else sl

    def _translator_system(self) -> str:
        s = self.settings
        text = (
            f"You are a professional translator of PDF documents. Translate from {self._src_lang()} "
            f"into {s.target_lang}.\n"
            "Rules:\n"
            "- Text is split into segments [S#] following the PDF layout: paragraphs, headings, table cells, "
            "captions, headers/footers or fragments. Translate each segment into its own slot. Never move "
            "content between segments, never merge or split segments.\n"
            "- The full page text is given only as context, translate only the requested segment ids.\n"
            "- Keep numbers, units, formulas, citation/footnote markers (e.g. 1), [3], *, †), URLs, e-mail "
            "addresses and code unchanged. Keep proper names in their usual form in the target language.\n"
            "- The translation is typeset into the original space: be faithful but concise and natural. "
            "No explanations, notes or added content.\n"
            "- A segment that is already in the target language or not translatable is returned unchanged.\n"
            "- Output only the JSON object."
        )
        if s.extra_instructions.strip():
            text += "\n\nAdditional instructions / glossary:\n" + s.extra_instructions.strip()
        return text

    def _page_context(self, pno: int) -> str:
        units = self.state["pages"][str(pno)]["units"]
        return (f"<page number=\"{pno + 1}\" total=\"{self.state['page_count']}\">\n"
                f"{_seg_lines(units)}\n</page>")

    def _apply_translations(self, units_by_id: dict, data: dict, wanted: set[str]) -> set[str]:
        got: set[str] = set()
        for item in data.get("translations", []) or []:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("id", "")).strip().strip("[]")
            text = _ID_PREFIX.sub("", str(item.get("text", ""))).strip()
            if uid in wanted and text:
                units_by_id[uid]["translation"] = text
                got.add(uid)
        return wanted - got

    async def _translate_ids(self, pno: int, ids: list[str], task_header: str, label: str,
                             extra: str = "") -> list[str]:
        page = self.state["pages"][str(pno)]
        by_id = {u["id"]: u for u in page["units"]}
        missing = list(ids)
        for _ in range(2):
            segs = [by_id[i] for i in missing]

            def make_prompt(segs=segs) -> Prompt:
                task = (f"{task_header}\n{extra}"
                        f"Segments to translate into {self.settings.target_lang}:\n{_seg_lines(segs)}\n\n"
                        f"{self._limits_text(segs)}"
                        'Return JSON {"translations": [{"id": "S#", "text": "..."}]} containing exactly '
                        f"these ids: {', '.join(u['id'] for u in segs)}")
                return Prompt(self._translator_system(), self._page_context(pno), task, TRANSLATION_SCHEMA)

            data = await self._call("translator", make_prompt, label)
            missing = sorted(self._apply_translations(by_id, data, set(missing)),
                             key=lambda x: int(x[1:]))
            self._save()
            if not missing:
                break
            self.log(f"{label}: 缺少 {missing}，補翻中", "warn")
        return missing

    # ---------- learned layout rules ----------
    def _relearn(self) -> None:
        s = self.settings
        self._rules = learn((p["units"] for p in self.state["pages"].values()), self._fonts,
                            s.min_font_scale, s.max_font_scale)
        self.state["layout_rules"] = self._rules
        text = summary(self._rules)
        if text and text != self._rules_text:
            self._rules_text = text
            self.emit({"type": "rules", "text": text})

    def _limits_text(self, segs: list[dict]) -> str:
        s = self.settings
        if not s.length_hints:
            return ""
        limits = length_limits(segs, self._rules, s.max_font_scale, target_char_em(s.target_lang))
        if not limits:
            return ""
        return ("Space limits: approximate maximum characters that fit the original space of these "
                "segments. An upper bound, not a target - stay within it without dropping meaning:\n"
                + "\n".join(f"{k} <= {v}" for k, v in limits.items()) + "\n\n")

    async def _shorten_page(self, pno: int) -> None:
        """Ask for a more concise translation of units that do not fit at full size."""
        s = self.settings
        lo, hi = s.min_font_scale, s.max_font_scale
        units = [u for u in self.state["pages"][str(pno)]["units"]
                 if not u["skip"] and u.get("translation") and not u.get("shortened")
                 # short labels / untranslated text (overflow comes from font width) cannot be condensed
                 and len(u["translation"]) >= 12 and u["translation"].strip() != u["text"].strip()
                 and u["pitch"] is not None]          # only multi-line paragraphs are rewritten
        over = []
        for u in units:
            f = unit_fit(u, self._fonts, lo, hi)
            if f < hi * 0.95:
                limit = max(2, int(len(u["translation"]) * (f / hi) ** 2 * 0.95))
                over.append((u, limit))
        if not over:
            return
        by_id = {u["id"]: u for u, _ in over}
        segs_txt = "\n".join(
            f"[{u['id']}] (max {lim} chars)\n  source: {u['text']}\n  current: {u['translation']}"
            for u, lim in over)

        def make_prompt() -> Prompt:
            task = (f"These translations on page {pno + 1} are too long to fit their space in the page layout "
                    "at the original font size. Rewrite each one more concisely within its character limit. "
                    "Keep the complete meaning, all facts, numbers, names and markers; only tighten the wording."
                    f"\n\n{segs_txt}\n\n"
                    'Return JSON {"translations": [{"id": "S#", "text": "..."}]} containing exactly these ids: '
                    + ", ".join(by_id))
            return Prompt(self._translator_system(), self._page_context(pno), task, TRANSLATION_SCHEMA)

        data = await self._call("translator", make_prompt, f"第 {pno + 1} 頁 精簡")
        shorter = 0
        for item in data.get("translations", []) or []:
            if not isinstance(item, dict):
                continue
            u = by_id.get(str(item.get("id", "")).strip().strip("[]"))
            text = _ID_PREFIX.sub("", str(item.get("text", ""))).strip()
            # accept only a real condensation, never one that drops most of the content
            if u is not None and text and 0.5 * len(u["translation"]) <= len(text) < len(u["translation"]):
                u["translation"] = text
                shorter += 1
        for u, _ in over:
            u["shortened"] = True
        self._save()
        self.log(f"第 {pno + 1} 頁 {len(over)} 段譯文超出原版面，已精簡 {shorter} 段")

    # ---------- page processing ----------
    def _assign_batches(self, units: list[dict]) -> None:
        s = self.settings
        todo = [u for u in units if not u["skip"]]
        if all("batch" in u for u in todo):
            return
        b, chars, count = 1, 0, 0
        for u in todo:
            n = len(u["text"])
            if count and (chars + n > s.batch_max_chars or count >= s.batch_max_segments):
                b, chars, count = b + 1, 0, 0
            u["batch"] = b
            chars += n
            count += 1

    async def _process_page(self, pno: int) -> None:
        try:
            async with self._page_limiter:
                await self._process_page_inner(pno)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            page = self.state["pages"][str(pno)]
            page["error"] = str(e)
            self.log(f"第 {pno + 1} 頁失敗: {e}", "error")
            self._set_status(pno, "error")

    async def _process_page_inner(self, pno: int) -> None:
        page = self.state["pages"][str(pno)]
        units = page["units"]
        todo = [u for u in units if not u["skip"]]
        if not todo:
            self._set_status(pno, "done")
            return
        self._assign_batches(units)
        nb = max(u["batch"] for u in todo)

        # ---- 1) translation ----
        if page["status"] in ("pending", "error", "translating"):
            page["error"] = None
            self._set_status(pno, "translating")
            jobs = []
            for b in range(1, nb + 1):
                ids = [u["id"] for u in todo if u["batch"] == b and not u.get("translation")]
                if ids:
                    jobs.append(self._translate_ids(
                        pno, ids, f"Translate batch {b}/{nb} of page {pno + 1}.",
                        f"第 {pno + 1} 頁 批次 {b}/{nb}"))
            results = await asyncio.gather(*jobs)
            missing = [m for r in results for m in r]
            if missing:
                raise RuntimeError(f"模型未回傳段落 {missing}")
            if self.settings.shorten_overflow:
                await self._shorten_page(pno)
            self._set_status(pno, "translated")
            self.log(f"第 {pno + 1} 頁翻譯完成（{len(todo)} 段，{nb} 批）")

        # ---- 2) check & fix loop ----
        if self.settings.enable_check and page["status"] in ("translated", "checking"):
            self._set_status(pno, "checking")
            while True:
                rejected = await self._check_page(pno)
                if not rejected:
                    self.log(f"第 {pno + 1} 頁檢查通過")
                    break
                if page["fix_rounds"] >= self.settings.max_check_rounds:
                    msg = f"已達修復上限 {self.settings.max_check_rounds} 輪，仍被否決: " + \
                          ", ".join(f"{r['start']}-{r['end']}" for r in rejected)
                    page["warnings"].append(msg)
                    self.log(f"第 {pno + 1} 頁 {msg}", "warn")
                    break
                page["fix_rounds"] += 1
                self.log(f"第 {pno + 1} 頁 第 {page['fix_rounds']} 輪修復: " +
                         "; ".join(f"{r['start']}-{r['end']} {r.get('reason', '')}".strip() for r in rejected),
                         "warn")
                await asyncio.gather(*(self._fix_range(pno, r) for r in rejected))
                self._save()
        self._relearn()
        self._set_status(pno, "done")

    async def _check_page(self, pno: int) -> list[dict]:
        s = self.settings
        units = [u for u in self.state["pages"][str(pno)]["units"] if not u["skip"]]
        reason_rule = ('Give a short "reason" (max 20 words) for each rejected range.'
                       if s.check_with_reason else "Do not explain, output only the verdict and ranges.")
        system = (
            "You are a translation QA reviewer. You get the full source text of one PDF page and its full "
            f"translation into {s.target_lang}. Both are split into segments [S#], grouped by translation batch.\n"
            "Do a quick, coarse check that the translation matches the source. Reject only real problems:\n"
            "- off-topic or hallucinated content, or text that does not correspond to its source segment\n"
            "- mistranslations that change the meaning (numbers, negation, facts, key terms)\n"
            "- substantive omissions or additions\n"
            "- segments left untranslated that should be translated\n"
            "Do NOT reject for style, wording preferences, terminology variants or punctuation.\n"
            'Output JSON {"pass": true|false, "rejected": [{"batch": n, "start": "S#", "end": "S#"'
            + (', "reason": "..."' if s.check_with_reason else "") + "}]}.\n"
            "Each rejected range is a contiguous range of segment ids inside one batch. "
            "If pass is true, rejected must be empty. " + reason_rule
        )
        context = f"<source page=\"{pno + 1}\">\n{_by_batch(units, 'text')}\n</source>"

        def make_prompt() -> Prompt:
            task = (f"<translation page=\"{pno + 1}\">\n{_by_batch(units, 'translation')}\n</translation>\n\n"
                    "Does the translation match the source? Answer with the JSON verdict.")
            return Prompt(system, context, task, check_schema(s.check_with_reason))

        data = await self._call("checker", make_prompt, f"第 {pno + 1} 頁檢查")
        if data.get("pass") is True and not data.get("rejected"):
            return []
        index = {u["id"]: i for i, u in enumerate(units)}
        spans: list[tuple[int, int, str]] = []
        for r in data.get("rejected", []) or []:
            a, b = str(r.get("start", "")).strip("[]"), str(r.get("end", "")).strip("[]")
            if a not in index and b not in index:
                continue
            ia, ib = index.get(a, index.get(b)), index.get(b, index.get(a))
            if ia > ib:
                ia, ib = ib, ia
            spans.append((ia, ib, str(r.get("reason", ""))))
        if not spans:
            if data.get("pass") is False:
                self.log(f"第 {pno + 1} 頁檢查判定否決但未給出有效段落，視為通過", "warn")
            return []
        spans.sort()
        merged: list[list] = []
        for ia, ib, reason in spans:                       # merge overlapping ranges
            if merged and ia <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], ib)
                merged[-1][2] = "; ".join(x for x in (merged[-1][2], reason) if x)
            else:
                merged.append([ia, ib, reason])
        return [{"start": units[a]["id"], "end": units[b]["id"], "reason": reason,
                 "ids": [u["id"] for u in units[a:b + 1]]} for a, b, reason in merged]

    async def _fix_range(self, pno: int, r: dict) -> None:
        page = self.state["pages"][str(pno)]
        by_id = {u["id"]: u for u in page["units"]}
        segs = [by_id[i] for i in r["ids"]]
        extra = (f"A reviewer rejected the current translation of segments {r['start']}–{r['end']} on this page."
                 + (f"\nReviewer note: {r['reason']}" if r.get("reason") else "")
                 + f"\nRejected translation:\n{_seg_lines(segs, 'translation')}\n"
                 "Re-translate ONLY these segments and fix the problem.\n\n")
        missing = await self._translate_ids(pno, r["ids"], f"Fix the translation of page {pno + 1}.",
                                            f"第 {pno + 1} 頁 修復 {r['start']}-{r['end']}", extra)
        if missing:
            self.log(f"第 {pno + 1} 頁修復時缺少 {missing}，保留原翻譯", "warn")
