"""Batch runner: translate (or only re-layout) several PDFs with shared settings.

Up to `settings.file_concurrency` books run at the same time (a change applies on resume);
inside each book the usual per-book page / request concurrency applies. Every book is an
ordinary Job with its own work dir / state.json, so a stopped or crashed batch resumes per
file. Pause / resume / stop / render_now are forwarded to every running book, so the UI can
treat a Batch like a Job.
"""
from __future__ import annotations

import glob
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import pymupdf

from .config import Settings
from .layout_rules import summary
from .pipeline import Job, layout_report, load_state, parse_page_range, render_from_state, work_dir_for

OUTPUT_SUFFIX = "_translated"   # default batch suffix; same as pipeline.default_output_path()

STATUS_LABELS = {"pending": "等待中", "running": "處理中", "done": "完成", "partial": "部分失敗",
                 "failed": "失敗", "stopped": "已停止", "skipped": "已略過"}


@dataclass
class BatchItem:
    input_pdf: str
    output_pdf: str
    status: str = "pending"     # pending | running | done | partial | failed | stopped | skipped
    message: str = ""


_BAD_NAME_CHARS = set('<>:"/\\|?*')


def check_suffix(suffix: str, output_dir: str = "") -> str | None:
    """Return an error message when the output-name suffix is unusable, else None."""
    if any(c in _BAD_NAME_CHARS or ord(c) < 32 for c in suffix):
        return f"輸出檔名後綴含有檔名不允許的字元（< > : \" / \\ | ? *）：{suffix!r}"
    if not suffix and not output_dir.strip():
        return "輸出檔名後綴空白時必須指定輸出資料夾，否則會覆蓋原檔"
    return None


def _skip_suffixes(suffix: str) -> tuple[str, ...]:
    """Name endings treated as previous outputs in folder scans: the current and the default suffix."""
    return tuple(s for s in {suffix, OUTPUT_SUFFIX} if s)


def _is_candidate(p: Path, skip: tuple[str, ...]) -> bool:
    return p.is_file() and p.suffix.lower() == ".pdf" and not p.stem.endswith(skip)


def collect_pdfs(paths: Iterable[str], recursive: bool = False,
                 suffix: str = OUTPUT_SUFFIX) -> tuple[list[str], list[str]]:
    """Expand files, folders and wildcard patterns into a de-duplicated list of PDFs.

    Folder scans and wildcards skip previous outputs (names ending in `suffix` or the default
    _translated); a file named explicitly is always kept. Returns (pdfs, entries that matched nothing).
    """
    skip = _skip_suffixes(suffix)
    pdfs: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        raw = raw.strip().strip('"')
        if not raw:
            continue
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found = sorted((q for q in it if _is_candidate(q, skip)), key=lambda q: str(q).lower())
        elif p.is_file():
            found = [p]
        elif any(c in raw for c in "*?["):
            found = sorted((Path(x) for x in glob.glob(raw, recursive=recursive) if _is_candidate(Path(x), skip)),
                           key=lambda q: str(q).lower())
        else:
            found = []
        if not found:
            missing.append(raw)
        for q in found:
            key = str(q.resolve()).lower()
            if key not in seen:
                seen.add(key)
                pdfs.append(str(q))
    return pdfs, missing


def output_name(input_pdf: str, suffix: str = OUTPUT_SUFFIX) -> str:
    return f"{Path(input_pdf).stem}{suffix}.pdf"


def plan_outputs(inputs: list[str], output_dir: str = "", suffix: str = OUTPUT_SUFFIX) -> list[str]:
    """Output path per input: <name><suffix>.pdf next to the input, or inside output_dir.
    Name clashes get a numeric ending; an input file is never chosen as an output."""
    used = {str(Path(p).resolve()).lower() for p in inputs}
    outs: list[str] = []
    for inp in inputs:
        folder = Path(output_dir.strip()) if output_dir.strip() else Path(inp).parent
        base = folder / output_name(inp, suffix)
        cand, n = base, 2
        while str(cand.resolve()).lower() in used:
            cand = base.with_name(f"{base.stem}_{n}{base.suffix}")
            n += 1
        used.add(str(cand.resolve()).lower())
        outs.append(str(cand))
    return outs


class Batch:
    def __init__(self, settings: Settings, items: list[BatchItem], emit: Callable[[dict], None],
                 relayout_only: bool = False, skip_existing: bool = False):
        self.settings = settings.copy()
        self.items = items
        self.emit = emit
        self.relayout_only = relayout_only
        self.skip_existing = skip_existing and not relayout_only
        self.thread: threading.Thread | None = None
        self.jobs: dict[int, Job] = {}       # running books by item index
        self.paused = False
        self._lock = threading.Lock()        # guards jobs / paused / progress; control calls run under it
        self._gate = threading.Event()       # cleared = paused: no new books are started
        self._gate.set()
        self._wake = threading.Event()       # wakes the scheduler (book finished, resume, stop)
        self._stopping = False
        self._progress: dict[int, dict] = {}  # last progress event per book
        self._totals = [0] * len(items)        # page count per book (before its job reports one)

    # ============ thread-facing API (same shape as Job) ============
    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def pause(self) -> None:
        self._set_paused()

    def resume(self, new_settings: Settings | None = None) -> None:
        with self._lock:
            if new_settings is not None:
                self.settings = new_settings.copy()    # also used for books started later
            self.paused = False
            self._gate.set()
            for job in self.jobs.values():
                self._call(job, "resume", new_settings)
            self.emit({"type": "state", "state": "running"})
        self._wake.set()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._gate.set()
            for job in self.jobs.values():
                self._call(job, "stop")
        self._wake.set()

    def render_now(self, layout_settings: Settings | None = None) -> None:
        with self._lock:
            for job in self.jobs.values():
                self._call(job, "render_now", layout_settings)

    def is_alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    # ============ internals ============
    @staticmethod
    def _call(job: Job, name: str, *args) -> None:
        try:
            getattr(job, name)(*args)
        except RuntimeError:        # its event loop just closed
            pass

    def _set_paused(self) -> None:
        """Pause the whole batch: by the user, or because one book hit a setting error
        (the same settings would fail in every other book too)."""
        with self._lock:
            if self.paused:
                return
            self.paused = True
            self._gate.clear()
            for job in self.jobs.values():
                self._call(job, "pause")
            self.emit({"type": "state", "state": "paused"})

    def _log(self, msg: str, level: str = "info") -> None:
        self.emit({"type": "log", "msg": time.strftime("%H:%M:%S ") + msg, "level": level})

    def _tag(self, i: int) -> str:
        name = Path(self.items[i].input_pdf).stem
        if len(name) > 24:
            name = name[:23] + "…"
        return f"[{i + 1}/{len(self.items)} {name}]"

    def _update(self, i: int) -> None:
        sts = [it.status for it in self.items]
        self.emit({"type": "batch", "index": i, "total": len(self.items), "item": asdict(self.items[i]),
                   "finished": sum(1 for s in sts if s not in ("pending", "running")),
                   "running": sts.count("running")})

    def _forward(self, i: int, ev: dict) -> None:
        t = ev["type"]
        if t == "state":
            # one book finishing must not end the batch; the batch emits its own states
            if ev["state"] == "paused":
                self._set_paused()
            return
        if t == "progress":
            with self._lock:
                self._progress[i] = ev
            self._emit_progress()
            return
        if t == "log":
            msg = ev["msg"]   # "HH:MM:SS text" -> "HH:MM:SS [i/n name] text"
            msg = (msg[:9] + self._tag(i) + " " + msg[9:]) if msg[2:3] == ":" and msg[8:9] == " " \
                else self._tag(i) + " " + msg
            ev = dict(ev, msg=msg)
        self.emit(ev)

    def _emit_progress(self) -> None:
        """Pages and token usage summed over all books of the batch."""
        with self._lock:
            ps = list(self._progress.items())
            totals = list(self._totals)
        for i, p in ps:
            totals[i] = p["total"]
        usage: dict[str, dict] = {}
        for _, p in ps:
            for role, u in p["usage"].items():
                acc = usage.setdefault(role, {})
                for k, v in u.items():
                    acc[k] = acc.get(k, 0) + v
        self.emit({"type": "progress", "done": sum(p["done"] for _, p in ps), "total": sum(totals),
                   "error": sum(p["error"] for _, p in ps), "usage": usage})

    def _count_pages(self) -> None:
        for i, it in enumerate(self.items):
            try:
                with pymupdf.open(it.input_pdf) as d:
                    self._totals[i] = len(parse_page_range(self.settings.page_range, d.page_count))
            except Exception:
                self._totals[i] = 0

    def _limit(self) -> int:
        try:
            return max(1, int(self.settings.file_concurrency))
        except (TypeError, ValueError):
            return 1

    def _run(self) -> None:
        n = len(self.items)
        s = self.settings
        kind = "重新排版（不呼叫 API）" if self.relayout_only else "翻譯"
        self.emit({"type": "state", "state": "running"})
        self._log(f"批次{kind}：共 {n} 個檔案，同時處理 {self._limit()} 本；目標語言 {s.target_lang}"
                  + ("" if self.relayout_only else f"；頁碼範圍 {s.page_range or '全部'}"))
        if not self.relayout_only:
            self._count_pages()
        pending = deque(range(n))
        workers: dict[int, threading.Thread] = {}
        try:
            while True:
                self._wake.clear()
                for i in [i for i, t in workers.items() if not t.is_alive()]:
                    del workers[i]
                if not workers and (self._stopping or not pending):
                    break
                if not self._stopping and self._gate.is_set():
                    while pending and len(workers) < self._limit():
                        i = pending.popleft()
                        item = self.items[i]
                        if self.skip_existing and Path(item.output_pdf).exists():
                            item.status, item.message = "skipped", "輸出檔已存在"
                            self._totals[i] = 0
                            self._log(f"{self._tag(i)} 略過（輸出檔已存在）")
                            self._update(i)
                            continue
                        item.status, item.message = "running", ""
                        workers[i] = threading.Thread(target=self._run_item, args=(i,), daemon=True)
                        workers[i].start()
                self._wake.wait(0.5)
        finally:
            cnt = {k: sum(1 for it in self.items if it.status == k) for k in STATUS_LABELS}
            parts = [f"{STATUS_LABELS[k]} {v}" for k, v in cnt.items() if v]
            bad = cnt["failed"] + cnt["partial"]
            self._log(f"批次結束：{'、'.join(parts)}", "warn" if bad or self._stopping else "info")
            if (bad or cnt["stopped"] or cnt["pending"]) and not self.relayout_only:
                self._log("未完成的檔案進度已保存，再次執行批次會從中斷處繼續。", "warn")
            self.emit({"type": "state", "state": "stopped" if self._stopping else "finished"})

    def _run_item(self, i: int) -> None:
        item = self.items[i]
        self._update(i)
        self._log(f"{self._tag(i)} ══════ 開始 ══════")
        try:
            Path(item.output_pdf).parent.mkdir(parents=True, exist_ok=True)
            if self.relayout_only:
                self._relayout(item)
            else:
                self._translate(i, item)
        except Exception as e:
            item.status, item.message = "failed", str(e)
        level = {"done": "info", "stopped": "warn", "partial": "warn"}.get(item.status, "error")
        self._log(f"{self._tag(i)} {STATUS_LABELS[item.status]}"
                  + (f"（{item.message}）" if item.message else ""), level)
        self._update(i)
        self._wake.set()

    def _translate(self, i: int, item: BatchItem) -> None:
        s = self.settings.copy()
        s.input_pdf, s.output_pdf = item.input_pdf, item.output_pdf
        job = Job(s, lambda ev: self._forward(i, ev))
        job.start()
        while not job.ready.wait(0.1):
            if not job.thread.is_alive():
                break
        with self._lock:
            self.jobs[i] = job
            # a pause / stop that arrived while the job was starting
            if self._stopping:
                self._call(job, "stop")
            elif self.paused:
                self._call(job, "pause")
        job.thread.join()
        with self._lock:
            self.jobs.pop(i, None)
        item.status = job.result["status"]
        item.message = job.result["message"]

    def _relayout(self, item: BatchItem) -> None:
        st = load_state(work_dir_for(item.input_pdf, self.settings.target_lang) / "state.json")
        if not st:
            item.status, item.message = "failed", "找不到此檔（此目標語言）的翻譯進度"
            return
        rules, stats = render_from_state(self.settings, st, item.output_pdf)
        self.emit({"type": "rules", "text": summary(rules)})
        self.emit({"type": "output", "path": item.output_pdf})
        item.status, item.message = "done", f"{item.output_pdf}（{layout_report(stats)}）"
