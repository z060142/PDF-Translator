"""Tkinter UI for the PDF translator."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from dotenv import load_dotenv

from .config import (ANTHROPIC_EFFORTS, APP_DIR, CONFIG_PATH, FONT_PRESETS, OPENAI_EFFORTS, Settings,
                     resolve_fonts)
from .layout_rules import summary
from .pipeline import Job, default_output_path, layout_report, load_state, render_from_state, work_dir_for

ANTHROPIC_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1", "claude-opus-5"]
OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "deepseek-chat", "qwen-plus"]
SOURCE_LANGS = ["自動偵測", "English", "한국어", "日本語", "简体中文", "Deutsch", "Français", "Español"]
TARGET_LANGS = ["繁體中文（台灣）", "简体中文", "English", "日本語", "한국어"]


class ModelFrame(ttk.Frame):
    """Settings for one model role (translator / checker)."""

    def __init__(self, master, title: str):
        super().__init__(master, padding=10)
        self.v = {
            "provider": tk.StringVar(), "model": tk.StringVar(), "effort": tk.StringVar(),
            "max_tokens": tk.IntVar(), "base_url": tk.StringVar(), "api_key_env": tk.StringVar(),
            "json_mode": tk.BooleanVar(), "refusal_fallback": tk.BooleanVar(),
        }
        ttk.Label(self, text=title, font=("", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        r = 1
        ttk.Label(self, text="供應商").grid(row=r, column=0, sticky="w")
        self.cb_provider = ttk.Combobox(self, textvariable=self.v["provider"], values=["anthropic", "openai"],
                                        state="readonly", width=18)
        self.cb_provider.grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(self, text="openai = 任何 OpenAI 相容 API（設定 Base URL）", foreground="#666").grid(
            row=r, column=2, sticky="w", padx=8)
        r += 1
        ttk.Label(self, text="模型").grid(row=r, column=0, sticky="w")
        self.cb_model = ttk.Combobox(self, textvariable=self.v["model"], width=36)
        self.cb_model.grid(row=r, column=1, columnspan=2, sticky="w", pady=2)
        r += 1
        ttk.Label(self, text="思考努力程度").grid(row=r, column=0, sticky="w")
        self.cb_effort = ttk.Combobox(self, textvariable=self.v["effort"], width=18)
        self.cb_effort.grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(self, text="default = 不傳此參數（用模型預設）", foreground="#666").grid(
            row=r, column=2, sticky="w", padx=8)
        r += 1
        ttk.Label(self, text="max_tokens").grid(row=r, column=0, sticky="w")
        self.sp_max = ttk.Spinbox(self, from_=1024, to=128000, increment=1024, textvariable=self.v["max_tokens"], width=10)
        self.sp_max.grid(row=r, column=1, sticky="w", pady=2)
        r += 1
        ttk.Label(self, text="Base URL").grid(row=r, column=0, sticky="w")
        self.e_base = ttk.Entry(self, textvariable=self.v["base_url"], width=50)
        self.e_base.grid(row=r, column=1, columnspan=2, sticky="w", pady=2)
        r += 1
        ttk.Label(self, text="API Key 環境變數").grid(row=r, column=0, sticky="w")
        self.e_key = ttk.Entry(self, textvariable=self.v["api_key_env"], width=24)
        self.e_key.grid(row=r, column=1, sticky="w", pady=2)
        self.lbl_key = ttk.Label(self, text="")
        self.lbl_key.grid(row=r, column=2, sticky="w", padx=8)
        r += 1
        self.ck_json = ttk.Checkbutton(self, text="JSON mode（response_format=json_object，供應商不支援時關閉）",
                                       variable=self.v["json_mode"])
        self.ck_json.grid(row=r, column=0, columnspan=3, sticky="w", pady=2)
        r += 1
        self.ck_fb = ttk.Checkbutton(self, text="Claude 拒絕回應時自動改用備援模型（server-side fallback）",
                                     variable=self.v["refusal_fallback"])
        self.ck_fb.grid(row=r, column=0, columnspan=3, sticky="w", pady=2)
        self.columnconfigure(2, weight=1)

        self.cb_provider.bind("<<ComboboxSelected>>", lambda e: self._on_provider(change_model=True))
        self.v["api_key_env"].trace_add("write", lambda *a: self._update_key_label())
        self.widgets = [self.cb_provider, self.cb_model, self.cb_effort, self.sp_max, self.e_base, self.e_key,
                        self.ck_json, self.ck_fb]

    def _on_provider(self, change_model: bool = False) -> None:
        p = self.v["provider"].get()
        if p == "anthropic":
            self.cb_model["values"] = ANTHROPIC_MODELS
            self.cb_effort["values"] = ANTHROPIC_EFFORTS
            if change_model and not self.v["model"].get().startswith("claude"):
                self.v["model"].set(ANTHROPIC_MODELS[0])
            if self.v["effort"].get() not in ANTHROPIC_EFFORTS:
                self.v["effort"].set("medium")
        else:
            self.cb_model["values"] = OPENAI_MODELS
            self.cb_effort["values"] = OPENAI_EFFORTS
            if change_model and self.v["model"].get().startswith("claude"):
                self.v["model"].set(OPENAI_MODELS[0])
            if self.v["effort"].get() not in OPENAI_EFFORTS:
                self.v["effort"].set("default")
        self._update_key_label()

    def _update_key_label(self) -> None:
        env = self.v["api_key_env"].get().strip() or (
            "ANTHROPIC_API_KEY" if self.v["provider"].get() == "anthropic" else "OPENAI_API_KEY")
        ok = bool(os.environ.get(env))
        self.lbl_key.config(text=f"{env}: {'已設定' if ok else '未設定'}", foreground="#2a7" if ok else "#c33")

    def load(self, mc) -> None:
        for k, var in self.v.items():
            var.set(getattr(mc, k))
        self._on_provider()

    def dump(self, mc) -> None:
        for k, var in self.v.items():
            try:
                setattr(mc, k, var.get())
            except tk.TclError:
                pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.job: Job | None = None
        self.events: queue.Queue = queue.Queue()
        self.settings = Settings.load(CONFIG_PATH)
        root.title("PDF 翻譯器")
        root.geometry("1000x820")
        root.minsize(860, 640)
        self._build()
        self._load_settings(self.settings)
        self._set_mode("idle")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.v = {
            "input_pdf": tk.StringVar(), "output_pdf": tk.StringVar(), "page_range": tk.StringVar(),
            "source_lang": tk.StringVar(), "target_lang": tk.StringVar(),
            "batch_max_chars": tk.IntVar(), "batch_max_segments": tk.IntVar(),
            "page_concurrency": tk.IntVar(), "request_concurrency": tk.IntVar(), "max_retries": tk.IntVar(),
            "enable_check": tk.BooleanVar(), "max_check_rounds": tk.IntVar(), "check_with_reason": tk.BooleanVar(),
            "font_preset": tk.StringVar(), "font_regular": tk.StringVar(), "font_bold": tk.StringVar(),
            "min_font_scale": tk.DoubleVar(), "max_font_scale": tk.DoubleVar(), "char_spacing": tk.DoubleVar(),
            "length_hints": tk.BooleanVar(), "shorten_overflow": tk.BooleanVar(),
        }
        nb = ttk.Notebook(self.root)
        nb.pack(fill="x", padx=8, pady=(8, 4))
        self.doc_widgets: list = []      # locked during a whole job (also while paused)
        self.setting_widgets: list = []  # editable while paused

        # --- general tab ---
        g = ttk.Frame(nb, padding=10)
        nb.add(g, text="一般")
        r = 0
        ttk.Label(g, text="輸入 PDF").grid(row=r, column=0, sticky="w")
        e = ttk.Entry(g, textvariable=self.v["input_pdf"], width=70)
        e.grid(row=r, column=1, sticky="we", pady=2)
        b = ttk.Button(g, text="瀏覽…", command=self._browse_input)
        b.grid(row=r, column=2, padx=4)
        self.doc_widgets += [e, b]
        r += 1
        ttk.Label(g, text="輸出 PDF").grid(row=r, column=0, sticky="w")
        e = ttk.Entry(g, textvariable=self.v["output_pdf"], width=70)
        e.grid(row=r, column=1, sticky="we", pady=2)
        b = ttk.Button(g, text="瀏覽…", command=self._browse_output)
        b.grid(row=r, column=2, padx=4)
        self.doc_widgets += [e, b]
        r += 1
        ttk.Label(g, text="頁碼範圍").grid(row=r, column=0, sticky="w")
        e = ttk.Entry(g, textvariable=self.v["page_range"], width=20)
        e.grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(g, text="例：1-5,8；空白 = 全部", foreground="#666").grid(row=r, column=1, sticky="w", padx=(170, 0))
        self.doc_widgets.append(e)
        r += 1
        ttk.Label(g, text="來源語言").grid(row=r, column=0, sticky="w")
        cb = ttk.Combobox(g, textvariable=self.v["source_lang"], values=SOURCE_LANGS, width=20)
        cb.grid(row=r, column=1, sticky="w", pady=2)
        self.setting_widgets.append(cb)
        r += 1
        ttk.Label(g, text="目標語言").grid(row=r, column=0, sticky="w")
        cb = ttk.Combobox(g, textvariable=self.v["target_lang"], values=TARGET_LANGS, width=20)
        cb.grid(row=r, column=1, sticky="w", pady=2)
        self.doc_widgets.append(cb)
        r += 1
        ttk.Label(g, text="額外指示 / 詞彙表").grid(row=r, column=0, sticky="nw")
        self.txt_extra = tk.Text(g, height=4, width=70, wrap="word")
        self.txt_extra.grid(row=r, column=1, columnspan=2, sticky="we", pady=2)
        self.setting_widgets.append(self.txt_extra)
        g.columnconfigure(1, weight=1)

        # --- model tabs ---
        self.mf_trans = ModelFrame(nb, "翻譯模型")
        nb.add(self.mf_trans, text="翻譯模型")
        self.mf_check = ModelFrame(nb, "檢查模型（獨立模型，模糊比對全頁原文與譯文）")
        nb.add(self.mf_check, text="檢查模型")
        self.setting_widgets += self.mf_trans.widgets + self.mf_check.widgets

        # --- workflow tab ---
        w = ttk.Frame(nb, padding=10)
        nb.add(w, text="流程 / 併發")
        rows = [
            ("同時處理頁數", "page_concurrency", 1, 64, 1),
            ("最大並發請求數（全部角色合計）", "request_concurrency", 1, 128, 1),
            ("每批最大字元數", "batch_max_chars", 200, 20000, 100),
            ("每批最多段落數", "batch_max_segments", 1, 500, 1),
            ("單一請求最多重試次數", "max_retries", 1, 20, 1),
            ("最多修復輪數（否決→修復→重檢）", "max_check_rounds", 0, 10, 1),
        ]
        for i, (label, key, lo, hi, inc) in enumerate(rows):
            ttk.Label(w, text=label).grid(row=i, column=0, sticky="w", pady=2)
            sp = ttk.Spinbox(w, from_=lo, to=hi, increment=inc, textvariable=self.v[key], width=10)
            sp.grid(row=i, column=1, sticky="w", padx=8)
            self.setting_widgets.append(sp)
        i = len(rows)
        ck = ttk.Checkbutton(w, text="啟用多步驟翻譯檢查（每頁翻完後由檢查模型判定，否決段落退回翻譯模型修復）",
                             variable=self.v["enable_check"])
        ck.grid(row=i, column=0, columnspan=2, sticky="w", pady=(8, 2))
        ck2 = ttk.Checkbutton(w, text="檢查結果附簡短原因（關閉則只輸出 是/否 + 否決區間）",
                              variable=self.v["check_with_reason"])
        ck2.grid(row=i + 1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(w, text="暫停後可修改這裡與模型設定，按「繼續」立即套用。",
                  foreground="#666").grid(row=i + 2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.setting_widgets += [ck, ck2]

        # --- layout tab ---
        t = ttk.Frame(nb, padding=10)
        nb.add(t, text="排版 / 字型")
        ttk.Label(t, text="字型（全部譯文統一使用）").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(t, textvariable=self.v["font_preset"], values=list(FONT_PRESETS), state="readonly", width=28)
        cb.grid(row=0, column=1, sticky="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._update_font_label())
        self.setting_widgets.append(cb)
        for i, (label, key) in enumerate((("自訂常規字型檔", "font_regular"), ("自訂粗體字型檔", "font_bold")), 1):
            ttk.Label(t, text=label).grid(row=i, column=0, sticky="w")
            e = ttk.Entry(t, textvariable=self.v[key], width=60)
            e.grid(row=i, column=1, sticky="we", pady=2)
            b = ttk.Button(t, text="瀏覽…", command=lambda k=key: self._browse_font(k))
            b.grid(row=i, column=2, padx=4)
            self.setting_widgets += [e, b]
        self.lbl_font = ttk.Label(t, text="", foreground="#666", wraplength=700, justify="left")
        self.lbl_font.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 8))
        ttk.Label(t, text="最小字級縮放（放不下時最多縮到原字級的比例）").grid(row=4, column=0, sticky="w")
        sp = ttk.Spinbox(t, from_=0.3, to=1.0, increment=0.05, textvariable=self.v["min_font_scale"], width=8)
        sp.grid(row=4, column=1, sticky="w")
        self.setting_widgets.append(sp)
        ttk.Label(t, text="最大字級縮放（>1 允許較短的譯文放大填滿原框）").grid(row=5, column=0, sticky="w")
        sp = ttk.Spinbox(t, from_=1.0, to=1.5, increment=0.05, textvariable=self.v["max_font_scale"], width=8)
        sp.grid(row=5, column=1, sticky="w")
        self.setting_widgets.append(sp)
        ttk.Label(t, text="字距（em，負值＝緊縮，例如 -0.05 / 0.05）").grid(row=6, column=0, sticky="w")
        sp = ttk.Spinbox(t, from_=-0.2, to=0.5, increment=0.01, textvariable=self.v["char_spacing"], width=8)
        sp.grid(row=6, column=1, sticky="w")
        self.setting_widgets.append(sp)
        ck1 = ttk.Checkbutton(t, text="動態學習版面規則：依文件累積的譯文/原文長度比，提示翻譯模型版面容量",
                              variable=self.v["length_hints"])
        ck1.grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ck2 = ttk.Checkbutton(t, text="譯文超出原版面時，先要求翻譯模型精簡（每段最多一次，在檢查之前）",
                              variable=self.v["shorten_overflow"])
        ck2.grid(row=8, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(t, text="同一樣式（原字級＋粗細）的文字在全文件使用同一縮放比例，由已翻譯的頁面自動學習；"
                          "輸出時以全文件規則重新排版，各頁字級一致。\n"
                          "翻譯完成後若只想改字型、字距或縮放：修改本頁設定後按「重新排版輸出」，"
                          "會用已保存的譯文重新學習規則並輸出，不呼叫 API。暫停中也可以先按它預覽。",
                  foreground="#666", wraplength=760, justify="left").grid(row=9, column=0, columnspan=3,
                                                                          sticky="w", pady=(6, 0))
        self.setting_widgets += [ck1, ck2]
        t.columnconfigure(1, weight=1)

        # --- controls ---
        c = ttk.Frame(self.root, padding=(8, 4))
        c.pack(fill="x")
        self.btn_start = ttk.Button(c, text="開始 / 恢復", command=self._start)
        self.btn_pause = ttk.Button(c, text="暫停", command=self._pause)
        self.btn_resume = ttk.Button(c, text="繼續（套用設定）", command=self._resume)
        self.btn_stop = ttk.Button(c, text="停止", command=self._stop)
        self.btn_render = ttk.Button(c, text="重新排版輸出（不翻譯）", command=self._render_now)
        self.btn_clear = ttk.Button(c, text="清除此檔進度", command=self._clear_progress)
        self.btn_open = ttk.Button(c, text="開啟輸出資料夾", command=self._open_output_dir)
        for b in (self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop, self.btn_render,
                  self.btn_clear, self.btn_open):
            b.pack(side="left", padx=3)

        s = ttk.Frame(self.root, padding=(8, 2))
        s.pack(fill="x")
        self.progress = ttk.Progressbar(s, maximum=1, value=0)
        self.progress.pack(fill="x")
        self.lbl_status = ttk.Label(s, text="就緒")
        self.lbl_status.pack(anchor="w", pady=(2, 0))
        self.lbl_usage = ttk.Label(s, text="", foreground="#555")
        self.lbl_usage.pack(anchor="w")
        self.lbl_rules = ttk.Label(s, text="", foreground="#555", wraplength=960, justify="left")
        self.lbl_rules.pack(anchor="w")

        self.log = ScrolledText(self.root, height=14, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.log.tag_config("warn", foreground="#b8860b")
        self.log.tag_config("error", foreground="#c62828")

    # ------------------------------------------------------------------ settings <-> widgets
    def _load_settings(self, s: Settings) -> None:
        for k, var in self.v.items():
            var.set(getattr(s, k))
        self.txt_extra.delete("1.0", "end")
        self.txt_extra.insert("1.0", s.extra_instructions)
        self.mf_trans.load(s.translator)
        self.mf_check.load(s.checker)
        self._update_font_label()

    def _collect(self) -> Settings:
        s = self.settings.copy()
        for k, var in self.v.items():
            try:
                setattr(s, k, var.get())
            except tk.TclError:
                pass
        s.extra_instructions = self.txt_extra.get("1.0", "end").strip()
        self.mf_trans.dump(s.translator)
        self.mf_check.dump(s.checker)
        self.settings = s
        try:
            s.save(CONFIG_PATH)
        except OSError:
            pass
        return s

    def _update_font_label(self) -> None:
        s = self._collect()
        reg, bold = resolve_fonts(s)
        self.lbl_font.config(text=f"常規：{reg or '內建 Droid Sans Fallback'}\n粗體：{bold or '（以描邊模擬粗體）'}")

    def _set_mode(self, mode: str) -> None:
        """mode: idle | running | paused"""
        self.mode = mode

        def enable(w, on: bool):
            if isinstance(w, tk.Text):
                w.config(state="normal" if on else "disabled")
            else:
                w.state(["!disabled"] if on else ["disabled"])

        for w in self.doc_widgets:
            enable(w, mode == "idle")
        for w in self.setting_widgets:
            enable(w, mode != "running")
        self.btn_start.state(["!disabled"] if mode == "idle" else ["disabled"])
        self.btn_clear.state(["!disabled"] if mode == "idle" else ["disabled"])
        self.btn_pause.state(["!disabled"] if mode == "running" else ["disabled"])
        self.btn_resume.state(["!disabled"] if mode == "paused" else ["disabled"])
        self.btn_stop.state(["!disabled"] if mode in ("running", "paused") else ["disabled"])

    # ------------------------------------------------------------------ actions
    def _browse_input(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if p:
            self.v["input_pdf"].set(p)
            self.v["output_pdf"].set(default_output_path(p))

    def _browse_output(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if p:
            self.v["output_pdf"].set(p)

    def _browse_font(self, key: str) -> None:
        p = filedialog.askopenfilename(filetypes=[("Font", "*.ttf *.otf *.ttc"), ("All", "*.*")])
        if p:
            self.v[key].set(p)
            self.v["font_preset"].set("自訂")
            self._update_font_label()

    def _start(self) -> None:
        s = self._collect()
        if not s.input_pdf or not Path(s.input_pdf).exists():
            messagebox.showerror("錯誤", "請選擇存在的輸入 PDF")
            return
        if not s.output_pdf.strip():
            s.output_pdf = default_output_path(s.input_pdf)
            self.v["output_pdf"].set(s.output_pdf)
        self._append_log(f"開始：{s.input_pdf}")
        self.job = Job(s, self.events.put)
        self.job.start()
        self._set_mode("running")

    def _pause(self) -> None:
        if self.job:
            self.job.pause()

    def _resume(self) -> None:
        if self.job:
            self.job.resume(self._collect())

    def _stop(self) -> None:
        if self.job:
            self.job.stop()

    def _render_now(self) -> None:
        s = self._collect()
        if self.job and self.job.is_alive():
            self.job.render_now(s)
            return
        if not s.input_pdf or not Path(s.input_pdf).exists():
            return
        st = load_state(work_dir_for(s.input_pdf, s.target_lang) / "state.json")
        if not st:
            messagebox.showinfo("提示", "此檔案（此目標語言）沒有已保存的進度")
            return
        out = s.output_pdf.strip() or default_output_path(s.input_pdf)

        self._append_log("重新排版中（使用已保存的譯文，不呼叫 API）…")

        def work():
            try:
                rules, stats = render_from_state(s, st, out)
                self.events.put({"type": "rules", "text": summary(rules)})
                self.events.put({"type": "log", "msg": f"已輸出：{out}（{layout_report(stats)}）", "level": "info"})
                self.events.put({"type": "output", "path": out})
            except Exception as e:
                self.events.put({"type": "log", "msg": f"輸出失敗：{e}", "level": "error"})

        threading.Thread(target=work, daemon=True).start()

    def _clear_progress(self) -> None:
        s = self._collect()
        if not s.input_pdf or not Path(s.input_pdf).exists():
            return
        wd = work_dir_for(s.input_pdf, s.target_lang)
        if not wd.exists():
            messagebox.showinfo("提示", "沒有可清除的進度")
            return
        if messagebox.askyesno("確認", f"刪除此檔（目標語言：{s.target_lang}）的所有翻譯進度？\n{wd}"):
            shutil.rmtree(wd, ignore_errors=True)
            self._append_log("已清除進度。")

    def _open_output_dir(self) -> None:
        out = self.v["output_pdf"].get().strip()
        d = Path(out).parent if out else APP_DIR
        if not d.exists():
            return
        if sys.platform.startswith("win"):
            os.startfile(d)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(d)])

    # ------------------------------------------------------------------ events
    def _append_log(self, msg: str, level: str = "info") -> None:
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n", (level,) if level in ("warn", "error") else ())
        self.log.see("end")
        self.log.config(state="disabled")

    def _poll(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                t = ev["type"]
                if t == "log":
                    self._append_log(ev["msg"], ev.get("level", "info"))
                elif t == "progress":
                    self.progress.config(maximum=max(1, ev["total"]), value=ev["done"])
                    u = ev["usage"]
                    tr, ck = u["translator"], u["checker"]
                    self.lbl_usage.config(text=(
                        f"翻譯：{tr['requests']} 次請求，輸入 {tr['input_tokens']:,}（快取 {tr['cache_read_tokens']:,}）"
                        f" / 輸出 {tr['output_tokens']:,} tokens    "
                        f"檢查：{ck['requests']} 次請求，輸入 {ck['input_tokens']:,} / 輸出 {ck['output_tokens']:,} tokens"))
                    self._progress_text = f"頁面 {ev['done']}/{ev['total']}" + (
                        f"，失敗 {ev['error']}" if ev["error"] else "")
                    self._refresh_status()
                elif t == "rules":
                    self.lbl_rules.config(text="版面規則（學習中）：" + ev["text"])
                elif t == "state":
                    st = ev["state"]
                    if st == "running":
                        self._set_mode("running")
                    elif st == "paused":
                        self._set_mode("paused")
                    elif st in ("finished", "stopped", "error"):
                        self._set_mode("idle")
                    self._state_text = {"running": "執行中", "paused": "已暫停（可修改設定）",
                                        "finished": "完成", "stopped": "已停止（進度已保存）",
                                        "error": "錯誤"}.get(st, st)
                    self._refresh_status()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _refresh_status(self) -> None:
        parts = [getattr(self, "_state_text", ""), getattr(self, "_progress_text", "")]
        self.lbl_status.config(text="　".join(p for p in parts if p))

    def _on_close(self) -> None:
        if self.job and self.job.is_alive():
            if not messagebox.askyesno("確認", "翻譯進行中，停止並離開？（進度會保存）"):
                return
            self.job.stop()
            self.job.thread.join(timeout=5)
        self._collect()
        self.root.destroy()


def main() -> None:
    load_dotenv(APP_DIR / ".env")
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if sys.platform.startswith("win") else "clam")
    except tk.TclError:
        pass
    if sys.platform.startswith("win"):
        root.option_add("*Font", ("Microsoft JhengHei UI", 10))
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
