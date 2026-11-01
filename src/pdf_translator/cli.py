"""Headless runner: uv run python -m pdf_translator.cli input.pdf [--pages 1-3] [--no-check] [--relayout]"""
from __future__ import annotations

import argparse
import queue

from dotenv import load_dotenv

from .config import APP_DIR, CONFIG_PATH, Settings
from .layout_rules import summary
from .pipeline import Job, default_output_path, layout_report, load_state, render_from_state, work_dir_for


def main() -> None:
    load_dotenv(APP_DIR / ".env")
    ap = argparse.ArgumentParser(description="PDF translator (headless)")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--pages", default="")
    ap.add_argument("--target", default=None)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--relayout", action="store_true",
                    help="re-layout saved translations with the current font/spacing/scale settings (no API calls)")
    ap.add_argument("--font", default=None, help="font preset name (see config.FONT_PRESETS)")
    ap.add_argument("--spacing", type=float, default=None, help="letter spacing in em")
    ap.add_argument("--min-scale", type=float, default=None)
    ap.add_argument("--max-scale", type=float, default=None)
    args = ap.parse_args()

    s = Settings.load(CONFIG_PATH)
    s.input_pdf, s.output_pdf, s.page_range = args.input, args.output, args.pages
    if args.target:
        s.target_lang = args.target
    if args.no_check:
        s.enable_check = False
    for attr, val in (("font_preset", args.font), ("char_spacing", args.spacing),
                      ("min_font_scale", args.min_scale), ("max_font_scale", args.max_scale)):
        if val is not None:
            setattr(s, attr, val)

    if args.relayout:
        st = load_state(work_dir_for(s.input_pdf, s.target_lang) / "state.json")
        if not st:
            print("找不到此檔（此目標語言）的翻譯進度。")
            return
        out = s.output_pdf or default_output_path(s.input_pdf)
        rules, stats = render_from_state(s, st, out)
        print("版面規則:", summary(rules))
        print(f"已輸出：{out}（{layout_report(stats)}）")
        return

    q: queue.Queue = queue.Queue()
    job = Job(s, q.put)
    job.start()
    while True:
        ev = q.get()
        if ev["type"] == "log":
            print(ev["msg"], flush=True)
        elif ev["type"] == "progress":
            u = ev["usage"]
            print(f"  進度 {ev['done']}/{ev['total']}  tokens 翻譯 {u['translator']['input_tokens']}/"
                  f"{u['translator']['output_tokens']}  檢查 {u['checker']['input_tokens']}/"
                  f"{u['checker']['output_tokens']}", flush=True)
        elif ev["type"] == "rules":
            print("  版面規則:", ev["text"], flush=True)
        elif ev["type"] == "state" and ev["state"] in ("finished", "stopped", "error"):
            break
        elif ev["type"] == "state" and ev["state"] == "paused":
            print("工作因錯誤暫停，CLI 模式下結束。", flush=True)
            job.stop()


if __name__ == "__main__":
    main()
