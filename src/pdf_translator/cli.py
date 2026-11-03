"""Headless runner.

    uv run python -m pdf_translator.cli input.pdf [-o out.pdf] [--pages 1-3] [--no-check] [--relayout]
    uv run python -m pdf_translator.cli a.pdf b.pdf docs/ "scans/*.pdf" [--output-dir out/] [-r] [--skip-existing]
        [--suffix _zh] [-j 3] [--page-concurrency 3] [--request-concurrency 6]
"""
from __future__ import annotations

import argparse
import queue
import sys

from dotenv import load_dotenv

from .batch import Batch, BatchItem, check_suffix, collect_pdfs, plan_outputs
from .config import APP_DIR, CONFIG_PATH, Settings


def main() -> None:
    load_dotenv(APP_DIR / ".env")
    ap = argparse.ArgumentParser(description="PDF translator (headless)")
    ap.add_argument("inputs", nargs="+", help="PDF files, folders or wildcard patterns (batch when more than one PDF)")
    ap.add_argument("-o", "--output", default="", help="output file (single input only)")
    ap.add_argument("--output-dir", default="",
                    help="output folder for <name><suffix>.pdf (default: next to each input)")
    ap.add_argument("--suffix", default=None,
                    help="output name suffix in batch mode (default: config.json, normally _translated); "
                         "blank is allowed only with --output-dir")
    ap.add_argument("-r", "--recursive", action="store_true", help="include sub-folders of folder inputs")
    ap.add_argument("--skip-existing", action="store_true", help="skip files whose output already exists")
    ap.add_argument("-j", "--book-concurrency", type=int, default=None,
                    help="books (PDF files) translated at the same time")
    ap.add_argument("--page-concurrency", type=int, default=None, help="pages at the same time, per book")
    ap.add_argument("--request-concurrency", type=int, default=None,
                    help="simultaneous API requests, per book")
    ap.add_argument("--cache", choices=["5m", "1h", "off"], default=None,
                    help="Anthropic prompt caching TTL (default: config.json, normally 5m)")
    ap.add_argument("--api", choices=["realtime", "batch"], default=None,
                    help="Anthropic call mode: realtime streaming, or Message Batches at 50%% price (slower)")
    ap.add_argument("--pages", default="", help="page range applied to every file, e.g. 1-5,8")
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
    suffix = s.batch_suffix if args.suffix is None else args.suffix.strip()
    if not args.output and (err := check_suffix(suffix, args.output_dir)):
        ap.error(err)

    pdfs, missing = collect_pdfs(args.inputs, args.recursive, suffix)
    for m in missing:
        print(f"找不到 PDF：{m}", file=sys.stderr)
    if not pdfs:
        sys.exit("沒有可處理的 PDF。")
    if args.output and len(pdfs) > 1:
        ap.error("-o/--output 只能用於單一輸入檔；多個檔案請用 --output-dir")

    s.page_range = args.pages
    if args.target:
        s.target_lang = args.target
    if args.no_check:
        s.enable_check = False
    for attr, val in (("font_preset", args.font), ("char_spacing", args.spacing),
                      ("min_font_scale", args.min_scale), ("max_font_scale", args.max_scale),
                      ("file_concurrency", args.book_concurrency), ("page_concurrency", args.page_concurrency),
                      ("request_concurrency", args.request_concurrency), ("prompt_cache", args.cache),
                      ("api_mode", args.api)):
        if val is not None:
            setattr(s, attr, val)

    outs = [args.output] if args.output else plan_outputs(pdfs, args.output_dir, suffix)
    items = [BatchItem(i, o) for i, o in zip(pdfs, outs)]

    q: queue.Queue = queue.Queue()
    batch = Batch(s, items, q.put, relayout_only=args.relayout, skip_existing=args.skip_existing)
    batch.start()
    while True:
        try:
            ev = q.get(timeout=0.5)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("中斷：停止中，進度會保存…", flush=True)
            batch.stop()
            continue
        if ev["type"] == "log":
            print(ev["msg"], flush=True)
        elif ev["type"] == "progress":
            u = ev["usage"]
            tr, ck = u["translator"], u["checker"]
            print(f"  進度 {ev['done']}/{ev['total']}  tokens 翻譯 輸入 {tr['input_tokens']}"
                  f"（快取讀 {tr['cache_read_tokens']}）/輸出 {tr['output_tokens']}"
                  f"  檢查 輸入 {ck['input_tokens']}（快取讀 {ck['cache_read_tokens']}）/輸出 {ck['output_tokens']}",
                  flush=True)
        elif ev["type"] == "rules":
            print("  版面規則:", ev["text"], flush=True)
        elif ev["type"] == "state" and ev["state"] in ("finished", "stopped"):
            break
        elif ev["type"] == "state" and ev["state"] == "paused":
            # only setting errors pause in CLI mode; they would fail every remaining file too
            print("工作因設定錯誤暫停，CLI 模式下停止批次。", flush=True)
            batch.stop()
    if batch.thread:
        batch.thread.join()
    sys.exit(0 if all(it.status in ("done", "skipped") for it in items) else 1)


if __name__ == "__main__":
    main()
