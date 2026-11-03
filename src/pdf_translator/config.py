"""Settings model, persistence and font presets."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = APP_DIR / "config.json"
WORK_DIR = APP_DIR / "work"

ANTHROPIC_EFFORTS = ["default", "low", "medium", "high", "xhigh", "max"]
OPENAI_EFFORTS = ["default", "none", "minimal", "low", "medium", "high", "xhigh"]

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


@dataclass
class ModelConfig:
    provider: str = "anthropic"            # "anthropic" | "openai" (any OpenAI-compatible endpoint)
    model: str = "claude-opus-5-5"
    effort: str = "medium"                 # thinking effort; "default" = do not send
    max_tokens: int = 32000
    base_url: str = ""                     # OpenAI-compatible only; blank = official OpenAI
    api_key_env: str = ""                  # env var holding the key; blank = provider default
    json_mode: bool = True                 # OpenAI-compatible: send response_format=json_object
    refusal_fallback: bool = True          # Anthropic: server-side fallback on refusal

    def key_env(self) -> str:
        if self.api_key_env.strip():
            return self.api_key_env.strip()
        return "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"


def _default_checker() -> ModelConfig:
    return ModelConfig(model="claude-sonnet-5-5", effort="low", max_tokens=16000)


@dataclass
class Settings:
    input_pdf: str = ""
    output_pdf: str = ""
    page_range: str = ""                   # e.g. "1-3,7"; blank = all
    source_lang: str = "自動偵測"
    target_lang: str = "繁體中文（台灣）"
    extra_instructions: str = ""           # glossary / style notes for the translator

    translator: ModelConfig = field(default_factory=ModelConfig)
    checker: ModelConfig = field(default_factory=_default_checker)

    batch_max_chars: int = 2500            # characters per translation batch
    batch_max_segments: int = 40
    file_concurrency: int = 1              # batch: books (PDF files) processed at the same time
    page_concurrency: int = 3              # per book: pages processed at the same time
    request_concurrency: int = 6           # per book: simultaneous API requests (all roles)
    max_retries: int = 3

    enable_check: bool = True
    max_check_rounds: int = 2              # fix-and-recheck rounds per page
    check_with_reason: bool = True         # checker adds a short reason per rejected range

    font_preset: str = "Noto Sans CJK TC"
    font_regular: str = ""                 # custom font file (used when preset = 自訂)
    font_bold: str = ""
    min_font_scale: float = 0.6            # smallest allowed shrink when text overflows
    max_font_scale: float = 1.0            # >1 lets short translations grow to fill their box
    char_spacing: float = 0.0              # extra letter spacing in em (negative = tighter)
    length_hints: bool = True              # tell the translator the space limit of tight segments
    shorten_overflow: bool = True          # ask for a concise rewrite of segments that overflow
    inline_markup: bool = True             # carry inline styles through translation as Markdown markers

    # batch mode (UI 批次 tab)
    batch_files: list[str] = field(default_factory=list)
    batch_output_dir: str = ""             # blank = next to each input
    batch_suffix: str = "_translated"      # output name = <input name><suffix>.pdf
    batch_recursive: bool = False          # include sub-folders when adding a folder
    batch_skip_existing: bool = False      # skip files whose output already exists

    # ---- persistence ----
    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings":
        s = cls()
        if not path.exists():
            return s
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return s
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        s = cls()
        names = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in ("translator", "checker") and isinstance(v, dict):
                mc = ModelConfig()
                mnames = {f.name for f in fields(ModelConfig)}
                for mk, mv in v.items():
                    if mk in mnames:
                        setattr(mc, mk, mv)
                setattr(s, k, mc)
            elif k in names:
                setattr(s, k, v)
        return s

    def copy(self) -> "Settings":
        return Settings.from_dict(json.loads(json.dumps(asdict(self))))


# ---------------- fonts ----------------

def _font_dirs() -> list[Path]:
    dirs: list[Path] = []
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(windir) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]
    return dirs


def _find_font(*names: str) -> str:
    for d in _font_dirs():
        if not d.exists():
            continue
        for n in names:
            p = d / n
            if p.exists():
                return str(p)
            hits = list(d.rglob(n))
            if hits:
                return str(hits[0])
    return ""


# preset name -> (regular candidates, bold candidates)
FONT_PRESETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Noto Sans CJK TC": (("NotoSansCJKtc-Regular.otf", "NotoSansCJK-Regular.ttc"),
                         ("NotoSansCJKtc-Bold.otf", "NotoSansCJK-Bold.ttc")),
    "Noto Serif CJK TC": (("NotoSerifCJKtc-Regular.otf", "NotoSerifCJK-Regular.ttc"),
                          ("NotoSerifCJKtc-Bold.otf", "NotoSerifCJK-Bold.ttc")),
    "微軟正黑體": (("msjh.ttc",), ("msjhbd.ttc",)),
    "內建 (Droid Sans Fallback)": ((), ()),
    "自訂": ((), ()),
}


def resolve_fonts(s: Settings) -> tuple[str, str]:
    """Return (regular_path, bold_path). Empty string = use PyMuPDF built-in CJK font."""
    if s.font_preset == "自訂":
        reg = s.font_regular.strip()
        bold = s.font_bold.strip() or reg
        return (reg if reg and Path(reg).exists() else "",
                bold if bold and Path(bold).exists() else "")
    regs, bolds = FONT_PRESETS.get(s.font_preset, ((), ()))
    reg = _find_font(*regs) if regs else ""
    bold = (_find_font(*bolds) if bolds else "") or reg
    if not reg and s.font_preset != "內建 (Droid Sans Fallback)":
        # fall back to any known CJK font on the system
        reg = _find_font("NotoSansCJKtc-Regular.otf", "msjh.ttc", "NotoSansCJK-Regular.ttc")
        bold = _find_font("NotoSansCJKtc-Bold.otf", "msjhbd.ttc") or reg
    return reg, bold
