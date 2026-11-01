#!/usr/bin/env sh
cd "$(dirname "$0")"
command -v uv >/dev/null 2>&1 || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
exec uv run pdf-translator
