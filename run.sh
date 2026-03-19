#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FONT_DIR="$SCRIPT_DIR/fonts"
FONT_FILE="$FONT_DIR/NotoSansSC-Regular.otf"
FONT_URL="https://fonts.gstatic.com/ea/notosanssc/v1/NotoSansSC-Regular.otf"
PYTHON_BIN="$SCRIPT_DIR/env/bin/python"

has_system_zh_font() {
    if ! command -v fc-list >/dev/null 2>&1; then
        return 1
    fi
    test -n "$(fc-list :lang=zh file 2>/dev/null | head -n 1)"
}

if [ ! -f "$FONT_FILE" ] && ! has_system_zh_font; then
    mkdir -p "$FONT_DIR"
    echo "未检测到系统中文字体，正在下载项目内字体..."
    curl -L --fail --retry 3 "$FONT_URL" -o "$FONT_FILE"
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"

cd "$SCRIPT_DIR"
if [ -x "$PYTHON_BIN" ]; then
    exec "$PYTHON_BIN" UI.py
fi

exec python UI.py
