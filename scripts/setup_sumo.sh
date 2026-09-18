#!/usr/bin/env bash
set -e

echo "=== CityMind AI: Environment & SUMO Setup ==="

# Check OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[✓] macOS detected."
else
    echo "[!] Warning: This script is configured primarily for macOS Apple Silicon."
fi

# Detect SUMO_HOME if not explicitly exported
if [ -z "$SUMO_HOME" ]; then
    if [ -d "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo" ]; then
        export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
    elif [ -d "/opt/homebrew/opt/sumo/share/sumo" ]; then
        export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"
    elif [ -d "/opt/homebrew/share/sumo" ]; then
        export SUMO_HOME="/opt/homebrew/share/sumo"
    else
        echo "[!] SUMO_HOME not found. Attempting to install SUMO via Homebrew..."
        brew install sumo
        export SUMO_HOME="/opt/homebrew/share/sumo"
    fi
fi

echo "[✓] SUMO_HOME: $SUMO_HOME"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# Verify sumo CLI
if command -v sumo &> /dev/null; then
    SUMO_VER=$(sumo --version | head -n 1)
    echo "[✓] Found SUMO binary: $SUMO_VER"
else
    echo "[x] Error: 'sumo' command not found in PATH."
    exit 1
fi

# Clone InTAS scenario if missing
INTAS_DIR="data/raw/intas"
if [ ! -d "$INTAS_DIR" ] || [ -z "$(ls -A "$INTAS_DIR" 2>/dev/null)" ]; then
    echo "[*] Cloning InTAS scenario repository..."
    mkdir -p data/raw
    git clone https://github.com/silaslobo/InTAS.git "$INTAS_DIR"
    echo "[✓] InTAS cloned to $INTAS_DIR"
else
    echo "[✓] InTAS scenario already present at $INTAS_DIR"
fi

echo "=== Setup verification complete ==="
