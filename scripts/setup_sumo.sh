#!/usr/bin/env bash
# CityMind AI - SUMO check + pinned InTAS scenario setup.
#
#   ./scripts/setup_sumo.sh            # verify SUMO, fetch InTAS at the pinned commit, install training cfg
#   source scripts/setup_sumo.sh       # same, and also exports SUMO_HOME/PATH into YOUR shell
#
# Side effects are limited to: creating data/raw/intas (git clone), copying
# configs/sumo/InTAS_train.sumocfg into it, and writing a sourceable .env.sumo.
# It never installs software; it prints the install command instead.

# NOTE: deliberately no `set -e`, so sourcing does not alter the caller's shell options.
# Detect "sourced" so we never `exit` the caller's shell.
_sourced=0
(return 0 2>/dev/null) && _sourced=1
_fail() { echo "[x] $*" >&2; if [ "$_sourced" = 1 ]; then return 1; else exit 1; fi; }

_citymind_setup() {
    local repo_root script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"

    # Pinned upstream commit of https://github.com/silaslobo/InTAS (override with INTAS_COMMIT=...)
    local INTAS_REPO="${INTAS_REPO:-https://github.com/silaslobo/InTAS.git}"
    local INTAS_COMMIT="${INTAS_COMMIT:-0f7951ba01dda8483f0a852f2c3e4ff0d8a1c0ee}"
    local INTAS_DIR="$repo_root/data/raw/intas"

    echo "=== CityMind AI: Environment & SUMO Setup ==="
    echo "[*] OS: $(uname -s)"

    # --- locate SUMO -------------------------------------------------------
    if [ -z "${SUMO_HOME:-}" ]; then
        local c
        for c in \
            "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo" \
            "/opt/homebrew/opt/sumo/share/sumo" "/opt/homebrew/share/sumo" \
            "/usr/local/share/sumo" "/usr/share/sumo"; do
            if [ -d "$c" ]; then SUMO_HOME="$c"; break; fi
        done
    fi
    if [ -z "${SUMO_HOME:-}" ] && command -v python3 >/dev/null 2>&1; then
        # eclipse-sumo wheel (pip install eclipse-sumo) ships SUMO_HOME
        SUMO_HOME="$(python3 -c 'import sumo; print(sumo.SUMO_HOME)' 2>/dev/null || true)"
    fi
    if [ -z "${SUMO_HOME:-}" ] || [ ! -d "$SUMO_HOME" ]; then
        echo "[x] SUMO not found. Install one of:" >&2
        echo "      macOS : brew install sumo   (or the EclipseSUMO .pkg)" >&2
        echo "      Linux : sudo apt-get install sumo sumo-tools   (or: pip install eclipse-sumo)" >&2
        return 1
    fi
    export SUMO_HOME
    export PATH="$SUMO_HOME/bin:$PATH"
    case ":${PYTHONPATH:-}:" in *":$SUMO_HOME/tools:"*) ;; *) export PYTHONPATH="$SUMO_HOME/tools${PYTHONPATH:+:$PYTHONPATH}";; esac
    echo "[ok] SUMO_HOME: $SUMO_HOME"

    if command -v sumo >/dev/null 2>&1; then
        echo "[ok] sumo binary: $(sumo --version | head -n 1)"
    else
        echo "[!] 'sumo' not on PATH (SUMO_HOME/bin missing?). Python/TraCI may still work." >&2
    fi

    # --- InTAS at pinned commit (idempotent, non-destructive) --------------
    if [ ! -d "$INTAS_DIR/.git" ]; then
        if [ -e "$INTAS_DIR" ] && [ -n "$(ls -A "$INTAS_DIR" 2>/dev/null)" ]; then
            echo "[x] $INTAS_DIR exists but is not a git checkout; refusing to touch it." >&2
            echo "    Move it away and re-run." >&2
            return 1
        fi
        echo "[*] Cloning InTAS ..."
        mkdir -p "$repo_root/data/raw"
        git clone --quiet "$INTAS_REPO" "$INTAS_DIR" || return 1
    fi
    local head
    head="$(git -C "$INTAS_DIR" rev-parse HEAD)"
    if [ "$head" != "$INTAS_COMMIT" ]; then
        echo "[*] Checking out pinned InTAS commit $INTAS_COMMIT"
        git -C "$INTAS_DIR" fetch --quiet origin "$INTAS_COMMIT" 2>/dev/null || git -C "$INTAS_DIR" fetch --quiet origin
        git -C "$INTAS_DIR" checkout --quiet "$INTAS_COMMIT" \
            || { echo "[x] checkout failed (local modifications?). Not forcing." >&2; return 1; }
    fi
    echo "[ok] InTAS @ $(git -C "$INTAS_DIR" rev-parse --short HEAD)"
    [ -f "$INTAS_DIR/scenario/ingolstadt.net.xml" ] \
        || { echo "[x] scenario/ingolstadt.net.xml missing - incomplete clone; delete $INTAS_DIR and retry." >&2; return 1; }

    # --- tracked training sumocfg -> scenario dir ---------------------------
    # configs/sumo/InTAS_train.sumocfg is the canonical copy (relative paths resolve
    # because it lives next to ingolstadt.net.xml). src/ and scripts/ read it from here.
    cp "$repo_root/configs/sumo/InTAS_train.sumocfg" "$INTAS_DIR/scenario/InTAS_train.sumocfg" || return 1
    echo "[ok] installed scenario/InTAS_train.sumocfg"

    # --- sourceable env file -------------------------------------------------
    cat > "$repo_root/.env.sumo" <<EOF
export SUMO_HOME="$SUMO_HOME"
export PATH="\$SUMO_HOME/bin:\$PATH"
export PYTHONPATH="\$SUMO_HOME/tools\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
    echo "[ok] wrote .env.sumo  (use: source .env.sumo)"
    echo "=== Setup verification complete ==="
}

_citymind_setup "$@"
_rc=$?
unset -f _citymind_setup
[ "$_sourced" = 1 ] && return $_rc
exit $_rc
