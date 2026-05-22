#!/usr/bin/env bash
# Clean-restart the PPP loop after a multi-driver mess.
#
# Steps:
#   1. Kill every PPP-related process (drivers, xargs, workers, pdp3).
#   2. Delete any per-day output dir without a kin_* file >=50 kB
#      (those are half-done or were corrupted by concurrent writers).
#   3. Launch ONE fresh `run_ppp.sh --jobs N --ppp-only` under nohup.
#
# Usage:
#   ./clean_restart_ppp.sh                # default --jobs 8
#   ./clean_restart_ppp.sh 12             # custom jobs
#   ./clean_restart_ppp.sh --dry-run      # print what would happen, do nothing
#   ./clean_restart_ppp.sh 8 --dry-run    # both
#
# Safe to re-run. Idempotent on already-clean state.

set -uo pipefail

JOBS=8
DRY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=true ;;
    [0-9]*)    JOBS="$arg" ;;
    *)         echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CODE_DIR}/.." && pwd)"
PPP_DIR="${ROOT_DIR}/work/ppp"
LOG="/tmp/ppp_full_run.log"

stamp() { echo "[$(date '+%H:%M:%S')] $*"; }

# ----------------------------------------------------------------
# 1) Stop everything
# ----------------------------------------------------------------
stamp "[1/4] Killing driver + workers (xargs, pride_runner, pdp3)..."
PATTERNS=(
  "run_ppp.sh"
  "xargs"
  "python -m ppp.pride_runner"
  "pdp3"
  "PRIDE_PPPAR_BIN"
)
if $DRY; then
  stamp "  (dry-run) currently alive:"
  pgrep -af "xargs|pride_runner|pdp3|PRIDE_PPPAR|run_ppp.sh" || stamp "    (none)"
else
  for p in "${PATTERNS[@]}"; do
    pkill -9 -f "$p" 2>/dev/null || true
  done
  sleep 3
  remaining="$(pgrep -af 'xargs|pride_runner|pdp3|PRIDE_PPPAR|run_ppp.sh' || true)"
  if [[ -n "$remaining" ]]; then
    stamp "  ERROR: processes still alive after kill:"
    echo "$remaining"
    exit 1
  fi
  stamp "  OK, all gone."
fi

# ----------------------------------------------------------------
# 2) Find half-done day directories
# ----------------------------------------------------------------
stamp "[2/4] Scanning ${PPP_DIR} for incomplete day directories..."
incomplete=()
if [[ -d "$PPP_DIR" ]]; then
  for d in "$PPP_DIR"/*/; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    # Only consider YYYYMMDD-named directories. Skip PRIDE's shared
    # `products/` IGS-cache and any other non-day folders.
    [[ "$name" =~ ^[0-9]{8}$ ]] || continue
    if ! find "$d" -name 'kin_*' -size +50k -print -quit 2>/dev/null | grep -q .; then
      incomplete+=("$name")
    fi
  done
fi

stamp "  Found ${#incomplete[@]} incomplete day directories."
if (( ${#incomplete[@]} > 0 )); then
  printf '    - %s\n' "${incomplete[@]}"
fi

# ----------------------------------------------------------------
# 3) Delete incomplete dirs
# ----------------------------------------------------------------
if (( ${#incomplete[@]} > 0 )); then
  if $DRY; then
    stamp "[3/4] (dry-run) would delete ${#incomplete[@]} directories"
  else
    stamp "[3/4] Deleting ${#incomplete[@]} incomplete directories..."
    for d in "${incomplete[@]}"; do
      rm -rf "${PPP_DIR}/${d}"
    done
    stamp "  Done."
  fi
else
  stamp "[3/4] Nothing to delete."
fi

# ----------------------------------------------------------------
# 4) Restart fresh driver
# ----------------------------------------------------------------
n_kin=$(find "$PPP_DIR" -name 'kin_*' -size +50k 2>/dev/null | wc -l)
n_dir=$(find "$PPP_DIR" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]{8}$' 2>/dev/null | wc -l)
stamp "[4/4] State after cleanup: ${n_kin} good kin_* files in ${n_dir} day directories (PRIDE products/ cache preserved)."

if $DRY; then
  stamp "  (dry-run) would launch: ./run_ppp.sh --jobs ${JOBS} --ppp-only"
  exit 0
fi

stamp "  Launching fresh driver: ./run_ppp.sh --jobs ${JOBS} --ppp-only"
cd "$CODE_DIR"
nohup ./run_ppp.sh --jobs "$JOBS" --ppp-only > "$LOG" 2>&1 &
sleep 5
disown 2>/dev/null || true

echo
stamp "Driver process:"
pgrep -af run_ppp.sh || stamp "  WARNING: no run_ppp.sh process found"
echo
stamp "Log head:"
tail -n 10 "$LOG"
echo
stamp "Monitor with:"
stamp "  tail -f ${LOG}"
stamp "  pgrep -cf 'python -m ppp.pride_runner'   # should be ${JOBS} when steady"
