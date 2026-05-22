#!/usr/bin/env bash
# Full RINEX + PPP reprocess of the 216-day cruise, post-bugfix.
#
# Phases:
#   1. Sanity-check that the three fixes are in place:
#      - convbin_runner.py uses TRSAX4E02 and /-separated -hr/-ha
#      - ~/.PRIDE_PPPAR_BIN/config_template has "G15 E15 C25 J15"
#      - PRIDE atx (~/Projects/.../PRIDE-PPPAR/table/igs20_*.atx) has TRSAX4E02
#   2. Stop any running PPP / convbin processes.
#   3. Wipe per-day RINEX and per-day PPP output trees.
#      Preserve the shared IGS-product cache at $PPP/products (re-download
#      would be hours).
#   4. Re-convert all .ubx → RINEX in parallel.
#   5. Run PPP on every day via run_ppp.sh.
#   6. Regenerate the four analysis tables.
#
# Usage:
#   ./reprocess_all.sh                 # default: --jobs 16
#   ./reprocess_all.sh 24              # custom jobs
#   ./reprocess_all.sh --dry-run       # show plan, do nothing
#   ./reprocess_all.sh 16 --dry-run    # both
#
# Logs:
#   /tmp/rinex_full.log
#   /tmp/ppp_full.log
#   /tmp/analytics.log

set -uo pipefail

# ---- arg parsing ----
JOBS=16
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
WORK_DIR="${ROOT_DIR}/work"
RINEX_DIR="${WORK_DIR}/rinex"
PPP_DIR="${WORK_DIR}/ppp"
PRIDE_TABLE="${HOME}/Projects/gps-data/data/code/PRIDE-PPPAR/table"
PRIDE_BIN_CFG="${HOME}/.PRIDE_PPPAR_BIN/config_template"

stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
die()   { stamp "ERROR: $*" >&2; exit 1; }

# ---- Phase 1: sanity checks ----
stamp "[1/6] Sanity-checking the three fixes are in place..."

# 1a. convbin_runner.py has TRSAX4E02 and /-separated -hr/-ha
if ! grep -q 'antenna_type: str = "TRSAX4E02"' "${CODE_DIR}/rinex/convbin_runner.py"; then
  die "convbin_runner.py does not declare TRSAX4E02 as default antenna_type. Sync the latest version from local."
fi
if ! grep -q '"-hr", f"0/' "${CODE_DIR}/rinex/convbin_runner.py"; then
  die "convbin_runner.py does not use /-separated -hr/-ha. Sync the latest version."
fi
stamp "  convbin_runner.py: OK"

# 1b. PRIDE config_template has the right freq combo
if [[ ! -r "$PRIDE_BIN_CFG" ]]; then
  die "PRIDE config_template not found at $PRIDE_BIN_CFG"
fi
if ! grep -q '^Frequency combination.*=.*G15 E15 C25 J15' "$PRIDE_BIN_CFG"; then
  die "PRIDE config_template still has default frequency combination. Edit it: change 'Default' to 'G15 E15 C25 J15'."
fi
stamp "  PRIDE config_template: OK"

# 1c. TRSAX4E02 in PRIDE atx
if ! grep -q 'TRSAX4E02' "${PRIDE_TABLE}/igs20_2388.atx" 2>/dev/null; then
  die "TRSAX4E02 not in PRIDE atx. Merge the NGS calibration into igs20_*.atx files."
fi
stamp "  PRIDE atx has TRSAX4E02: OK"

# ---- Phase 2: stop running processes ----
stamp "[2/6] Killing any running convbin / pdp3 / run_ppp / xargs processes..."
if $DRY; then
  pgrep -af 'pdp3|pride_runner|run_ppp.sh|xargs|convbin' || stamp "  (none running)"
else
  for p in run_ppp.sh xargs "python -m ppp.pride_runner" pdp3 PRIDE_PPPAR_BIN "python -m rinex.convbin_runner" convbin; do
    pkill -9 -f "$p" 2>/dev/null || true
  done
  sleep 3
  remaining="$(pgrep -af 'xargs|pride_runner|pdp3|PRIDE_PPPAR|run_ppp.sh|convbin' || true)"
  if [[ -n "$remaining" ]]; then
    stamp "  WARNING: processes still alive after kill:"
    echo "$remaining"
  else
    stamp "  Clean."
  fi
fi

# ---- Phase 3: wipe per-day outputs, preserve products cache ----
stamp "[3/6] Wiping per-day RINEX and PPP output trees (preserving IGS products cache)..."
if $DRY; then
  stamp "  (dry-run) would do:"
  stamp "    mv $PPP_DIR/products /tmp/_ppp_products_keep"
  stamp "    rm -rf $PPP_DIR/[0-9]*"
  stamp "    mv /tmp/_ppp_products_keep $PPP_DIR/products"
  stamp "    rm -rf $RINEX_DIR/[0-9]*"
else
  if [[ -d "$PPP_DIR/products" ]]; then
    mv "$PPP_DIR/products" /tmp/_ppp_products_keep || die "Failed to move products cache aside"
    n=$(ls /tmp/_ppp_products_keep | wc -l)
    sz=$(du -sh /tmp/_ppp_products_keep | awk '{print $1}')
    stamp "  Cached $n day-subdirs ($sz) aside."
  fi
  rm -rf "$PPP_DIR"/[0-9]* || true
  if [[ -d /tmp/_ppp_products_keep ]]; then
    mv /tmp/_ppp_products_keep "$PPP_DIR/products" || die "Failed to restore products cache"
    stamp "  Restored products cache to $PPP_DIR/products"
  fi
  rm -rf "$RINEX_DIR"/[0-9]* || true
  stamp "  Wipe complete."
fi

# ---- Phase 4: re-convert RINEX ----
stamp "[4/6] Re-converting all .ubx files to RINEX..."
if $DRY; then
  stamp "  (dry-run) would launch: python -m rinex.convbin_runner > /tmp/rinex_full.log 2>&1 &"
else
  cd "$CODE_DIR"
  nohup python -m rinex.convbin_runner > /tmp/rinex_full.log 2>&1 &
  RINEX_PID=$!
  stamp "  Launched (PID $RINEX_PID). Waiting for RINEX conversion to complete..."
  wait $RINEX_PID
  rinex_rc=$?
  if [[ $rinex_rc -ne 0 ]]; then
    die "RINEX conversion exited with rc=$rinex_rc. See /tmp/rinex_full.log"
  fi
  n_rinex=$(find "$RINEX_DIR" -name 'obs.rnx' | wc -l)
  stamp "  Done. $n_rinex RINEX OBS files written."
fi

# ---- Phase 5: PPP on all days ----
stamp "[5/6] Running PPP on all days (jobs=$JOBS)..."
if $DRY; then
  stamp "  (dry-run) would launch: ./run_ppp.sh --jobs $JOBS --ppp-only > /tmp/ppp_full.log 2>&1 &"
else
  cd "$CODE_DIR"
  nohup ./run_ppp.sh --jobs "$JOBS" --ppp-only > /tmp/ppp_full.log 2>&1 &
  PPP_PID=$!
  stamp "  Launched (PID $PPP_PID). PPP runs in background under nohup."
  stamp "  Monitor: tail -f /tmp/ppp_full.log"
  stamp "  This script will now exit; analytics (phase 6) must be run manually"
  stamp "  AFTER PPP completes:"
  cat <<EOF

    cd "$CODE_DIR"
    python -m ppp.compare                      > /tmp/analytics.log 2>&1
    python -m ppp.percentiles                  >> /tmp/analytics.log 2>&1
    python -m analysis.jamming_confound        >> /tmp/analytics.log 2>&1
    python -m analysis.internal_consistency    >> /tmp/analytics.log 2>&1

EOF
  stamp "  When all four return, send T_pos_percentiles.csv back for the paper table."
  disown 2>/dev/null || true
fi

stamp "[6/6] Script done. PPP continues in background."
exit 0
