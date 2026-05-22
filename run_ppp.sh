#!/usr/bin/env bash
# Full RINEX + kinematic-PPP pipeline for the 216-day cruise.
#
# Idempotent: re-running this skips days whose RINEX (obs.rnx) or PPP
# (kin_*) outputs already exist. Safe to invoke under nohup.
#
# Usage:
#   ./run_ppp.sh                          # sequential, both stages
#   ./run_ppp.sh --jobs 4                 # 4-way parallel
#   ./run_ppp.sh --rinex-only             # only RINEX
#   ./run_ppp.sh --ppp-only               # only PPP (assumes RINEX exists)
#   ./run_ppp.sh --jobs 4 --log /tmp/x.log
#
# Notes on parallelism:
#   - convbin is single-threaded per file; IO-bound. Safe to run up to
#     ~16 concurrent days on a single NVMe.
#   - pdp3 uses ~2 threads internally and writes per-day output. PRIDE
#     also caches downloaded IGS products in a shared directory
#     (~/.PRIDE_PPPAR/data) — first day populates the cache so let it
#     run alone the first time, then bump --jobs for the rest.

set -uo pipefail  # NOT -e: per-day failure must not abort the loop

# ------------------------------------------------------------------
# Paths (resolved relative to the location of this script)
# ------------------------------------------------------------------
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CODE_DIR}/.." && pwd)"
UBX_DIR="${ROOT_DIR}/antarctica2026"
WORK_DIR="${ROOT_DIR}/work"
RINEX_DIR="${WORK_DIR}/derived/rinex"
PPP_DIR="${WORK_DIR}/ppp"

# ------------------------------------------------------------------
# Defaults & arg parsing
# ------------------------------------------------------------------
JOBS=1
DO_RINEX=true
DO_PPP=true
LOG="/tmp/ppp_full_run.log"
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rinex-only) DO_PPP=false; shift ;;
    --ppp-only)   DO_RINEX=false; shift ;;
    --jobs)       JOBS="${2:-1}"; shift 2 ;;
    --log)        LOG="${2}"; shift 2 ;;
    --force)      FORCE=true; shift ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#$//'
      exit 0 ;;
    *)
      echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
log() {
  local msg="$(date '+%Y-%m-%d %H:%M:%S') $*"
  echo "$msg" | tee -a "$LOG"
}

have_rinex() {
  [[ -s "$RINEX_DIR/$1/obs.rnx" ]]
}

have_ppp() {
  # PRIDE writes kin_* into either <yyyy>/<doy>/ or <yyyy>/<doy1>-<doy2>/
  compgen -G "$PPP_DIR/$1/[0-9][0-9][0-9][0-9]/*/kin_*" >/dev/null
}

# Worker functions executed via xargs subshells; need exported state.
run_one_rinex() {
  local day="$1"
  if ! $FORCE && have_rinex "$day"; then
    echo "$(date '+%H:%M:%S') SKIP-RINEX $day (already done)"
    return 0
  fi
  cd "$CODE_DIR" || return 99
  local t0=$(date +%s)
  if python -m rinex.convbin_runner --day "$day" >/dev/null 2>&1; then
    local dt=$(( $(date +%s) - t0 ))
    echo "$(date '+%H:%M:%S') OK-RINEX   $day  (${dt}s)"
  else
    echo "$(date '+%H:%M:%S') FAIL-RINEX $day"
  fi
}
export -f run_one_rinex have_rinex
export CODE_DIR RINEX_DIR FORCE

run_one_ppp() {
  local day="$1"
  if ! $FORCE && have_ppp "$day"; then
    echo "$(date '+%H:%M:%S') SKIP-PPP   $day (already done)"
    return 0
  fi
  if ! have_rinex "$day"; then
    echo "$(date '+%H:%M:%S') NO-RINEX   $day  (run RINEX first)"
    return 0
  fi
  cd "$CODE_DIR" || return 99
  local t0=$(date +%s)
  if python -m ppp.pride_runner --day "$day" >/dev/null 2>&1; then
    local dt=$(( $(date +%s) - t0 ))
    echo "$(date '+%H:%M:%S') OK-PPP     $day  (${dt}s)"
  else
    echo "$(date '+%H:%M:%S') FAIL-PPP   $day"
  fi
}
export -f run_one_ppp have_ppp
export PPP_DIR

# ------------------------------------------------------------------
# Discover days
# ------------------------------------------------------------------
if [[ ! -d "$UBX_DIR" ]]; then
  log "ERROR: UBX dir not found: $UBX_DIR"
  exit 1
fi
mapfile -t ALL_DAYS < <(
  cd "$UBX_DIR" && ls *.ubx 2>/dev/null | sed 's/\.ubx$//' | sort
)
TOTAL=${#ALL_DAYS[@]}
if (( TOTAL == 0 )); then
  log "ERROR: no .ubx files in $UBX_DIR"
  exit 1
fi

> "$LOG"  # truncate
log "============================================================"
log " PPP full-cruise driver"
log "============================================================"
log " UBX dir:    $UBX_DIR ($TOTAL days)"
log " RINEX dir:  $RINEX_DIR"
log " PPP dir:    $PPP_DIR"
log " Log file:   $LOG"
log " Parallel:   $JOBS"
log " Stages:     rinex=$DO_RINEX  ppp=$DO_PPP   force=$FORCE"
log "============================================================"

# ------------------------------------------------------------------
# RINEX stage
# ------------------------------------------------------------------
if $DO_RINEX; then
  todo=0
  for d in "${ALL_DAYS[@]}"; do
    if $FORCE || ! have_rinex "$d"; then todo=$((todo+1)); fi
  done
  log "RINEX: $todo of $TOTAL days need conversion"
  if (( todo > 0 )); then
    printf '%s\n' "${ALL_DAYS[@]}" \
      | xargs -n1 -P"$JOBS" -I{} bash -c 'run_one_rinex "$@"' _ {} \
      | tee -a "$LOG"
  fi
  log "RINEX stage complete"
fi

# ------------------------------------------------------------------
# PPP stage
# ------------------------------------------------------------------
if $DO_PPP; then
  todo=0
  for d in "${ALL_DAYS[@]}"; do
    if have_rinex "$d" && { $FORCE || ! have_ppp "$d"; }; then
      todo=$((todo+1))
    fi
  done
  log "PPP: $todo of $TOTAL days need processing"
  if (( todo > 0 )); then
    printf '%s\n' "${ALL_DAYS[@]}" \
      | xargs -n1 -P"$JOBS" -I{} bash -c 'run_one_ppp "$@"' _ {} \
      | tee -a "$LOG"
  fi
  log "PPP stage complete"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
n_rinex=0
n_ppp=0
n_no_rinex=0
n_no_ppp=0
for d in "${ALL_DAYS[@]}"; do
  if have_rinex "$d"; then n_rinex=$((n_rinex+1)); else n_no_rinex=$((n_no_rinex+1)); fi
  if have_ppp   "$d"; then n_ppp=$((n_ppp+1));   else n_no_ppp=$((n_no_ppp+1));   fi
done

log "============================================================"
log " SUMMARY"
log "------------------------------------------------------------"
log " Expected days:       $TOTAL"
log " RINEX successful:    $n_rinex"
log " RINEX missing:       $n_no_rinex"
log " PPP successful:      $n_ppp"
log " PPP missing:         $n_no_ppp"
log "============================================================"

if (( n_no_ppp > 0 )); then
  log "Days missing PPP output:"
  for d in "${ALL_DAYS[@]}"; do
    if ! have_ppp "$d"; then echo "  $d" | tee -a "$LOG"; fi
  done
fi

log "Done. Next steps:"
log "  python -m ppp.compare              # aggregate diffs"
log "  python -m ppp.percentiles          # generate the manuscript table"
log "  python3 -m ppp._diag_diff --all    # full-cruise sanity sweep"
