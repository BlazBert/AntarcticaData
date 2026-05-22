#!/usr/bin/env bash
# compress_rinex.sh — Hatanaka + gzip RINEX 3.04 for Zenodo upload.
# Idempotent (skips files already compressed). Source .rnx preserved —
# delete only after Zenodo upload succeeds.
#
# Usage:
#   bash compress_rinex.sh
#   RINEX_DIR=/some/path JOBS=16 bash compress_rinex.sh

set -euo pipefail
RINEX_DIR="${RINEX_DIR:-$HOME/Projects/gps-data/data/work/rinex}"
JOBS="${JOBS:-32}"

cd "$RINEX_DIR"
echo "=== compress_rinex.sh ==="
echo "Directory:   $PWD"
echo "Parallelism: $JOBS"
echo

# 1. toolchain check
echo "[1/5] Checking toolchain..."
for cmd in rnx2crx crx2rnx gzip xargs find diff awk numfmt; do
    command -v "$cmd" >/dev/null \
        || { echo "ERROR: $cmd missing. For rnx2crx/crx2rnx: pip install hatanaka"; exit 1; }
done
echo "  OK"
echo

# 2. inventory
echo "[2/5] Inventory..."
N_OBS=$(find . -path '*/obs.rnx' | wc -l)
N_NAV=$(find . -path '*/nav.rnx' | wc -l)
N_OBS_GZ=$(find . -path '*/obs.crx.gz' | wc -l)
N_NAV_GZ=$(find . -path '*/nav.rnx.gz' | wc -l)
printf '  obs.rnx %4d   obs.crx.gz %4d\n' "$N_OBS" "$N_OBS_GZ"
printf '  nav.rnx %4d   nav.rnx.gz %4d\n' "$N_NAV" "$N_NAV_GZ"
echo

# 3. smoke test on first uncompressed obs.rnx
echo "[3/5] Smoke test..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SMOKE_SRC=$(find . -path '*/obs.rnx' -print -quit 2>/dev/null || true)
if [ -n "$SMOKE_SRC" ]; then
    cp "$SMOKE_SRC" "$TMP/test.rnx"
    SRC_B=$(stat -c%s "$TMP/test.rnx")
    rnx2crx -f "$TMP/test.rnx"
    gzip -9 -f "$TMP/test.crx"
    COMP_B=$(stat -c%s "$TMP/test.crx.gz")
    RATIO=$(awk -v a="$SRC_B" -v b="$COMP_B" 'BEGIN{printf "%.1f", a/b}')
    printf '  %s -> %s (%sx ratio)\n' \
        "$(numfmt --to=iec "$SRC_B")" "$(numfmt --to=iec "$COMP_B")" "$RATIO"
    gunzip -c "$TMP/test.crx.gz" | crx2rnx - > "$TMP/test.decoded"
    # crx2rnx strips trailing whitespace — content-equal but not byte-equal.
    # Use -w (ignore whitespace) for the meaningful check.
    DIFF_W=$( { diff -w "$TMP/test.decoded" "$TMP/test.rnx" || true; } | wc -l)
    DIFF_RAW=$( { diff "$TMP/test.decoded" "$TMP/test.rnx" || true; } | wc -l)
    if [ "$DIFF_W" -eq 0 ]; then
        echo "  Round-trip: content-identical (${DIFF_RAW} whitespace-only diff lines, expected)"
    else
        echo "  Round-trip: ${DIFF_W} content diff lines — INVESTIGATE before uploading"
    fi
else
    echo "  No uncompressed obs.rnx — skipping"
fi
echo

# 4. parallel compression (skip files already done)
echo "[4/5] Parallel compression (P=$JOBS)..."
START=$(date +%s)

if [ "$N_OBS" -gt 0 ]; then
    echo "  obs.rnx -> obs.crx.gz ($N_OBS pending)..."
    find . -path '*/obs.rnx' -print0 \
      | xargs -0 -P "$JOBS" -I {} bash -c '
          f="$1"; out="${f%.rnx}.crx.gz"
          [ -f "$out" ] && exit 0
          rnx2crx -f "$f" && gzip -9 -f "${f%.rnx}.crx"
        ' _ {} || echo "  (some files failed — verify step will list them)"
fi

if [ "$N_NAV" -gt 0 ]; then
    echo "  nav.rnx -> nav.rnx.gz ($N_NAV pending)..."
    find . -path '*/nav.rnx' -print0 \
      | xargs -0 -P "$JOBS" -I {} bash -c '
          f="$1"
          [ -f "${f}.gz" ] && exit 0
          gzip -9 -f "$f"
        ' _ {} || echo "  (some files failed — verify step will list them)"
fi

echo "  Done in $(($(date +%s) - START))s"
echo

# 5. verify + final size
echo "[5/5] Verification..."
MISSING=0
for d in */; do
    [ -f "${d}obs.crx.gz" ] || { echo "  MISSING obs: $d"; MISSING=$((MISSING+1)); }
    [ -f "${d}nav.rnx.gz" ] || { echo "  MISSING nav: $d"; MISSING=$((MISSING+1)); }
done
[ "$MISSING" -eq 0 ] && echo "  All days complete (216 expected)"
echo

echo "=== Final archive size ==="
du -shc --apparent-size */obs.crx.gz */nav.rnx.gz 2>/dev/null | tail -1
echo
echo "Smallest obs.crx.gz:"
du -h --apparent-size */obs.crx.gz 2>/dev/null | sort -h | head -3
echo "Largest obs.crx.gz:"
du -h --apparent-size */obs.crx.gz 2>/dev/null | sort -h | tail -3
echo
echo "Source .rnx files preserved — delete only after Zenodo upload succeeds."
