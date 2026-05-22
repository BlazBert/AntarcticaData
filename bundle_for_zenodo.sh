#!/usr/bin/env bash
# bundle_for_zenodo.sh — bundle raw UBX + Hatanaka RINEX into Zenodo-ready tarballs.
#
# Output layout in $OUT_DIR:
#   rinex3_obs_nav.tar.gz         (single archive, ~16 GB)
#   raw_ubx_YYYYMM.tar            (per-month .ubx, ~20-25 GB each, 8 files)
#   metadata/SHA256SUMS           (checksums for every tarball)
#   metadata/README.md            (top-level dataset description placeholder)
#
# Usage:
#   bash bundle_for_zenodo.sh
#   UBX_DIR=... RINEX_DIR=... OUT_DIR=... bash bundle_for_zenodo.sh

set -euo pipefail

UBX_DIR="${UBX_DIR:-$HOME/Projects/gps-data/data/antarctica2026}"
RINEX_DIR="${RINEX_DIR:-$HOME/Projects/gps-data/data/work/rinex}"
OUT_DIR="${OUT_DIR:-$HOME/Projects/gps-data/data/zenodo_recordA}"

mkdir -p "$OUT_DIR/metadata"

echo "=== bundle_for_zenodo.sh ==="
echo "UBX source:    $UBX_DIR"
echo "RINEX source:  $RINEX_DIR"
echo "Output:        $OUT_DIR"
echo

# 1. RINEX — single .tar.gz (already Hatanaka+gzipped per file, tar just concatenates)
echo "[1/3] Bundling RINEX..."
cd "$RINEX_DIR"
# include only obs.crx.gz and nav.rnx.gz — exclude convbin.log and brdm* external files
tar cf "$OUT_DIR/rinex3_obs_nav.tar" \
    $(find . \( -name 'obs.crx.gz' -o -name 'nav.rnx.gz' \) | sort)
echo "  $(du -h --apparent-size "$OUT_DIR/rinex3_obs_nav.tar" | cut -f1)  rinex3_obs_nav.tar"
echo

# 2. UBX — per-month .tar (binary, gzip wouldn't help much)
echo "[2/3] Bundling raw UBX by month..."
cd "$UBX_DIR"
MONTHS=$(ls *.ubx 2>/dev/null | sed 's/^\(....\)\(..\).*/\1\2/' | sort -u)
for ym in $MONTHS; do
    out="$OUT_DIR/raw_ubx_${ym}.tar"
    tar cf "$out" ${ym}*.ubx
    echo "  $(du -h --apparent-size "$out" | cut -f1)  $(basename "$out")"
done
echo

# 3. Checksums + size report
echo "[3/3] Computing SHA-256 checksums (parallel)..."
cd "$OUT_DIR"
find . -maxdepth 1 -name '*.tar*' -print0 \
    | xargs -0 -P 8 -I {} sha256sum {} \
    | sort > metadata/SHA256SUMS
echo "  metadata/SHA256SUMS:"
sed 's/^/    /' metadata/SHA256SUMS
echo

echo "=== Bundle summary ==="
du -shc --apparent-size *.tar 2>/dev/null | tail -1
echo
ls -lh *.tar
echo
echo "Bundle ready at: $OUT_DIR"
echo "Next: edit metadata/README.md, then run upload_to_zenodo.py"
