#!/usr/bin/env bash
# bundle_code_for_zenodo.sh — snapshot the code/ directory as a citable archive.
#
# Strips build artefacts, caches, secrets. Adds LICENSE + VERSION if missing.
# Output is a single tar.gz suitable for Zenodo software-record upload.
#
# Usage:
#   bash bundle_code_for_zenodo.sh
#   VERSION=2026.05.21 bash bundle_code_for_zenodo.sh

set -euo pipefail

CODE_DIR="${CODE_DIR:-$HOME/Projects/gps-data/data/code}"
OUT_DIR="${OUT_DIR:-$HOME/Projects/gps-data/data/zenodo_recordC}"
VERSION="${VERSION:-$(date +%Y.%m.%d)}"

STAGE="$OUT_DIR/antarctica-gnss-pipeline-v${VERSION}"
rm -rf "$STAGE"
mkdir -p "$STAGE" "$OUT_DIR/metadata"

echo "=== bundle_code_for_zenodo.sh ==="
echo "Source:  $CODE_DIR"
echo "Output:  $OUT_DIR"
echo "Version: v${VERSION}"
echo

# 1. clean copy of source
echo "[1/4] Copying source (excluding caches, build artefacts, secrets)..."
rsync -a \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '*.pyo' \
    --exclude '.ipynb_checkpoints/' \
    --exclude '.snakemake/' \
    --exclude '.git/' \
    --exclude '.venv/' --exclude 'venv/' --exclude '.uv/' \
    --exclude '*.egg-info/' --exclude 'dist/' --exclude 'build/' \
    --exclude '.pytest_cache/' --exclude '.mypy_cache/' --exclude '.ruff_cache/' \
    --exclude '*.parquet' --exclude '*.zarr/' --exclude '*.npz' \
    --exclude '*.log' --exclude '*.pid' \
    --exclude '.env' --exclude '.env.*' --exclude 'secrets/' \
    --exclude 'work/' --exclude 'staging/' --exclude 'derived/' \
    --exclude 'figures/output/' --exclude 'tables/output/' \
    --exclude '.DS_Store' \
    "$CODE_DIR/" "$STAGE/"

NFILES=$(find "$STAGE" -type f | wc -l)
SIZE=$(du -sh "$STAGE" | cut -f1)
echo "  files: $NFILES   size: $SIZE"
echo

# 2. LICENSE (MIT default — edit if needed)
if [ ! -f "$STAGE/LICENSE" ] && [ ! -f "$STAGE/LICENSE.txt" ] && [ ! -f "$STAGE/LICENSE.md" ]; then
    echo "[2/4] Writing default MIT LICENSE (edit if you want a different license)..."
    cat > "$STAGE/LICENSE" << 'EOF'
MIT License

Copyright (c) 2026 Bertrand Bertalanic / Jožef Stefan Institute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
EOF
else
    echo "[2/4] LICENSE present — keeping existing"
fi
echo

# 3. VERSION file
echo "[3/4] Writing VERSION..."
cat > "$STAGE/VERSION" << EOF
${VERSION}

Snapshot date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')
Source path:   ${CODE_DIR}
Hostname:      $(hostname)
EOF
echo

# 4. tarball + checksum
echo "[4/4] Creating tarball + checksum..."
cd "$OUT_DIR"
TARBALL="antarctica-gnss-pipeline-v${VERSION}.tar.gz"
tar czf "$TARBALL" "antarctica-gnss-pipeline-v${VERSION}/"
sha256sum "$TARBALL" > metadata/SHA256SUMS

echo
echo "=== Bundle summary ==="
ls -lh "$TARBALL"
echo
echo "  sha256: $(cut -d' ' -f1 metadata/SHA256SUMS)"
echo
echo "Bundle ready at: $OUT_DIR/$TARBALL"
echo "Inspect with:    tar tzf $OUT_DIR/$TARBALL | head -30"
