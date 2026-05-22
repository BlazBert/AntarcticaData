"""Diagnose PRIDE 3.x kin file parsing for one day.

Usage:
    python3 -m ppp._diag_kin /path/to/kin_2025273_jsi1

Prints:
  - distribution of field counts per data row
  - distribution of column-3 flag values
  - parser output (lat/lon/h range, first/last rows)
  - 3 raw worst-residual data rows (after sanity clip would have rejected them)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python3 -m ppp._diag_kin <kin_path>")
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"missing: {path}")
        return 2

    text = path.read_text().splitlines()
    is_3x = any("END OF HEADER" in line for line in text)
    print(f"file: {path}")
    print(f"is_3x: {is_3x}")
    print(f"total lines: {len(text)}")

    # Collect data rows
    in_data = not is_3x
    data_rows: list[list[str]] = []
    for raw in text:
        if not in_data:
            if "END OF HEADER" in raw:
                in_data = True
            continue
        line = raw.strip()
        if not line or line.startswith(("*", "#")):
            continue
        data_rows.append(line.split())
    print(f"data rows: {len(data_rows)}")

    # Field-count histogram
    nf = Counter(len(r) for r in data_rows)
    print()
    print("Field count distribution:")
    for n, c in sorted(nf.items()):
        print(f"  {n} fields: {c:>6} rows")

    # Column-3 flag distribution (the * / + / etc.)
    flags = Counter(r[2] if len(r) > 2 else "<short>" for r in data_rows)
    print()
    print("Column-3 flag distribution (top 10):")
    for f, c in flags.most_common(10):
        print(f"  {f!r:<6} {c:>6}")

    # For each unique field count, show one example row
    print()
    print("One example row per field count:")
    seen = set()
    for r in data_rows:
        if len(r) not in seen:
            seen.add(len(r))
            print(f"  [{len(r)} fields] {' '.join(r[:20])}{' ...' if len(r) > 20 else ''}")

    # Try the parser
    print()
    print("Parser output:")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ppp.compare import _read_pride_kin
        df = _read_pride_kin(path)
        print(f"  rows parsed: {df.height}")
        if df.height:
            print(f"  lat range:   {df['lat'].min():.4f} .. {df['lat'].max():.4f}")
            print(f"  lon range:   {df['lon'].min():.4f} .. {df['lon'].max():.4f}")
            print(f"  h_ell range: {df['h_ell'].min():.4f} .. {df['h_ell'].max():.4f}")
            print(f"  first 3:")
            for row in df.head(3).iter_rows(named=True):
                print(f"    {row}")
            print(f"  last 3:")
            for row in df.tail(3).iter_rows(named=True):
                print(f"    {row}")
    except Exception as exc:
        print(f"  parser raised: {exc!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
