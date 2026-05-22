"""Per-day and cross-day analysis modules.

Each module reads from ``staging/<yyyymmdd>/*.parquet`` and the per-day
Zarr group, and writes to ``derived/`` and ``tables/``. Modules are
independent (no imports between analysis/* files) so they can be invoked
in any order from the Snakefile.
"""
