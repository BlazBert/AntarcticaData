"""Shared helpers across analysis modules.

Things that are too small to live in ``ubx.*`` and that the per-day
analysis modules reuse: config loading, day enumeration, Polars frame
helpers, geographic utilities (haversine), GPS-week / iTOW conversion.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

# --------------------------------------------------------------------------
# Paths and config
# --------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent.parent  # code/
CONFIG_DIR = CODE_DIR / "config"


def load_config(name: str = "pipeline") -> dict[str, Any]:
    """Load a YAML config file from ``code/config/``."""
    p = CONFIG_DIR / f"{name}.yaml"
    with p.open() as fh:
        return yaml.safe_load(fh)


def resolve_path(rel: str, base: Path | None = None) -> Path:
    """Resolve a config-relative path against ``code/`` (default)."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return ((base or CODE_DIR) / p).resolve()


def staged_path(day: str, msg: str, cfg: dict | None = None) -> Path:
    """``staging/<day>/<msg>.parquet`` resolved to absolute path."""
    cfg = cfg or load_config()
    return resolve_path(cfg["paths"]["staging"]) / day / f"{msg}.parquet"


def derived_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = resolve_path(cfg["paths"]["derived"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def tables_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = resolve_path(cfg["paths"]["tables"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def figures_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = resolve_path(cfg["paths"]["figures"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_days(cfg: dict | None = None) -> list[str]:
    """Discover days that have at least nav_pvt.parquet staged."""
    cfg = cfg or load_config()
    staging = resolve_path(cfg["paths"]["staging"])
    if not staging.exists():
        return []
    days = []
    for d in sorted(staging.iterdir()):
        if d.is_dir() and (d / "nav_pvt.parquet").exists():
            days.append(d.name)
    return days


# --------------------------------------------------------------------------
# Geographic helpers
# --------------------------------------------------------------------------

R_EARTH_KM = 6371.0088


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle distance in km."""
    rlat1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    rlat2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    dlat = rlat2 - rlat1
    dlon = np.deg2rad(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64))
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R_EARTH_KM * c


def cumulative_distance_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance along a track. Length matches input."""
    if len(lat) < 2:
        return np.zeros_like(lat, dtype=np.float64)
    step = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum = np.concatenate(([0.0], np.cumsum(step)))
    return cum


# --------------------------------------------------------------------------
# Polars helpers
# --------------------------------------------------------------------------


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a Parquet file with sane defaults. Wraps ``pl.read_parquet``."""
    return pl.read_parquet(path)


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", compression_level=3)


def t_ns_to_dt(df: pl.DataFrame, col: str = "t_ns") -> pl.DataFrame:
    """Add a ``t`` column (Datetime UTC) derived from a nanosecond column."""
    return df.with_columns(
        pl.from_epoch(pl.col(col), time_unit="ns").alias("t")
    )


# --------------------------------------------------------------------------
# Frequency / wavelength helpers (GLONASS FDMA needs freqId)
# --------------------------------------------------------------------------

C_M_S = 299_792_458.0


def gnss_signal_freq(gnssId: int, sigId: int, freqId: int = 0) -> float:
    """Return carrier frequency in Hz for a (gnss, sig, freqId) triple.

    GLONASS L1OF: f1 = 1602.0 MHz + (freqId − 7) * 0.5625 MHz   (n = -7..6)
    GLONASS L2OF: f2 = 1246.0 MHz + (freqId − 7) * 0.4375 MHz
    Other constellations: per ``config/receiver.yaml``.
    """
    if gnssId == 6:  # GLO
        n = int(freqId) - 7
        if sigId == 0:
            return 1_602_000_000.0 + n * 562_500.0
        if sigId == 2:
            return 1_246_000_000.0 + n * 437_500.0
    # Fall back to typical L1 / L2/L5 frequencies
    cfg = load_config("receiver")
    name_map = cfg["gnss_names"]
    name = name_map.get(int(gnssId))
    if name is None:
        return float("nan")
    sig_table = cfg["signals"].get(name, {})
    entry = sig_table.get(str(int(sigId)))
    if entry is None:
        return float("nan")
    return float(entry["freq_hz"])


def gnss_wavelength_m(gnssId: int, sigId: int, freqId: int = 0) -> float:
    f = gnss_signal_freq(gnssId, sigId, freqId)
    if not math.isfinite(f) or f <= 0:
        return float("nan")
    return C_M_S / f


__all__ = [
    "load_config",
    "resolve_path",
    "staged_path",
    "derived_dir",
    "tables_dir",
    "figures_dir",
    "list_days",
    "haversine_km",
    "cumulative_distance_km",
    "read_parquet",
    "write_parquet",
    "t_ns_to_dt",
    "gnss_signal_freq",
    "gnss_wavelength_m",
    "C_M_S",
    "R_EARTH_KM",
]
