"""Plotting helpers shared across figure scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

CODE_DIR = Path(__file__).resolve().parent.parent
STYLE_PATH = CODE_DIR / "figures" / "style.mplstyle"


def apply_style() -> None:
    """Apply the ESSD-friendly matplotlib style."""
    mpl.style.use(str(STYLE_PATH))


GNSS_COLORS = {
    0: "#E41A1C",   # GPS — red
    1: "#A65628",   # SBAS — brown
    2: "#377EB8",   # GAL — blue
    3: "#FF7F00",   # BDS — orange
    5: "#4DAF4A",   # QZSS — green
    6: "#984EA3",   # GLO — purple
    7: "#999999",   # NavIC — grey
}

GNSS_LABEL = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS", 5: "QZSS", 6: "GLO", 7: "NavIC"}


def gnss_color(gnss_id: int) -> str:
    return GNSS_COLORS.get(int(gnss_id), "#000000")
