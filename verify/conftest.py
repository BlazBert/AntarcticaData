"""pytest fixtures: the 60-second clip used as a golden sample."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parent.parent
FIXTURE_DIR = CODE_DIR / "verify" / "fixtures"
SAMPLE_60S = FIXTURE_DIR / "sample_60s.ubx"


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory) -> Path:
    """Return the path to ``sample_60s.ubx``; build it lazily on first use.

    The clip is a 60-second slice from the 2025-09-30 reference file. We
    build it via the ``ubx`` package itself: read frames, keep those
    spanning 60 s of receiver time, write back as raw UBX bytes.
    """
    if SAMPLE_60S.exists() and SAMPLE_60S.stat().st_size > 0:
        return SAMPLE_60S
    src = CODE_DIR.parent / "antarctica2026" / "data" / "20250930.ubx"
    if not src.exists():
        pytest.skip(f"Reference UBX file missing: {src}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Walk frames; collect bytes of the first ~60 s of receiver time. We
    # use NAV-PVT as the time anchor.
    from ubx.parser import iter_frames, open_mmap  # noqa: PLC0415
    from ubx.messages import CLASS_ID_TO_NAME, MessageName, MSG_DECODERS  # noqa: PLC0415

    out: list[bytes] = []
    t0_ns: int | None = None
    target_ns = 60_000_000_000
    with open_mmap(src) as mm:
        i = 0
        ctx = {"t_ns_pvt": 0}
        for fr in iter_frames(mm):
            # Reconstruct full UBX frame bytes to keep checksum integrity.
            length = fr.payload_length
            full_len = 6 + length + 2
            frame_bytes = bytes(mm[fr.payload_offset - 6 : fr.payload_offset - 6 + full_len])
            out.append(frame_bytes)
            if (fr.class_id, fr.msg_id) == (0x01, 0x07):
                d = MSG_DECODERS[MessageName.NAV_PVT](fr.payload, ctx)
                if d is not None:
                    t = int(d["t_ns"][0])
                    if t > 0:
                        if t0_ns is None:
                            t0_ns = t
                        elif t - t0_ns > target_ns:
                            break
            i += 1
    SAMPLE_60S.write_bytes(b"".join(out))
    return SAMPLE_60S
