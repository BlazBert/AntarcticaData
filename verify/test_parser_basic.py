"""Smoke tests for the UBX parser core."""

from __future__ import annotations

from ubx.messages import CLASS_ID_TO_NAME, MSG_DECODERS, MessageName
from ubx.parser import fletcher8, fletcher8_np, iter_frames, open_mmap

import numpy as np


def test_fletcher8_matches_numpy():
    payload = bytes(range(64)) + b"\x05\x06"
    a, b = fletcher8(payload, 0, len(payload))
    arr = np.frombuffer(payload, dtype=np.uint8)
    a2, b2 = fletcher8_np(arr)
    assert (a, b) == (a2, b2)


def test_iter_frames_finds_known_messages(sample_clip):
    counts = {}
    with open_mmap(sample_clip) as mm:
        for fr in iter_frames(mm):
            counts[(fr.class_id, fr.msg_id)] = counts.get((fr.class_id, fr.msg_id), 0) + 1
    # Must contain every recognised message type
    seen_names = {
        CLASS_ID_TO_NAME.get(k) for k in counts.keys() if k in CLASS_ID_TO_NAME
    }
    expected = {
        MessageName.NAV_PVT,
        MessageName.NAV_HPPOSLLH,
        MessageName.NAV_SAT,
        MessageName.RXM_RAWX,
        MessageName.RXM_SFRBX,
        MessageName.RXM_MEASX,
        MessageName.MON_RF,
        MessageName.MON_SYS,
        MessageName.MON_SPAN,
    }
    missing = expected - seen_names
    assert not missing, f"missing message types in clip: {missing}"


def test_decoders_run_without_error(sample_clip):
    seen: dict[MessageName, int] = {}
    ctx = {"t_ns_pvt": 0}
    with open_mmap(sample_clip) as mm:
        for fr in iter_frames(mm):
            name = CLASS_ID_TO_NAME.get((fr.class_id, fr.msg_id))
            if name is None:
                continue
            d = MSG_DECODERS[name](fr.payload, ctx)
            if d is None:
                continue
            seen[name] = seen.get(name, 0) + 1
            if name is MessageName.NAV_PVT:
                ctx["t_ns_pvt"] = int(d["t_ns"][0])
    assert len(seen) == 9


def test_nav_pvt_field_sanity(sample_clip):
    """A few hand-checked invariants for NAV-PVT."""
    ctx = {"t_ns_pvt": 0}
    with open_mmap(sample_clip) as mm:
        for fr in iter_frames(mm):
            if (fr.class_id, fr.msg_id) != (0x01, 0x07):
                continue
            d = MSG_DECODERS[MessageName.NAV_PVT](fr.payload, ctx)
            if d is None:
                continue
            year = int(d["year"][0])
            month = int(d["month"][0])
            day_ = int(d["day"][0])
            assert 2025 <= year <= 2026, year
            assert 1 <= month <= 12
            assert 1 <= day_ <= 31
            assert 0 <= int(d["fixType"][0]) <= 5
            assert -90e7 <= int(d["lat_1e7"][0]) <= 90e7
            assert -180e7 <= int(d["lon_1e7"][0]) <= 180e7
            return
    raise AssertionError("no NAV-PVT in clip")
