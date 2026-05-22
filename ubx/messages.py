"""Per-message UBX decoders.

Each decoder takes a payload ``memoryview`` and returns a ``dict[str, ndarray]``
of column arrays (already aligned to the schemas in ``ubx.schemas``). The
caller is responsible for:

* providing ``t_ns`` (host wall-clock receive time, ns since epoch — see
  ``parallel.py``: we use the receiver's PVT-derived UTC for consistency
  across files);
* concatenating arrays across many messages and handing them to
  ``writers.ParquetSink`` in chunks.

The repeating substructures (RAWX measurements, NAV-SAT per-SV blocks,
MON-SPAN RF blocks, RXM-MEASX per-SV blocks) use ``np.frombuffer(buf, dtype=...)``
over a *structured dtype* — no per-message Python loops on the hot path.

Bit-level decoding (e.g. NAV-SAT.flags) uses NumPy boolean operations on the
flat array; again no Python loops.
"""

from __future__ import annotations

import struct
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Message identifiers
# ---------------------------------------------------------------------------


class MessageName(str, Enum):
    """Stable string identifiers — used as Parquet table names."""

    NAV_PVT = "nav_pvt"
    NAV_HPPOSLLH = "nav_hpposllh"
    NAV_SAT = "nav_sat"
    RXM_RAWX = "rxm_rawx"
    RXM_SFRBX = "rxm_sfrbx"
    RXM_MEASX = "rxm_measx"
    MON_RF = "mon_rf"
    MON_SYS = "mon_sys"
    MON_SPAN = "mon_span"  # written to Zarr, not Parquet


# (class, id) → message name
CLASS_ID_TO_NAME: dict[tuple[int, int], MessageName] = {
    (0x01, 0x07): MessageName.NAV_PVT,
    (0x01, 0x14): MessageName.NAV_HPPOSLLH,
    (0x01, 0x35): MessageName.NAV_SAT,
    (0x02, 0x13): MessageName.RXM_SFRBX,
    (0x02, 0x14): MessageName.RXM_MEASX,
    (0x02, 0x15): MessageName.RXM_RAWX,
    (0x0A, 0x31): MessageName.MON_SPAN,
    (0x0A, 0x38): MessageName.MON_RF,
    (0x0A, 0x39): MessageName.MON_SYS,
}


# ---------------------------------------------------------------------------
# Structured dtypes for repeating substructures
# ---------------------------------------------------------------------------

# RXM-RAWX measurement record (32 bytes)
RAWX_MEAS_DT = np.dtype(
    [
        ("prMes", "<f8"),
        ("cpMes", "<f8"),
        ("doMes", "<f4"),
        ("gnssId", "u1"),
        ("svId", "u1"),
        ("sigId", "u1"),
        ("freqId", "u1"),
        ("locktime", "<u2"),
        ("cno", "u1"),
        ("prStdev", "u1"),
        ("cpStdev", "u1"),
        ("doStdev", "u1"),
        ("trkStat", "u1"),
        ("reserved", "u1"),
    ],
    align=False,
)
assert RAWX_MEAS_DT.itemsize == 32

# NAV-SAT per-SV block (12 bytes)
NAV_SAT_BLK_DT = np.dtype(
    [
        ("gnssId", "u1"),
        ("svId", "u1"),
        ("cno", "u1"),
        ("elev", "i1"),
        ("azim", "<i2"),
        ("prRes", "<i2"),
        ("flags", "<u4"),
    ],
    align=False,
)
assert NAV_SAT_BLK_DT.itemsize == 12

# MON-RF per-block record (24 bytes)
MON_RF_BLK_DT = np.dtype(
    [
        ("blockId", "u1"),
        ("flags", "u1"),
        ("antStatus", "u1"),
        ("antPower", "u1"),
        ("postStatus", "<u4"),
        ("reserved2", "<u4"),
        ("noisePerMS", "<u2"),
        ("agcCnt", "<u2"),
        ("jamInd", "u1"),
        ("ofsI", "i1"),
        ("magI", "u1"),
        ("ofsQ", "i1"),
        ("magQ", "u1"),
        ("reserved3_a", "u1"),
        ("reserved3_b", "u1"),
        ("reserved3_c", "u1"),
    ],
    align=False,
)
assert MON_RF_BLK_DT.itemsize == 24

# MON-SPAN per-RF-block record (272 bytes): 256 bin samples + 16 bytes meta
MON_SPAN_BLK_DT = np.dtype(
    [
        ("spectrum", "u1", (256,)),
        ("span", "<u4"),
        ("res", "<u4"),
        ("center", "<u4"),
        ("pga", "u1"),
        ("reserved2", "u1", (3,)),
    ],
    align=False,
)
assert MON_SPAN_BLK_DT.itemsize == 272

# RXM-MEASX per-SV block (24 bytes)
RXM_MEASX_BLK_DT = np.dtype(
    [
        ("gnssId", "u1"),
        ("svId", "u1"),
        ("cNo", "u1"),
        ("mpathIndic", "u1"),
        ("dopplerMS", "<i4"),
        ("dopplerHz", "<i4"),
        ("wholeChips", "<u2"),
        ("fracChips", "<u2"),
        ("codePhase", "<u4"),
        ("intCodePhase", "u1"),
        ("pseuRangeRMSErr", "u1"),
        ("reserved5", "u1", (2,)),
    ],
    align=False,
)
assert RXM_MEASX_BLK_DT.itemsize == 24


# ---------------------------------------------------------------------------
# Header structs (fixed-size — Struct objects amortise format compilation)
# ---------------------------------------------------------------------------

_S_NAV_PVT = struct.Struct(
    "<I H B B B B B B I i B B B B i i i i I I i i i i i I I H H 4s i h H"
)  # 92 bytes (the 4s skips reserved0 at offset 80)
assert _S_NAV_PVT.size == 92

_S_NAV_SAT_HDR2 = struct.Struct("<I B B 2s")  # iTOW(U4) version(U1) numSvs(U1) reserved0(2)
assert _S_NAV_SAT_HDR2.size == 8

_S_RXM_RAWX_HDR = struct.Struct("<d H b B B B 2s")  # 16 bytes
assert _S_RXM_RAWX_HDR.size == 16

_S_RXM_SFRBX_HDR = struct.Struct("<B B B B B B B B")  # 8 bytes
assert _S_RXM_SFRBX_HDR.size == 8

_S_RXM_MEASX_HDR = struct.Struct("<B 3s I I I 4s I H H H 2s H B B 8s")  # 44 bytes
assert _S_RXM_MEASX_HDR.size == 44

_S_MON_RF_HDR = struct.Struct("<B B 2s")  # 4 bytes (version, nBlocks, reserved0)
assert _S_MON_RF_HDR.size == 4

_S_MON_SPAN_HDR = struct.Struct("<B B 2s")  # 4 bytes
assert _S_MON_SPAN_HDR.size == 4

_S_MON_SYS = struct.Struct("<B B B B B B B B I H H H b B 4s")  # 24 bytes
assert _S_MON_SYS.size == 24


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _utc_to_ns(year: int, month: int, day: int, hour: int, minute: int, sec: int, nano: int) -> int:
    """Convert UBX UTC fields to ns since 1970-01-01.

    NAV-PVT carries (year, month, day, hour, min, sec, nano). ``nano`` may be
    negative (it's a signed offset to the integer second). We compute via
    ``np.datetime64`` for correctness around month boundaries.

    Returns 0 for any out-of-range / invalid date — the receiver emits
    ``year=2025, month=0`` and similar before it has acquired time.
    """
    if not (1970 <= year <= 2200):
        return 0
    if not (1 <= month <= 12):
        return 0
    if not (1 <= day <= 31):
        return 0
    if not (0 <= hour <= 23):
        return 0
    if not (0 <= minute <= 59):
        return 0
    if not (0 <= sec <= 60):  # 60 allowed for leap second
        return 0
    try:
        s = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec:02d}"
        base = np.datetime64(s, "ns").astype("int64")
    except (ValueError, OverflowError):
        return 0
    return int(base) + int(nano)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------
# Each ``decode_*`` returns a dict[str, np.ndarray | scalar]. Scalars are
# wrapped to length-N arrays in the writer when concatenating.

_NaN64 = np.float64(np.nan)
_NaN32 = np.float32(np.nan)


def decode_nav_pvt(payload: memoryview, _ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """NAV-PVT (0x01 0x07). Returns one row of arrays, length 1."""
    if len(payload) < 92:
        return None
    (
        iTOW,
        year,
        month,
        day,
        hour,
        minute,
        sec,
        valid,
        tAcc,
        nano,
        fixType,
        flags,
        flags2,
        numSV,
        lon,
        lat,
        height,
        hMSL,
        hAcc,
        vAcc,
        velN,
        velE,
        velD,
        gSpeed,
        headMot,
        sAcc,
        headAcc,
        pDOP,
        flags3,
        _r0,
        headVeh,
        magDec,
        magAcc,
    ) = _S_NAV_PVT.unpack_from(payload, 0)
    t_ns = _utc_to_ns(year, month, day, hour, minute, sec, nano)
    return {
        "t_ns": np.array([t_ns], dtype=np.int64),
        "iTOW": np.array([iTOW], dtype=np.uint32),
        "year": np.array([year], dtype=np.uint16),
        "month": np.array([month], dtype=np.uint8),
        "day": np.array([day], dtype=np.uint8),
        "hour": np.array([hour], dtype=np.uint8),
        "min": np.array([minute], dtype=np.uint8),
        "sec": np.array([sec], dtype=np.uint8),
        "valid": np.array([valid], dtype=np.uint8),
        "tAcc": np.array([tAcc], dtype=np.uint32),
        "nano": np.array([nano], dtype=np.int32),
        "fixType": np.array([fixType], dtype=np.uint8),
        "flags": np.array([flags], dtype=np.uint8),
        "flags2": np.array([flags2], dtype=np.uint8),
        "numSV": np.array([numSV], dtype=np.uint8),
        "lon_1e7": np.array([lon], dtype=np.int32),
        "lat_1e7": np.array([lat], dtype=np.int32),
        "height_mm": np.array([height], dtype=np.int32),
        "hMSL_mm": np.array([hMSL], dtype=np.int32),
        "hAcc_mm": np.array([hAcc], dtype=np.uint32),
        "vAcc_mm": np.array([vAcc], dtype=np.uint32),
        "velN_mm_s": np.array([velN], dtype=np.int32),
        "velE_mm_s": np.array([velE], dtype=np.int32),
        "velD_mm_s": np.array([velD], dtype=np.int32),
        "gSpeed_mm_s": np.array([gSpeed], dtype=np.int32),
        "headMot_1e5": np.array([headMot], dtype=np.int32),
        "sAcc": np.array([sAcc], dtype=np.uint32),
        "headAcc": np.array([headAcc], dtype=np.uint32),
        "pDOP": np.array([pDOP], dtype=np.uint16),
        "flags3": np.array([flags3], dtype=np.uint16),
        "headVeh_1e5": np.array([headVeh], dtype=np.int32),
        "magDec_1e2": np.array([magDec], dtype=np.int16),
        "magAcc_1e2": np.array([magAcc], dtype=np.uint16),
    }


def decode_nav_hpposllh(payload: memoryview, ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """NAV-HPPOSLLH (0x01 0x14). 36 B.

    Layout: version(U1) reserved0(2) flags(U1) iTOW(U4) lon(I4) lat(I4)
    height(I4) hMSL(I4) lonHp(I1) latHp(I1) heightHp(I1) hMSLHp(I1)
    hAcc(U4) vAcc(U4).
    """
    if len(payload) < 36:
        return None
    version = payload[0]
    flags = payload[3]
    iTOW = struct.unpack_from("<I", payload, 4)[0]
    lon, lat, height, hMSL = struct.unpack_from("<iiii", payload, 8)
    lonHp, latHp, heightHp, hMSLHp = struct.unpack_from("<bbbb", payload, 24)
    hAcc, vAcc = struct.unpack_from("<II", payload, 28)
    # ctx['t_ns_pvt'] carries the latest NAV-PVT time so HPPOSLLH joins on the
    # same epoch (HPPOSLLH itself doesn't carry UTC). Fallback: 0.
    t_ns = int(ctx.get("t_ns_pvt", 0))
    return {
        "t_ns": np.array([t_ns], dtype=np.int64),
        "version": np.array([version], dtype=np.uint8),
        "flags": np.array([flags], dtype=np.uint8),
        "iTOW": np.array([iTOW], dtype=np.uint32),
        "lon_1e7": np.array([lon], dtype=np.int32),
        "lat_1e7": np.array([lat], dtype=np.int32),
        "height_mm": np.array([height], dtype=np.int32),
        "hMSL_mm": np.array([hMSL], dtype=np.int32),
        "lonHp_1e9": np.array([lonHp], dtype=np.int8),
        "latHp_1e9": np.array([latHp], dtype=np.int8),
        "heightHp_0p1mm": np.array([heightHp], dtype=np.int8),
        "hMSLHp_0p1mm": np.array([hMSLHp], dtype=np.int8),
        "hAcc_0p1mm": np.array([hAcc], dtype=np.uint32),
        "vAcc_0p1mm": np.array([vAcc], dtype=np.uint32),
    }


def decode_nav_sat(payload: memoryview, ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """NAV-SAT (0x01 0x35). Variable length: 8 + 12*numSvs."""
    if len(payload) < 8:
        return None
    iTOW, version, numSvs, _r = _S_NAV_SAT_HDR2.unpack_from(payload, 0)
    if numSvs == 0 or len(payload) < 8 + 12 * numSvs:
        # Empty / truncated — emit nothing
        return None
    blocks = np.frombuffer(payload, dtype=NAV_SAT_BLK_DT, count=numSvs, offset=8)
    flags = blocks["flags"]
    # Decode bitfields vectorised
    qualityInd = (flags & 0x07).astype(np.uint8)
    svUsed = ((flags >> 3) & 0x01).astype(bool)
    health = ((flags >> 4) & 0x03).astype(np.uint8)
    diffCorr = ((flags >> 6) & 0x01).astype(bool)
    smoothed = ((flags >> 7) & 0x01).astype(bool)
    orbitSource = ((flags >> 8) & 0x07).astype(np.uint8)
    ephAvail = ((flags >> 11) & 0x01).astype(bool)
    almAvail = ((flags >> 12) & 0x01).astype(bool)
    anoAvail = ((flags >> 13) & 0x01).astype(bool)
    aopAvail = ((flags >> 14) & 0x01).astype(bool)
    sbasCorrUsed = ((flags >> 16) & 0x01).astype(bool)
    rtcmCorrUsed = ((flags >> 17) & 0x01).astype(bool)
    slasCorrUsed = ((flags >> 18) & 0x01).astype(bool)
    prCorrUsed = ((flags >> 20) & 0x01).astype(bool)
    crCorrUsed = ((flags >> 21) & 0x01).astype(bool)
    doCorrUsed = ((flags >> 22) & 0x01).astype(bool)
    n = numSvs
    t_ns = int(ctx.get("t_ns_pvt", 0))
    return {
        "t_ns": np.full(n, t_ns, dtype=np.int64),
        "iTOW": np.full(n, iTOW, dtype=np.uint32),
        "gnssId": blocks["gnssId"].copy(),
        "svId": blocks["svId"].copy(),
        "cno": blocks["cno"].copy(),
        "elev": blocks["elev"].copy(),
        "azim": blocks["azim"].copy(),
        "prRes_0p1m": blocks["prRes"].copy(),
        "flags": flags.copy(),
        "qualityInd": qualityInd,
        "svUsed": svUsed,
        "health": health,
        "diffCorr": diffCorr,
        "smoothed": smoothed,
        "orbitSource": orbitSource,
        "ephAvail": ephAvail,
        "almAvail": almAvail,
        "anoAvail": anoAvail,
        "aopAvail": aopAvail,
        "sbasCorrUsed": sbasCorrUsed,
        "rtcmCorrUsed": rtcmCorrUsed,
        "slasCorrUsed": slasCorrUsed,
        "prCorrUsed": prCorrUsed,
        "crCorrUsed": crCorrUsed,
        "doCorrUsed": doCorrUsed,
    }


def decode_rxm_rawx(payload: memoryview, ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """RXM-RAWX (0x02 0x15). Variable length: 16 + 32*numMeas."""
    if len(payload) < 16:
        return None
    rcvTow, week, leapS, numMeas, recStat, version, _r = _S_RXM_RAWX_HDR.unpack_from(payload, 0)
    if numMeas == 0 or len(payload) < 16 + 32 * numMeas:
        return None
    blocks = np.frombuffer(payload, dtype=RAWX_MEAS_DT, count=numMeas, offset=16)
    n = numMeas
    t_ns = int(ctx.get("t_ns_pvt", 0))
    return {
        "t_ns": np.full(n, t_ns, dtype=np.int64),
        "rcvTow": np.full(n, rcvTow, dtype=np.float64),
        "week": np.full(n, week, dtype=np.uint16),
        "leapS": np.full(n, leapS, dtype=np.int8),
        "recStat": np.full(n, recStat, dtype=np.uint8),
        "version": np.full(n, version, dtype=np.uint8),
        "prMes": blocks["prMes"].copy(),
        "cpMes": blocks["cpMes"].copy(),
        "doMes": blocks["doMes"].copy(),
        "gnssId": blocks["gnssId"].copy(),
        "svId": blocks["svId"].copy(),
        "sigId": blocks["sigId"].copy(),
        "freqId": blocks["freqId"].copy(),
        "locktime_ms": blocks["locktime"].copy(),
        "cno": blocks["cno"].copy(),
        "prStdev": blocks["prStdev"].copy(),
        "cpStdev": blocks["cpStdev"].copy(),
        "doStdev": blocks["doStdev"].copy(),
        "trkStat": blocks["trkStat"].copy(),
    }


def decode_rxm_sfrbx(payload: bytes, ctx: dict[str, Any]) -> dict[str, Any] | None:
    """RXM-SFRBX (0x02 0x13). Variable length: 8 + 4*numWords.

    ``dwrd_bytes`` is the raw little-endian dword block — much faster to
    write to Parquet than ``list<uint32>``.
    """
    if len(payload) < 8:
        return None
    gnssId, svId, sigId, freqId, numWords, chn, version, _r = _S_RXM_SFRBX_HDR.unpack_from(payload, 0)
    end = 8 + 4 * numWords
    if len(payload) < end:
        return None
    dwrd = bytes(payload[8:end])
    t_ns = int(ctx.get("t_ns_pvt", 0))
    return {
        "t_ns": np.array([t_ns], dtype=np.int64),
        "gnssId": np.array([gnssId], dtype=np.uint8),
        "svId": np.array([svId], dtype=np.uint8),
        "sigId": np.array([sigId], dtype=np.uint8),
        "freqId": np.array([freqId], dtype=np.uint8),
        "numWords": np.array([numWords], dtype=np.uint8),
        "chn": np.array([chn], dtype=np.uint8),
        "version": np.array([version], dtype=np.uint8),
        "dwrd_bytes": [dwrd],  # length-1 list — Parquet writer concatenates
    }


def decode_rxm_measx(payload: memoryview, ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """RXM-MEASX (0x02 0x14). Variable: 44 + 24*numSV."""
    if len(payload) < 44:
        return None
    (
        version,
        _r,
        gpsTOW,
        gloTOW,
        bdsTOW,
        _r1,
        qzssTOW,
        gpsAcc,
        gloAcc,
        bdsAcc,
        _r2,
        qzssAcc,
        numSV,
        flags,
        _r3,
    ) = _S_RXM_MEASX_HDR.unpack_from(payload, 0)
    if numSV == 0 or len(payload) < 44 + 24 * numSV:
        return None
    blocks = np.frombuffer(payload, dtype=RXM_MEASX_BLK_DT, count=numSV, offset=44)
    n = numSV
    t_ns = int(ctx.get("t_ns_pvt", 0))
    return {
        "t_ns": np.full(n, t_ns, dtype=np.int64),
        "version": np.full(n, version, dtype=np.uint8),
        "gpsTOW": np.full(n, gpsTOW, dtype=np.uint32),
        "gloTOW": np.full(n, gloTOW, dtype=np.uint32),
        "bdsTOW": np.full(n, bdsTOW, dtype=np.uint32),
        "qzssTOW": np.full(n, qzssTOW, dtype=np.uint32),
        "gpsTOWacc_2e_4ms": np.full(n, gpsAcc, dtype=np.uint16),
        "gloTOWacc_2e_4ms": np.full(n, gloAcc, dtype=np.uint16),
        "bdsTOWacc_2e_4ms": np.full(n, bdsAcc, dtype=np.uint16),
        "qzssTOWacc_2e_4ms": np.full(n, qzssAcc, dtype=np.uint16),
        "numSV": np.full(n, numSV, dtype=np.uint8),
        "flags": np.full(n, flags, dtype=np.uint8),
        "gnssId": blocks["gnssId"].copy(),
        "svId": blocks["svId"].copy(),
        "cNo": blocks["cNo"].copy(),
        "mpathIndic": blocks["mpathIndic"].copy(),
        "dopplerMS_0p04m_s": blocks["dopplerMS"].copy(),
        "dopplerHz_0p2hz": blocks["dopplerHz"].copy(),
        "wholeChips": blocks["wholeChips"].copy(),
        "fracChips": blocks["fracChips"].copy(),
        "codePhase_2e_21ms": blocks["codePhase"].copy(),
        "intCodePhase_ms": blocks["intCodePhase"].copy(),
        "pseuRangeRMSErr": blocks["pseuRangeRMSErr"].copy(),
    }


def decode_mon_rf(payload: memoryview, ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """MON-RF (0x0A 0x38). Variable: 4 + 24*nBlocks."""
    if len(payload) < 4:
        return None
    version, nBlocks, _r = _S_MON_RF_HDR.unpack_from(payload, 0)
    if nBlocks == 0 or len(payload) < 4 + 24 * nBlocks:
        return None
    blocks = np.frombuffer(payload, dtype=MON_RF_BLK_DT, count=nBlocks, offset=4)
    n = nBlocks
    t_ns = int(ctx.get("t_ns_pvt", 0))
    return {
        "t_ns": np.full(n, t_ns, dtype=np.int64),
        "version": np.full(n, version, dtype=np.uint8),
        "nBlocks": np.full(n, nBlocks, dtype=np.uint8),
        "blockId": blocks["blockId"].copy(),
        "flags": blocks["flags"].copy(),
        "antStatus": blocks["antStatus"].copy(),
        "antPower": blocks["antPower"].copy(),
        "postStatus": blocks["postStatus"].copy(),
        "noisePerMS": blocks["noisePerMS"].copy(),
        "agcCnt": blocks["agcCnt"].copy(),
        "jamInd": blocks["jamInd"].copy(),
        "ofsI": blocks["ofsI"].copy(),
        "magI": blocks["magI"].copy(),
        "ofsQ": blocks["ofsQ"].copy(),
        "magQ": blocks["magQ"].copy(),
    }


def decode_mon_sys(payload: memoryview, ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """MON-SYS (0x0A 0x39). 24 B."""
    if len(payload) < 24:
        return None
    (
        msgVer,
        bootType,
        cpuLoad,
        cpuLoadMax,
        memUsage,
        memUsageMax,
        ioUsage,
        ioUsageMax,
        runTime,
        noticeCount,
        warnCount,
        errorCount,
        tempValue,
        tempState,
        _r,
    ) = _S_MON_SYS.unpack_from(payload, 0)
    t_ns = int(ctx.get("t_ns_pvt", 0))
    return {
        "t_ns": np.array([t_ns], dtype=np.int64),
        "msgVer": np.array([msgVer], dtype=np.uint8),
        "bootType": np.array([bootType], dtype=np.uint8),
        "cpuLoad": np.array([cpuLoad], dtype=np.uint8),
        "cpuLoadMax": np.array([cpuLoadMax], dtype=np.uint8),
        "memUsage": np.array([memUsage], dtype=np.uint8),
        "memUsageMax": np.array([memUsageMax], dtype=np.uint8),
        "ioUsage": np.array([ioUsage], dtype=np.uint8),
        "ioUsageMax": np.array([ioUsageMax], dtype=np.uint8),
        "runTime_s": np.array([runTime], dtype=np.uint32),
        "noticeCount": np.array([noticeCount], dtype=np.uint16),
        "warnCount": np.array([warnCount], dtype=np.uint16),
        "errorCount": np.array([errorCount], dtype=np.uint16),
        "tempValue_C": np.array([tempValue], dtype=np.int8),
        "tempState": np.array([tempState], dtype=np.uint8),
    }


def decode_mon_span(payload: memoryview, ctx: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """MON-SPAN (0x0A 0x31). Variable: 4 + 272*numRfBlocks.

    Returns spectrum and per-block metadata as aligned arrays — caller writes
    these to the Zarr store via ``ZarrSpanSink``.
    """
    if len(payload) < 4:
        return None
    version, nBlocks, _r = _S_MON_SPAN_HDR.unpack_from(payload, 0)
    if nBlocks == 0 or len(payload) < 4 + 272 * nBlocks:
        return None
    blocks = np.frombuffer(payload, dtype=MON_SPAN_BLK_DT, count=nBlocks, offset=4)
    t_ns = int(ctx.get("t_ns_pvt", 0))
    # spectrum: shape (nBlocks, 256) uint8
    spectrum = blocks["spectrum"].copy()
    return {
        "t_ns": np.array([t_ns], dtype=np.int64),
        "version": np.array([version], dtype=np.uint8),
        "nBlocks": np.array([nBlocks], dtype=np.uint8),
        # per-block arrays of length nBlocks
        "spectrum": spectrum,                            # (nBlocks, 256)
        "span_hz": blocks["span"].copy(),
        "res_hz": blocks["res"].copy(),
        "center_hz": blocks["center"].copy(),
        "pga_db": blocks["pga"].copy(),
    }


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

MSG_DECODERS: dict[MessageName, Any] = {
    MessageName.NAV_PVT: decode_nav_pvt,
    MessageName.NAV_HPPOSLLH: decode_nav_hpposllh,
    MessageName.NAV_SAT: decode_nav_sat,
    MessageName.RXM_RAWX: decode_rxm_rawx,
    MessageName.RXM_SFRBX: decode_rxm_sfrbx,
    MessageName.RXM_MEASX: decode_rxm_measx,
    MessageName.MON_RF: decode_mon_rf,
    MessageName.MON_SYS: decode_mon_sys,
    MessageName.MON_SPAN: decode_mon_span,
}
