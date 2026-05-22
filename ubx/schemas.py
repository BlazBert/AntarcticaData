"""PyArrow schemas for the parsed UBX messages.

Single source of truth — both ``ubx.messages`` and ``ubx.writers`` import from
here. Keeping schemas in one place means a missed field shows up as a schema
mismatch at write time rather than as silently-divergent column orders.

For RAWX and NAV-SAT we keep the data in *long* form (one row per
signal-measurement, not per epoch) to make filtering by ``(gnssId, sigId, svId)``
cheap and Polars-friendly.
"""

from __future__ import annotations

import pyarrow as pa


# ---------------------------------------------------------------------------
# NAV class
# ---------------------------------------------------------------------------
NAV_PVT_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("iTOW", pa.uint32()),
        pa.field("year", pa.uint16()),
        pa.field("month", pa.uint8()),
        pa.field("day", pa.uint8()),
        pa.field("hour", pa.uint8()),
        pa.field("min", pa.uint8()),
        pa.field("sec", pa.uint8()),
        pa.field("valid", pa.uint8()),
        pa.field("tAcc", pa.uint32()),
        pa.field("nano", pa.int32()),
        pa.field("fixType", pa.uint8()),
        pa.field("flags", pa.uint8()),
        pa.field("flags2", pa.uint8()),
        pa.field("numSV", pa.uint8()),
        pa.field("lon_1e7", pa.int32()),
        pa.field("lat_1e7", pa.int32()),
        pa.field("height_mm", pa.int32()),
        pa.field("hMSL_mm", pa.int32()),
        pa.field("hAcc_mm", pa.uint32()),
        pa.field("vAcc_mm", pa.uint32()),
        pa.field("velN_mm_s", pa.int32()),
        pa.field("velE_mm_s", pa.int32()),
        pa.field("velD_mm_s", pa.int32()),
        pa.field("gSpeed_mm_s", pa.int32()),
        pa.field("headMot_1e5", pa.int32()),
        pa.field("sAcc", pa.uint32()),
        pa.field("headAcc", pa.uint32()),
        pa.field("pDOP", pa.uint16()),
        pa.field("flags3", pa.uint16()),
        pa.field("headVeh_1e5", pa.int32()),
        pa.field("magDec_1e2", pa.int16()),
        pa.field("magAcc_1e2", pa.uint16()),
    ]
)

NAV_HPPOSLLH_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("version", pa.uint8()),
        pa.field("flags", pa.uint8()),
        pa.field("iTOW", pa.uint32()),
        pa.field("lon_1e7", pa.int32()),
        pa.field("lat_1e7", pa.int32()),
        pa.field("height_mm", pa.int32()),
        pa.field("hMSL_mm", pa.int32()),
        pa.field("lonHp_1e9", pa.int8()),
        pa.field("latHp_1e9", pa.int8()),
        pa.field("heightHp_0p1mm", pa.int8()),
        pa.field("hMSLHp_0p1mm", pa.int8()),
        pa.field("hAcc_0p1mm", pa.uint32()),
        pa.field("vAcc_0p1mm", pa.uint32()),
    ]
)

# Long form: one row per (epoch, gnssId, svId)
NAV_SAT_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("iTOW", pa.uint32()),
        pa.field("gnssId", pa.uint8()),
        pa.field("svId", pa.uint8()),
        pa.field("cno", pa.uint8()),
        pa.field("elev", pa.int8()),
        pa.field("azim", pa.int16()),
        pa.field("prRes_0p1m", pa.int16()),
        pa.field("flags", pa.uint32()),
        # decoded bitfields (cheap to keep alongside, saves later joins)
        pa.field("qualityInd", pa.uint8()),
        pa.field("svUsed", pa.bool_()),
        pa.field("health", pa.uint8()),
        pa.field("diffCorr", pa.bool_()),
        pa.field("smoothed", pa.bool_()),
        pa.field("orbitSource", pa.uint8()),
        pa.field("ephAvail", pa.bool_()),
        pa.field("almAvail", pa.bool_()),
        pa.field("anoAvail", pa.bool_()),
        pa.field("aopAvail", pa.bool_()),
        pa.field("sbasCorrUsed", pa.bool_()),
        pa.field("rtcmCorrUsed", pa.bool_()),
        pa.field("slasCorrUsed", pa.bool_()),
        pa.field("prCorrUsed", pa.bool_()),
        pa.field("crCorrUsed", pa.bool_()),
        pa.field("doCorrUsed", pa.bool_()),
    ]
)


# ---------------------------------------------------------------------------
# RXM class
# ---------------------------------------------------------------------------
# Long form: one row per signal-measurement. ``epoch_id`` is a stable per-file
# epoch index so the receiver-time scalars (``rcvTow``, ``week``, ``leapS``)
# don't need to be repeated on every row when we don't want them; we still
# repeat them because Parquet zstd makes the redundancy nearly free.
RXM_RAWX_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("rcvTow", pa.float64()),
        pa.field("week", pa.uint16()),
        pa.field("leapS", pa.int8()),
        pa.field("recStat", pa.uint8()),
        pa.field("version", pa.uint8()),
        pa.field("prMes", pa.float64()),
        pa.field("cpMes", pa.float64()),
        pa.field("doMes", pa.float32()),
        pa.field("gnssId", pa.uint8()),
        pa.field("svId", pa.uint8()),
        pa.field("sigId", pa.uint8()),
        pa.field("freqId", pa.uint8()),
        pa.field("locktime_ms", pa.uint16()),
        pa.field("cno", pa.uint8()),
        pa.field("prStdev", pa.uint8()),
        pa.field("cpStdev", pa.uint8()),
        pa.field("doStdev", pa.uint8()),
        pa.field("trkStat", pa.uint8()),
    ]
)

RXM_SFRBX_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("gnssId", pa.uint8()),
        pa.field("svId", pa.uint8()),
        pa.field("sigId", pa.uint8()),
        pa.field("freqId", pa.uint8()),
        pa.field("numWords", pa.uint8()),
        pa.field("chn", pa.uint8()),
        pa.field("version", pa.uint8()),
        # Raw subframe dwords as 4*numWords little-endian bytes. This is far
        # faster to (de)serialise than ``list<uint32>`` for ~2 M rows/day.
        # Decoders re-interpret with ``np.frombuffer(..., dtype='<u4')``.
        pa.field("dwrd_bytes", pa.binary()),
    ]
)

RXM_MEASX_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("version", pa.uint8()),
        pa.field("gpsTOW", pa.uint32()),
        pa.field("gloTOW", pa.uint32()),
        pa.field("bdsTOW", pa.uint32()),
        pa.field("qzssTOW", pa.uint32()),
        pa.field("gpsTOWacc_2e_4ms", pa.uint16()),
        pa.field("gloTOWacc_2e_4ms", pa.uint16()),
        pa.field("bdsTOWacc_2e_4ms", pa.uint16()),
        pa.field("qzssTOWacc_2e_4ms", pa.uint16()),
        pa.field("numSV", pa.uint8()),
        pa.field("flags", pa.uint8()),
        # repeating block (long form)
        pa.field("gnssId", pa.uint8()),
        pa.field("svId", pa.uint8()),
        pa.field("cNo", pa.uint8()),
        pa.field("mpathIndic", pa.uint8()),
        pa.field("dopplerMS_0p04m_s", pa.int32()),
        pa.field("dopplerHz_0p2hz", pa.int32()),
        pa.field("wholeChips", pa.uint16()),
        pa.field("fracChips", pa.uint16()),
        pa.field("codePhase_2e_21ms", pa.uint32()),
        pa.field("intCodePhase_ms", pa.uint8()),
        pa.field("pseuRangeRMSErr", pa.uint8()),
    ]
)


# ---------------------------------------------------------------------------
# MON class
# ---------------------------------------------------------------------------
MON_RF_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("version", pa.uint8()),
        pa.field("nBlocks", pa.uint8()),
        pa.field("blockId", pa.uint8()),
        pa.field("flags", pa.uint8()),
        pa.field("antStatus", pa.uint8()),
        pa.field("antPower", pa.uint8()),
        pa.field("postStatus", pa.uint32()),
        pa.field("noisePerMS", pa.uint16()),
        pa.field("agcCnt", pa.uint16()),
        pa.field("jamInd", pa.uint8()),
        pa.field("ofsI", pa.int8()),
        pa.field("magI", pa.uint8()),
        pa.field("ofsQ", pa.int8()),
        pa.field("magQ", pa.uint8()),
    ]
)

MON_SYS_SCHEMA = pa.schema(
    [
        pa.field("t_ns", pa.int64()),
        pa.field("msgVer", pa.uint8()),
        pa.field("bootType", pa.uint8()),
        pa.field("cpuLoad", pa.uint8()),
        pa.field("cpuLoadMax", pa.uint8()),
        pa.field("memUsage", pa.uint8()),
        pa.field("memUsageMax", pa.uint8()),
        pa.field("ioUsage", pa.uint8()),
        pa.field("ioUsageMax", pa.uint8()),
        pa.field("runTime_s", pa.uint32()),
        pa.field("noticeCount", pa.uint16()),
        pa.field("warnCount", pa.uint16()),
        pa.field("errorCount", pa.uint16()),
        pa.field("tempValue_C", pa.int8()),
        pa.field("tempState", pa.uint8()),
    ]
)

# MON-SPAN goes to Zarr, not Parquet; schema described in writers.py.

ALL_SCHEMAS: dict[str, pa.Schema] = {
    "nav_pvt": NAV_PVT_SCHEMA,
    "nav_hpposllh": NAV_HPPOSLLH_SCHEMA,
    "nav_sat": NAV_SAT_SCHEMA,
    "rxm_rawx": RXM_RAWX_SCHEMA,
    "rxm_sfrbx": RXM_SFRBX_SCHEMA,
    "rxm_measx": RXM_MEASX_SCHEMA,
    "mon_rf": MON_RF_SCHEMA,
    "mon_sys": MON_SYS_SCHEMA,
}
