"""Low-level UBX frame iterator.

A UBX frame is laid out as:

    sync1 sync2  class id  length(LE u16)  payload[length]  ck_a ck_b

with sync1=0xB5, sync2=0x62. The Fletcher-8 checksum is computed over
[class, id, length_lo, length_hi, payload].

This module exposes a single function ``iter_frames(mm)`` which walks an
``mmap.mmap`` and yields ``(class_id, msg_id, payload_offset, payload_length, mv)``
where ``mv`` is a zero-copy ``memoryview`` over the payload.

The scanner is tolerant of garbage between frames: when sync bytes are not
followed by a valid Fletcher-8 checksum the scanner advances by one byte and
keeps searching. This is robust against truncated tails (common when a logger
is killed mid-frame).
"""

from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np

UBX_SYNC1: int = 0xB5
UBX_SYNC2: int = 0x62

# Header layout for unpack_from on a 6-byte slice starting at the first sync byte
# is: 'BBBBH' for sync1, sync2, class_id, msg_id, length(u16-LE).
_HEADER_LE_U16 = struct.Struct("<H")


class UbxFrame(NamedTuple):
    """A successfully-checksummed UBX frame.

    ``payload`` is a copied ``bytes`` object so the frame outlives the
    source mmap. The copy is a single per-message allocation (~50 B–2 KB)
    which is dominated by parsing cost, not memory bandwidth.
    """

    class_id: int  # 0x01, 0x02, 0x0A, ...
    msg_id: int  # within-class message id
    payload_offset: int  # absolute offset of the payload within the source buffer
    payload_length: int  # in bytes
    payload: bytes  # copied — safe to retain after mmap closes


def fletcher8(buf: bytes | memoryview, start: int, end: int) -> tuple[int, int]:
    """Fletcher-8 over buf[start:end]. Returns (ck_a, ck_b).

    NOTE: kept as a fallback / for tests. The hot loop uses NumPy below.
    """
    ck_a = 0
    ck_b = 0
    for byte in buf[start:end]:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b


def fletcher8_np(arr: np.ndarray) -> tuple[int, int]:
    """Vectorised Fletcher-8 for a 1-D uint8 array.

    Equivalent to:

        ck_a = sum(arr)               mod 256
        ck_b = sum(arr_i * (n-i))     mod 256

    where the second term is the well-known Fletcher rearrangement.

    Returns Python ``int`` (not numpy scalars) so the caller can compare
    them to ``mm[off]`` directly without a type-promotion surprise on
    older NumPy versions.
    """
    n = arr.shape[0]
    if n == 0:
        return 0, 0
    # Convert numpy scalars to Python int *before* the bitwise mask. Older
    # NumPy versions reject ``np.uint64 & int(0xFF)`` under strict casting.
    a_arr = arr.astype(np.int64, copy=False)
    a = int(a_arr.sum()) & 0xFF
    weights = np.arange(n, 0, -1, dtype=np.int64)
    b = int((a_arr * weights).sum()) & 0xFF
    return a, b


@contextmanager
def open_mmap(path: str | Path) -> Iterator[mmap.mmap]:
    """Memory-map a file read-only. Yields the ``mmap.mmap``."""
    p = Path(path)
    with p.open("rb") as fh:
        # length=0 maps the whole file
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            yield mm


def iter_frames(mm: mmap.mmap | bytes) -> Iterator[UbxFrame]:
    """Walk a UBX byte stream and yield validated frames.

    Implementation note: we deliberately do **not** wrap ``mm`` in a
    ``np.frombuffer`` view. Doing so pins the mmap (numpy holds a buffer
    export), which means any exception raised from inside the iterator
    triggers a ``BufferError: cannot close exported pointers exist`` when
    the outer mmap context manager tries to close. Instead we use
    ``mm.find(b"\\xb5\\x62", i)`` for the sync-byte scan (a single C call
    per hop, no Python-level scanning) and pass per-frame ``bytes`` copies
    into the checksum routines. ``bytes`` slices of an ``mmap`` are
    independent of it, so closing the mmap is always safe.
    """
    n = len(mm)
    # mm.find/mmap supports `bytes` patterns; bytes objects support .find too.
    sync = b"\xb5\x62"
    i = 0
    while True:
        i = mm.find(sync, i)
        if i < 0:
            return
        # Need at least 8 bytes for header + checksum
        if i + 8 > n:
            return
        # Fixed 4-byte sub-header: class, id, length(u16-LE)
        cls = mm[i + 2]
        mid = mm[i + 3]
        length = _HEADER_LE_U16.unpack_from(mm, i + 4)[0]
        end_payload = i + 6 + length
        if end_payload + 2 > n:
            # Truncated tail — advance past this sync and keep scanning.
            i += 1
            continue
        # Checksum over [class, id, length_lo, length_hi, payload]. We always
        # take a bytes copy of the body — cheap (≤2 KB per frame) and the
        # only way to keep the mmap unpinned across exceptions.
        body = mm[i + 2 : end_payload]   # bytes
        if length <= 64:
            ck_a, ck_b = fletcher8(body, 0, length + 4)
        else:
            ck_a, ck_b = fletcher8_np(np.frombuffer(body, dtype=np.uint8))
        if mm[end_payload] != ck_a or mm[end_payload + 1] != ck_b:
            i += 1
            continue
        payload_offset = i + 6
        # ``body[4:]`` is the payload portion of the bytes copy we already made.
        yield UbxFrame(cls, mid, payload_offset, length, body[4:])
        i = end_payload + 2


def parse_file(path: str | Path) -> dict[tuple[int, int], int]:
    """Quick utility: count frames by (class_id, msg_id). For smoke-tests only."""
    counts: dict[tuple[int, int], int] = {}
    with open_mmap(path) as mm:
        for frame in iter_frames(mm):
            key = (frame.class_id, frame.msg_id)
            counts[key] = counts.get(key, 0) + 1
    return counts
