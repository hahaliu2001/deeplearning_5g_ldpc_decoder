# -*- coding: utf-8 -*-
"""Compare the C CRC helper against the Python CRC reference."""

import ctypes
from pathlib import Path
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

if __name__ == "__main__":
    # When this file is run directly, Python sets sys.path to test/. Add the
    # repo root so py5gphy imports work the same way they do under pytest.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from py5gphy.crc import crc

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


LIB_PATH = Path(__file__).resolve().parents[1] / "c_lib" / "libnrLDPC.so"

CRC_CASES = [
    # (Python CRC polynomial name, C crctype enum value, CRC width in bits)
    ("16", 0, 16),
    ("24A", 1, 24),
    ("24B", 2, 24),
    ("24C", 3, 24),
]


def _load_crc_library():
    """Load the C LDPC shared library or skip when it has not been built."""
    if not LIB_PATH.exists():
        message = f"{LIB_PATH} does not exist; run make in c_lib first"
        if pytest is not None:
            pytest.skip(message)
        raise FileNotFoundError(message)

    lib = ctypes.CDLL(str(LIB_PATH))
    # calculate_crc(uint8_t *data, int len, int crctype) consumes byte-aligned
    # input data and returns the integer CRC remainder.
    lib.calculate_crc.argtypes = [
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.calculate_crc.restype = ctypes.c_uint32
    return lib


def _crc_to_bits(crc_value, crc_len):
    """Convert an integer CRC remainder to MSB-first bits."""
    return np.array(
        [(crc_value >> bit_idx) & 1 for bit_idx in range(crc_len - 1, -1, -1)],
        dtype=np.uint8,
    )


if pytest is not None:
    # Under pytest, expand one test function across all CRC variants and load
    # the shared library only once per module.
    crc_case_params = pytest.mark.parametrize("poly, c_crc_type, crc_len", CRC_CASES)
    library_fixture = pytest.fixture(scope="module")
else:
    # Keep direct execution lightweight even when pytest is not installed.
    crc_case_params = lambda func: func
    library_fixture = lambda func: func


@library_fixture
def libnrLDPC():
    return _load_crc_library()


@crc_case_params
def test_c_crc_matches_python(libnrLDPC, poly, c_crc_type, crc_len):
    """Check one C CRC variant against the Python bit-level reference."""
    payload_len = 256 - crc_len
    rng = np.random.default_rng(crc_len)
    info_bits = rng.integers(0, 2, payload_len, dtype=np.uint8)

    # Python works on one bit per array element. The C helper expects packed
    # bytes, so pack the same payload MSB-first before calling into ctypes.
    py_blkandcrc_bits = crc.nr_crc_encode(info_bits, poly)
    info_bytes = np.packbits(info_bits, bitorder="big")

    c_crc = libnrLDPC.calculate_crc(info_bytes, payload_len // 8, c_crc_type)
    c_crc_bits = _crc_to_bits(c_crc, crc_len)

    # Compare only the CRC tail; the Python result is payload + CRC.
    assert np.array_equal(py_blkandcrc_bits[payload_len:], c_crc_bits)


if __name__ == "__main__":
    try:
        library = _load_crc_library()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(0)

    for case in CRC_CASES:
        # Reuse the same comparison function used by pytest.
        test_c_crc_matches_python(library, *case)
    print("All C CRC checks passed")
