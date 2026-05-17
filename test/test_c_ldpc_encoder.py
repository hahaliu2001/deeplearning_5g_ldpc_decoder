# -*- coding: utf-8 -*-
"""Verify the C 5G LDPC encoder against the Python reference encoder.

For each test case, the same random information bits and LDPC configuration are
sent through:

* Python reference: ``py5gphy.ldpc_5g_encoder.cb_ldpc_5g_encode()``
* C shared library: ``c_ldpc_5g_encode()``

The C encoder returns packed bytes, so the test unpacks those bytes and compares
the first ``E`` rate-matched bits against the Python one-bit-per-element output.
"""

import ctypes
from pathlib import Path
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

if __name__ == "__main__":
    # Direct execution starts with test/ on sys.path. Add the repo root so the
    # py5gphy package imports the same way it does under pytest.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pytest
except ModuleNotFoundError:
    pytest = None

from py5gphy import ldpc_5g_encoder


LIB_PATH = Path(__file__).resolve().parents[1] / "c_lib" / "libnrLDPC.so"

CRC_TYPES = {
    "16": 0,
    "24A": 1,
    "24B": 2,
    "24C": 3,
}

ZC_LIST = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
    20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 208, 224, 240, 256, 288, 320, 352, 384,
]

CODE_RATES = [1 / 8, 1 / 5, 1 / 3, 1 / 2, 1]
MODULATION_ORDERS = [1, 2, 4, 6, 8]
REDUNDANCY_VERSIONS = [0, 1, 2, 3]


def _all_encoder_cases():
    """Generate the full encoder configuration sweep used by this test."""
    cases = []
    for Zc in ZC_LIST:
        for bgn in [1, 2]:
            A = 24 * Zc
            if bgn == 1 and A > 8448-24:
                continue
            
            if bgn == 2 and A > 3840 - 24:
                continue

            for crcpoly in CRC_TYPES:
                for rv in REDUNDANCY_VERSIONS:
                    for coderate in CODE_RATES:
                        E = int((A + 48) / coderate)
                        for Qm in MODULATION_ORDERS:
                            cases.append((A, E, Qm, rv, crcpoly, bgn))

    return cases


ENCODER_CASES = _all_encoder_cases()


class LdpcInConfig(ctypes.Structure):
    """ctypes mirror of ``ldpc_in_config_t`` from c_lib/nr_ldpc.h."""

    _fields_ = [
        ("bgn", ctypes.c_uint16),
        ("A", ctypes.c_uint16),
        ("crctype", ctypes.c_uint16),
        ("E", ctypes.c_uint16),
        ("rv", ctypes.c_uint16),
        ("Qm", ctypes.c_uint16),
        ("HARQ_on", ctypes.c_uint16),
    ]


def _load_ldpc_library():
    """Load the C LDPC shared library or skip when it has not been built."""
    if not LIB_PATH.exists():
        message = f"{LIB_PATH} does not exist; run make in c_lib first"
        if pytest is not None:
            pytest.skip(message)
        raise FileNotFoundError(message)

    lib = ctypes.CDLL(str(LIB_PATH))
    lib.c_ldpc_5g_encode.argtypes = [
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.POINTER(LdpcInConfig),
    ]
    lib.c_ldpc_5g_encode.restype = ctypes.c_int32
    return lib


def _make_config(A, E, Qm, rv, crcpoly, bgn):
    """Build the C encoder input configuration."""
    config = LdpcInConfig()
    config.bgn = bgn
    config.A = A
    config.crctype = CRC_TYPES[crcpoly]
    config.E = E
    config.rv = rv
    config.Qm = Qm
    config.HARQ_on = 0
    return config


def _random_info_bits(A, seed):
    """Return deterministic random payload bits for a test case."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, A, dtype=np.uint8)


def _python_encode(info_bits, E, Qm, rv, crcpoly, bgn):
    """Run the Python reference encoder."""
    try:
        return ldpc_5g_encoder.cb_ldpc_5g_encode(
            info_bits=info_bits,
            E=E,
            Qm=Qm,
            rv=rv,
            crcpoly=crcpoly,
            bgn=bgn,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        message = "scipy is required by the Python LDPC reference encoder"
        if pytest is not None:
            pytest.skip(message)
        raise


def _c_encode(lib, info_bits, config):
    """Run the C encoder and return unpacked rate-matched output bits."""
    info_bytes = np.packbits(info_bits, bitorder="big")
    rm_bytes = np.zeros((config.E + 7) // 8, dtype=np.uint8)

    status = lib.c_ldpc_5g_encode(info_bytes, rm_bytes, config)
    assert status == 0

    # c_ldpc_5g_encode() packs E output bits into bytes. The final byte may
    # contain padding bits, so callers compare only the first E unpacked bits.
    return np.unpackbits(rm_bytes, bitorder="big")[: config.E]


def compare_c_and_python_encoder(lib, A, E, Qm, rv, crcpoly, bgn):
    """Compare C and Python LDPC encoder outputs for one configuration."""
    seed = A + E + 10 * Qm + 100 * rv + 1000 * bgn + CRC_TYPES[crcpoly]
    info_bits = _random_info_bits(A, seed)
    config = _make_config(A, E, Qm, rv, crcpoly, bgn)

    py_codeword = _python_encode(info_bits, E, Qm, rv, crcpoly, bgn)
    c_codeword = _c_encode(lib, info_bits, config)

    assert np.array_equal(py_codeword, c_codeword)


if pytest is not None:
    encoder_case_params = pytest.mark.parametrize(
        "A, E, Qm, rv, crcpoly, bgn",
        ENCODER_CASES,
    )
    library_fixture = pytest.fixture(scope="module")
else:
    encoder_case_params = lambda func: func
    library_fixture = lambda func: func


@library_fixture
def libnrLDPC():
    return _load_ldpc_library()


@encoder_case_params
def test_nr_ldpc_encoder_matches_python(libnrLDPC, A, E, Qm, rv, crcpoly, bgn):
    compare_c_and_python_encoder(libnrLDPC, A, E, Qm, rv, crcpoly, bgn)


if __name__ == "__main__":
    try:
        library = _load_ldpc_library()
        for idx, case in enumerate(ENCODER_CASES, start=1):
            A, E, Qm, rv, crcpoly, bgn = case
            print(
                f"test {idx}/{len(ENCODER_CASES)}: "
                f"A={A}, E={E}, Qm={Qm}, rv={rv}, crc={crcpoly}, bgn={bgn}"
            )
            compare_c_and_python_encoder(library, *case)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(0)
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        print("scipy is required by the Python LDPC reference encoder")
        raise SystemExit(0)

    print("All C LDPC encoder checks passed")
