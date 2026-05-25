# -*- coding: utf-8 -*-
"""Shared helpers for C LDPC encoder/channel/decoder simulations."""

import ctypes
import itertools
import os
import pickle
import warnings

import numpy as np
from numpy.ctypeslib import ndpointer


LIB_PATH = os.path.abspath("c_lib/libnrLDPC.so")
CUDA_LIB_PATH = os.path.abspath("CUDA/libnrLDPC_cuda.so")

# Values accepted by the C/CUDA LDPC entry points. Filtering them here catches
# bad sweep inputs before a long simulation reaches ctypes or device code.
VALID_QM = {1, 2, 4, 6, 8}
VALID_RV = {0, 1, 2, 3}
VALID_BGN = {1, 2}
VALID_CRCTYPE = {0, 1, 2, 3}

# B is the information block length after transport-block CRC attachment. This
# repository simulates one code block at a time, so BG1/BG2 have fixed max B.
MAX_B_BY_BGN = {
    1: 8448,
    2: 3840,
}


class LdpcInConfig(ctypes.Structure):
    """ctypes mirror of ldpc_in_config_t in c_lib/nr_ldpc.h."""

    _fields_ = [
        ("bgn", ctypes.c_uint16),
        ("A", ctypes.c_uint16),
        ("crctype", ctypes.c_uint16),
        ("E", ctypes.c_uint16),
        ("rv", ctypes.c_uint16),
        ("Qm", ctypes.c_uint16),
        ("HARQ_on", ctypes.c_uint16),
    ]


class DecoderConfig(ctypes.Structure):
    """ctypes mirror of decoder_config_t in c_lib/nr_ldpc.h."""

    _fields_ = [
        ("algo", ctypes.c_uint16),
        ("max_iterations", ctypes.c_uint16),
        ("alpha", ctypes.c_uint16),
        ("beta", ctypes.c_uint16),
        ("filling_bit_value", ctypes.c_int16),
        ("max_llr", ctypes.c_int16),
        ("min_llr", ctypes.c_int16),
    ]


def load_c_library(
    lib_path=LIB_PATH,
    encode_symbol="c_ldpc_5g_encode",
    decode_symbol="c_ldpc_5g_decode",
    channel_lib_path=None,
):
    """Load an LDPC library and register ctypes signatures.

    encode_symbol/decode_symbol let CUDA simulations reuse the same BLER
    harness with cuda_ldpc_5g_encode/cuda_ldpc_5g_decode. tx_rx_process lives
    in the C library, so channel_lib_path can point there when lib_path is the
    CUDA shared object.
    """
    lib = ctypes.CDLL(lib_path)
    channel_lib = lib if channel_lib_path is None else ctypes.CDLL(channel_lib_path)

    encode_fn = getattr(lib, encode_symbol)
    encode_fn.argtypes = [
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.POINTER(LdpcInConfig),
    ]
    encode_fn.restype = ctypes.c_int32
    lib.ldpc_5g_encode = encode_fn

    tx_rx_process = channel_lib.tx_rx_process
    tx_rx_process.argtypes = [
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_uint16,
        ctypes.c_float,
        ndpointer(dtype=np.int16, flags="C_CONTIGUOUS"),
    ]
    tx_rx_process.restype = ctypes.c_int32
    lib.tx_rx_process_fn = tx_rx_process

    decode_fn = getattr(lib, decode_symbol)
    decode_fn.argtypes = [
        ndpointer(dtype=np.int16, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.POINTER(LdpcInConfig),
        ctypes.POINTER(DecoderConfig),
    ]
    decode_fn.restype = ctypes.c_int32
    lib.ldpc_5g_decode = decode_fn

    # Keep the channel library alive when it is a separate CDLL.
    lib._channel_lib = channel_lib

    return lib


def make_decoder_config(values):
    """Build a DecoderConfig from one decoder configuration row."""
    return DecoderConfig(
        algo=values[0],
        max_iterations=values[1],
        alpha=values[2],
        beta=values[3],
        filling_bit_value=values[4],
        max_llr=values[5],
        min_llr=values[6],
    )


def format_coderate(coderate, max_denominator=1000):
    """Return a compact fraction-like label for a coderate value."""
    for denominator in range(1, max_denominator + 1):
        numerator = round(coderate * denominator)
        if abs(coderate - numerator / denominator) < 1e-12:
            return f"{numerator}/{denominator}"

    return f"{coderate:.4g}"


def make_ldpc_config(Qm, rv, B, coderate, bgn, crctype):
    """Build one LDPC input configuration for a BLER sweep point."""
    # The simulation grid uses B, matching 38.212 notation after CRC. The C
    # encoder API wants A, the payload length before CRC, so derive it here.
    A = B - 16 if crctype == 0 else B - 24
    E = int(B / coderate)

    return LdpcInConfig(
        bgn=bgn,
        A=A,
        crctype=crctype,
        E=E,
        rv=rv,
        Qm=Qm,
        HARQ_on=0,
    )


def get_crc_length(crctype):
    """Return the CRC length in bits for a numeric C CRC type."""
    return 16 if crctype == 0 else 24


def get_config_grid_rejection_reason(Qm, rv, B, coderate, bgn, crctype):
    """Return why a grid tuple is unsupported, or None if it is usable."""
    # The grid is often built from broad lists. Reject impossible combinations
    # early so the reported "Generated N configurations" reflects runnable work.
    if Qm not in VALID_QM:
        return f"Qm must be one of {sorted(VALID_QM)}"
    if rv not in VALID_RV:
        return f"rv must be one of {sorted(VALID_RV)}"
    if bgn not in VALID_BGN:
        return f"bgn must be one of {sorted(VALID_BGN)}"
    if crctype not in VALID_CRCTYPE:
        return f"crctype must be one of {sorted(VALID_CRCTYPE)}"
    if coderate <= 0 or coderate > 1:
        return "coderate must be in the range (0, 1]"
    if B <= get_crc_length(crctype):
        return "B must be larger than the selected CRC length"

    A = B - get_crc_length(crctype)
    if A % 8 != 0:
        return "A, derived as B minus CRC length, must be a multiple of 8"
    if B > MAX_B_BY_BGN[bgn]:
        return f"BG{bgn} supports B <= {MAX_B_BY_BGN[bgn]}"

    # Rate matching emits E bits, which are interleaved by modulation order.
    # E must therefore be a valid codeword length and align to Qm.
    E = int(B / coderate)
    if E < B:
        return "E, derived as int(B / coderate), must be at least B"
    if E % Qm != 0:
        return "E, derived as int(B / coderate), must be a multiple of Qm"

    return None


def make_config_grid(Qm_list, rv_list, B_list, coderate_list, bgn_list,
    crctype_list, strict=False):
    """Return the supported Cartesian product of LDPC test parameters."""
    config_grid = []
    skipped_reasons = {}

    for config_tuple in itertools.product(
        Qm_list, rv_list, B_list, coderate_list, bgn_list, crctype_list
    ):
        reason = get_config_grid_rejection_reason(*config_tuple)
        if reason is None:
            config_grid.append(config_tuple)
            continue

        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        if strict:
            raise ValueError(f"unsupported LDPC config {config_tuple}: {reason}")

    if skipped_reasons:
        # Group rejected tuples by reason to keep large sweeps readable.
        reason_text = ", ".join(
            f"{count} skipped because {reason}"
            for reason, count in sorted(skipped_reasons.items())
        )
        warnings.warn(
            f"make_config_grid skipped unsupported LDPC configs: {reason_text}",
            stacklevel=2,
        )

    if not config_grid:
        raise ValueError("make_config_grid did not produce any supported configs")

    return config_grid


def run_ldpc_trial(libnrLDPC, ldpc_config, decoder_config_list, noise_power_db):
    """Run one random code block through C encode, C channel, and C decode.

    Returns block-error flags and information-bit error counts for each decoder.
    """
    # NumPy stores one bit per element for convenience. The C APIs use packed
    # bytes at their boundaries, matching transport-block byte alignment.
    info_bits = np.random.randint(0, 2, ldpc_config.A).astype(np.uint8)
    info_bytes = np.packbits(info_bits, bitorder="big")

    encoded_bytes = np.zeros((ldpc_config.E + 7) // 8, dtype=np.uint8)
    status = libnrLDPC.ldpc_5g_encode(
        info_bytes, encoded_bytes, ctypes.byref(ldpc_config)
    )
    if status != 0:
        raise RuntimeError(f"LDPC encode failed with status {status}")

    llr_int16 = np.zeros(ldpc_config.E, dtype=np.int16)
    status = libnrLDPC.tx_rx_process_fn(
        encoded_bytes,
        ldpc_config.E,
        ldpc_config.Qm,
        ctypes.c_float(noise_power_db),
        llr_int16,
    )
    if status != 0:
        raise RuntimeError(f"tx_rx_process failed with status {status}")

    B = ldpc_config.A + get_crc_length(ldpc_config.crctype)
    hardbits = np.zeros((B + 7) // 8, dtype=np.uint8)
    decode_status = np.zeros(len(decoder_config_list), dtype=np.float64)
    bit_error_count = np.zeros(len(decoder_config_list), dtype=np.float64)

    for idx, config_values in enumerate(decoder_config_list):
        # The C decoders pack bits by shifting into the output bytes, so clear
        # the buffer before each algorithm uses the shared channel realization.
        hardbits.fill(0)
        decoder_config = make_decoder_config(config_values)
        status = libnrLDPC.ldpc_5g_decode(
            llr_int16, hardbits, ctypes.byref(ldpc_config), ctypes.byref(decoder_config)
        )
        decoded_bits = np.unpackbits(hardbits, bitorder="big")[:ldpc_config.A]
        decode_status[idx] = 0 if status == 0 else 1
        bit_error_count[idx] = np.count_nonzero(decoded_bits != info_bits)

    return decode_status, bit_error_count


def run_bler_sweep(libnrLDPC, ldpc_config, coderate, decoder_config_list,
    snr_power_db_list, test_num):
    """Run all SNR points for one LDPC configuration and compute BLER/BER."""
    bler_list = []
    ber_list = []

    for snr_power_db in snr_power_db_list:
        error_count = np.zeros(len(decoder_config_list), dtype=np.float64)
        bit_error_count = np.zeros(len(decoder_config_list), dtype=np.float64)
        # tx_rx_process expects noise power. Convert from SNR and include Qm so
        # different modulations use a comparable energy-per-bit convention.
        noise_power_db = -snr_power_db - 10 * np.log10(ldpc_config.Qm)

        for _ in range(test_num):
            block_errors, bit_errors = run_ldpc_trial(
                libnrLDPC, ldpc_config, decoder_config_list, noise_power_db
            )
            error_count += block_errors
            bit_error_count += bit_errors

        bler = error_count / test_num
        ber = bit_error_count / (test_num * ldpc_config.A)
        print(
            f"Qm {ldpc_config.Qm}, rv {ldpc_config.rv}, A {ldpc_config.A}, "
            f"E {ldpc_config.E}, coderate {format_coderate(coderate)}, "
            f"bgn {ldpc_config.bgn}, crc {ldpc_config.crctype}, "
            f"snr_power_db {snr_power_db:.2f}, BLER={bler}, BER={ber}"
        )
        bler_list.append(bler)
        ber_list.append(ber)

    return bler_list, ber_list


def run_simulation(config_grid, decoder_config_list, snr_power_db_list, test_num,
    result_file, lib_path=LIB_PATH, encode_symbol="c_ldpc_5g_encode",
    decode_symbol="c_ldpc_5g_decode", channel_lib_path=None):
    """Run a full BLER/BER simulation and save results to a pickle file."""
    libnrLDPC = load_c_library(
        lib_path,
        encode_symbol=encode_symbol,
        decode_symbol=decode_symbol,
        channel_lib_path=channel_lib_path,
    )
    print(f"Generated {len(config_grid)} configurations.")

    test_result = []
    for config_tuple in config_grid:
        Qm, rv, B, coderate, bgn, crctype = config_tuple
        # Keep this historical guard for callers that pass a hand-built grid
        # instead of using make_config_grid().
        if bgn == 2 and B > 3840:
            continue

        ldpc_config = make_ldpc_config(Qm, rv, B, coderate, bgn, crctype)
        bler_list, ber_list = run_bler_sweep(
            libnrLDPC, ldpc_config, coderate, decoder_config_list,
            snr_power_db_list, test_num
        )
        test_result.append([config_tuple, bler_list, ber_list])

    with open(result_file, "wb") as f:
        # Store the sweep metadata beside results so plotting scripts can label
        # decoder curves and SNR points without re-importing this module.
        pickle.dump(
            (test_result, decoder_config_list, snr_power_db_list, test_num), f
        )

    return test_result
