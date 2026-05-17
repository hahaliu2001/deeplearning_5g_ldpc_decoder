"""Paired evaluation against C fixed-point LDPC decoder baselines."""

from __future__ import annotations

import ctypes
import os
from typing import Dict, Optional

import numpy as np
import torch
from numpy.ctypeslib import ndpointer

from deep_learning.deeplearning_ldpc_model import CTableNeuralLayeredMinSumDecoder
from deep_learning.ldpc_model_evaluation import decode_c_table_model_with_crc_early_stop
from deep_learning.ldpc_training_utils import CRC_TYPE_TO_POLY, infer_model_device
from py5gphy.crc import crc
from py5gphy.ldpc import nr_ldpc_raterecover


DEFAULT_C_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "c_lib",
    "libnrLDPC.so",
)


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
    """ctypes mirror of ldpc_decoder_cfg_t in c_lib/nr_ldpc.h."""

    _fields_ = [
        ("algo", ctypes.c_uint16),
        ("max_iterations", ctypes.c_uint16),
        ("alpha", ctypes.c_uint16),
        ("beta", ctypes.c_uint16),
        ("filling_bit_value", ctypes.c_int16),
        ("max_llr", ctypes.c_int16),
        ("min_llr", ctypes.c_int16),
    ]


def load_c_ldpc_library(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """Load libnrLDPC.so and register the functions used for evaluation."""
    lib = ctypes.CDLL(lib_path or DEFAULT_C_LIB_PATH)

    lib.c_ldpc_5g_encode.argtypes = [
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.POINTER(LdpcInConfig),
    ]
    lib.c_ldpc_5g_encode.restype = ctypes.c_int32

    lib.tx_rx_process.argtypes = [
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_uint16,
        ctypes.c_float,
        ndpointer(dtype=np.int16, flags="C_CONTIGUOUS"),
    ]
    lib.tx_rx_process.restype = ctypes.c_int32

    lib.c_ldpc_5g_decode.argtypes = [
        ndpointer(dtype=np.int16, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
        ctypes.POINTER(LdpcInConfig),
        ctypes.POINTER(DecoderConfig),
    ]
    lib.c_ldpc_5g_decode.restype = ctypes.c_int32
    return lib


def make_c_in_config(model: CTableNeuralLayeredMinSumDecoder) -> LdpcInConfig:
    """Build the C LDPC input config from a trained PyTorch model config."""
    cfg = model.config
    return LdpcInConfig(
        bgn=cfg.bgn,
        A=cfg.A,
        crctype=cfg.crctype,
        E=cfg.E,
        rv=cfg.rv,
        Qm=cfg.Qm,
        HARQ_on=0,
    )


def make_c_decoder_config(
    algo: int,
    max_iterations: int,
    alpha: int = 22936,
    beta: int = 0,
    filling_bit_value: int = 1280,
    max_llr: int = 32767,
    min_llr: int = -32768,
) -> DecoderConfig:
    """Build a C decoder config for c_ldpc_5g_decode().

    algo=1 dispatches to layered_minsum_ldpc_decode().
    algo=3 dispatches to layered_degree_minsum_ldpc_decode().
    """
    return DecoderConfig(
        algo=algo,
        max_iterations=max_iterations,
        alpha=alpha,
        beta=beta,
        filling_bit_value=filling_bit_value,
        max_llr=max_llr,
        min_llr=min_llr,
    )


def _make_one_c_channel_sample(
    lib: ctypes.CDLL,
    c_config: LdpcInConfig,
    noise_power_db: float,
):
    """Generate one encoded C channel sample and return target bits and int16 LLRs."""
    info_bits = np.random.randint(0, 2, c_config.A).astype(np.uint8)
    target_bits = crc.nr_crc_encode(
        info_bits,
        CRC_TYPE_TO_POLY[c_config.crctype],
        mask=0,
    ).astype(np.uint8)
    info_bytes = np.packbits(info_bits, bitorder="big")

    encoded_bytes = np.zeros((c_config.E + 7) // 8, dtype=np.uint8)
    status = lib.c_ldpc_5g_encode(info_bytes, encoded_bytes, ctypes.byref(c_config))
    if status != 0:
        raise RuntimeError(f"c_ldpc_5g_encode failed with status {status}")

    llr_int16 = np.zeros(c_config.E, dtype=np.int16)
    status = lib.tx_rx_process(
        encoded_bytes,
        c_config.E,
        c_config.Qm,
        ctypes.c_float(noise_power_db),
        llr_int16,
    )
    if status != 0:
        raise RuntimeError(f"tx_rx_process failed with status {status}")

    return target_bits, llr_int16


def _model_input_from_c_llr(
    model: CTableNeuralLayeredMinSumDecoder,
    llr_int16: np.ndarray,
) -> np.ndarray:
    """Convert C rate-matched int16 LLRs to the PyTorch decoder LQ layout."""
    cfg = model.config
    channel_llr = llr_int16.astype(np.float32) / 128.0
    recovered_llr = nr_ldpc_raterecover.raterecover_ldpc(
        channel_llr,
        Ncb=cfg.N,
        N=cfg.N,
        k0=cfg.k0,
        Qm=cfg.Qm,
        Zc=cfg.Zc,
        K_apo=cfg.B,
        K=cfg.K,
    )
    return np.concatenate(
        (np.zeros(2 * cfg.Zc, dtype=recovered_llr.dtype), recovered_llr)
    ).astype(np.float32)


def _decode_c_baseline(
    lib: ctypes.CDLL,
    llr_int16: np.ndarray,
    c_config: LdpcInConfig,
    decoder_config: DecoderConfig,
    B: int,
) -> np.ndarray:
    """Run one C decoder baseline and return unpacked decoded bits."""
    hardbits = np.zeros((B + 7) // 8, dtype=np.uint8)
    status = lib.c_ldpc_5g_decode(
        llr_int16.copy(),
        hardbits,
        ctypes.byref(c_config),
        ctypes.byref(decoder_config),
    )
    decoded_bits = np.unpackbits(hardbits, bitorder="big")[:B]
    if status < 0:
        raise RuntimeError(f"c_ldpc_5g_decode failed with status {status}")
    return decoded_bits.astype(np.int64)


def _empty_totals(names):
    """Create metric accumulators for a list of decoder names."""
    return {
        name: {
            "total_bits": 0,
            "bit_errors": 0,
            "block_errors": 0,
            "tested_blocks": 0,
        }
        for name in names
    }


def _update_totals(total, decoded_bits, target_bits):
    """Update BER/BLER accumulators for one decoded block."""
    errors = decoded_bits != target_bits
    total["bit_errors"] += int(np.count_nonzero(errors))
    total["block_errors"] += int(np.any(errors))
    total["total_bits"] += int(target_bits.size)
    total["tested_blocks"] += 1


def _finalize_totals(totals: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """Convert integer accumulators to BER/BLER result dictionaries."""
    results = {}
    for name, total in totals.items():
        total_bits = total["total_bits"]
        tested = total["tested_blocks"]
        ber = total["bit_errors"] / total_bits if total_bits else 0.0
        bler = total["block_errors"] / tested if tested else 0.0
        results[name] = {
            "num_test_blocks": float(tested),
            "total_bits": float(total_bits),
            "bit_errors": float(total["bit_errors"]),
            "block_errors": float(total["block_errors"]),
            "BER": ber,
            "BLER": bler,
        }
    return results


def evaluate_trained_model_with_c_baselines(
    model: CTableNeuralLayeredMinSumDecoder,
    noise_power_db: float,
    num_test_blocks: int = 100,
    batch_size: int = 8,
    lib_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    minsum_alpha: int = 22936,
    minsum_beta: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a trained PyTorch decoder against C fixed-point baselines.

    The C baselines are:
      - ``c_layered_minsum``: algo=1 -> layered_minsum_ldpc_decode().
      - ``c_layered_degree_minsum``: algo=3 -> layered_degree_minsum_ldpc_decode().

    Each generated C channel sample is reused by all decoders. The PyTorch model
    receives dequantized/rate-recovered LLRs derived from the same int16 C LLRs.
    """
    if num_test_blocks <= 0:
        raise ValueError("num_test_blocks must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    lib = load_c_ldpc_library(lib_path)
    c_config = make_c_in_config(model)
    device = device or infer_model_device(model)
    model.to(device)
    model.eval()

    c_decoders = {
        "c_layered_minsum": make_c_decoder_config(
            algo=1,
            max_iterations=model.num_iterations,
            alpha=minsum_alpha,
            beta=minsum_beta,
        ),
        "c_layered_degree_minsum": make_c_decoder_config(
            algo=3,
            max_iterations=model.num_iterations,
            beta=minsum_beta,
        ),
    }
    totals = _empty_totals(["trained_alpha", *c_decoders.keys()])

    tested_blocks = 0
    with torch.no_grad():
        while tested_blocks < num_test_blocks:
            current_batch_size = min(batch_size, num_test_blocks - tested_blocks)
            targets = []
            model_inputs = []
            c_llrs = []

            for _ in range(current_batch_size):
                target_bits, llr_int16 = _make_one_c_channel_sample(
                    lib,
                    c_config,
                    noise_power_db,
                )
                targets.append(target_bits.astype(np.int64))
                c_llrs.append(llr_int16)
                model_inputs.append(_model_input_from_c_llr(model, llr_int16))

            llr_batch = torch.from_numpy(np.stack(model_inputs)).to(device)
            trained_bits = (
                decode_c_table_model_with_crc_early_stop(model, llr_batch)
                .detach()
                .cpu()
                .numpy()
            )

            for idx, target_bits in enumerate(targets):
                _update_totals(totals["trained_alpha"], trained_bits[idx], target_bits)
                for name, decoder_config in c_decoders.items():
                    decoded_bits = _decode_c_baseline(
                        lib,
                        c_llrs[idx],
                        c_config,
                        decoder_config,
                        model.config.B,
                    )
                    _update_totals(totals[name], decoded_bits, target_bits)

            tested_blocks += current_batch_size

    results = _finalize_totals(totals)
    if verbose:
        for name, result in results.items():
            print(
                f"{name} C-paired test result: "
                f"blocks={int(result['num_test_blocks'])} "
                f"bit_errors={int(result['bit_errors'])}/{int(result['total_bits'])} "
                f"block_errors={int(result['block_errors'])}/{int(result['num_test_blocks'])} "
                f"BER={result['BER']:.6e} BLER={result['BLER']:.6e}"
            )

    return results
