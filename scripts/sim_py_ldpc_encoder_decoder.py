# -*- coding:utf-8 -*-
"""BLER simulation for the pure Python LDPC encoder/channel/decoder path.

This script mirrors the configuration-grid style used by the C simulation,
but every block is encoded, transmitted, and decoded by the Python modules.
"""

from pathlib import Path
import pickle
import sys

import numpy as np

if __name__ == "__main__":
    # When this file is run directly from scripts/, add the repository root so
    # absolute package imports such as py5gphy.ldpc_5g_encoder still resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldpc_c_sim_common import format_coderate, make_config_grid
from py5gphy.ldpc_5g_decoder import cb_ldpc_5g_decode
from py5gphy.ldpc_5g_encoder import cb_ldpc_5g_encode
from py5gphy.ldpc_utils import tx_rx_process


RESULT_FILE = Path("out/py_ldpc_decoder_result.pkl")

# LDPC test grid. make_config_grid() expands these lists into independent
# sweep points with tuple order: Qm, rv, B, coderate, bgn, crctype.
Qm_LIST = [2]
RV_LIST = [0]
B_LIST = [640]
CODERATE_LIST = [1 / 3]
BGN_LIST = [1]
CRCTYPE_LIST = [1]

# Channel sweep and Monte Carlo settings. The pure-Python path is useful as a
# reference implementation, but each extra SNR or trial can be noticeably slow.
# noise_power_db = -snr_power_db - 10 * np.log10(ldpc_config["Qm"])
SNR_POWER_DB_LIST = np.arange(-2, -1, 0.5) 
TEST_NUM = 100

# Keep the same numeric CRC IDs as the C simulation and map them to the Python
# CRC names expected by cb_ldpc_5g_encode() and cb_ldpc_5g_decode().
CRC_POLY_BY_TYPE = {
    0: "16",
    1: "24A",
    2: "24B",
    3: "24C",
}


def get_decoder_config_list():
    """Return the Python decoder configurations compared in each BLER sweep."""
    return [
        {"L": 32, "algo": "BP", "alpha": 0.7, "beta": 0},
        {"L": 32, "algo": "min-sum", "alpha": 0.7, "beta": 0},
        {"L": 32, "algo": "layered-BP", "alpha": 0.7, "beta": 0},
        {"L": 32, "algo": "layered-min-sum", "alpha": 0.7, "beta": 0},
    ]


def make_ldpc_config(Qm, rv, B, coderate, bgn, crctype):
    """Build one pure-Python LDPC simulation config from a grid tuple.

    B follows the same convention as the C simulation: it is the information
    block length after CRC attachment. The encoder needs A, the information
    length before CRC, so A is derived from B and the selected CRC type.
    """
    if crctype not in CRC_POLY_BY_TYPE:
        raise ValueError(f"unsupported CRC type {crctype}")

    # Keep the same A/B/E derivation as the C simulation so Python, C, and CUDA
    # sweeps can be compared from the saved pickle files.
    crc_len = 16 if crctype == 0 else 24
    return {
        "Qm": Qm,
        "rv": rv,
        "B": B,
        "A": B - crc_len,
        "E": int(B / coderate),
        "coderate": coderate,
        "bgn": bgn,
        "crctype": crctype,
        "crcpoly": CRC_POLY_BY_TYPE[crctype],
    }


def run_ldpc_trial(ldpc_config, decoder_config_list, noise_power_db):
    """Run one random transport block through encode, channel, and decode.

    Returns block-error flags and information-bit error counts for each decoder.
    A block-error flag of 1 means decoding failed, and 0 means success.
    """
    # Generate a new payload for each Monte Carlo trial; all decoder configs
    # below decode the same received LLRs for a fair comparison.
    info_bits = np.random.randint(0, 2, size=ldpc_config["A"])
    codeword = cb_ldpc_5g_encode(
        info_bits=info_bits,
        E=ldpc_config["E"],
        Qm=ldpc_config["Qm"],
        rv=ldpc_config["rv"],
        crcpoly=ldpc_config["crcpoly"],
        bgn=ldpc_config["bgn"],
    )

    llr = tx_rx_process(
        codeword,
        Qm=ldpc_config["Qm"],
        noise_power_db=noise_power_db,
    )

    error_flags = []
    bit_errors = []
    for decoder_config in decoder_config_list:
        # Reuse the same received LLRs so each decoder configuration sees an
        # identical channel realization for this trial.
        decoded_bits, status, _ = cb_ldpc_5g_decode(
            llr,
            B=ldpc_config["B"],
            Qm=ldpc_config["Qm"],
            rv=ldpc_config["rv"],
            crcpoly=ldpc_config["crcpoly"],
            bgn=ldpc_config["bgn"],
            ldpcdecoder_config=decoder_config,
        )
        decoded_info_bits = decoded_bits[:ldpc_config["A"]]
        error_flags.append(0 if status else 1)
        bit_errors.append(np.count_nonzero(decoded_info_bits != info_bits))

    return (
        np.array(error_flags, dtype=np.float64),
        np.array(bit_errors, dtype=np.float64),
    )


def run_bler_sweep(ldpc_config, decoder_config_list, snr_power_db_list, test_num):
    """Run all SNR points for one LDPC configuration and compute BLER/BER."""
    bler_list = []
    ber_list = []

    for snr_power_db in snr_power_db_list:
        error_count = np.zeros(len(decoder_config_list), dtype=np.float64)
        bit_error_count = np.zeros(len(decoder_config_list), dtype=np.float64)
        # tx_rx_process() takes noise power in dB. Convert from SNR using the
        # same convention as the C simulation helper.
        noise_power_db = -snr_power_db - 10 * np.log10(ldpc_config["Qm"])

        for _ in range(test_num):
            block_errors, bit_errors = run_ldpc_trial(
                ldpc_config, decoder_config_list, noise_power_db
            )
            error_count += block_errors
            bit_error_count += bit_errors

        bler = error_count / test_num
        ber = bit_error_count / (test_num * ldpc_config["A"])
        print(
            f"Qm {ldpc_config['Qm']}, rv {ldpc_config['rv']}, "
            f"A {ldpc_config['A']}, E {ldpc_config['E']}, "
            f"coderate {format_coderate(ldpc_config['coderate'])}, "
            f"bgn {ldpc_config['bgn']}, crc {ldpc_config['crctype']}, "
            f"snr_power_db {snr_power_db:.2f}, BLER={bler}, BER={ber}"
        )
        bler_list.append(bler)
        ber_list.append(ber)

    return bler_list, ber_list


def run_simulation(
    config_grid, decoder_config_list, snr_power_db_list, test_num, result_file
):
    """Run the full Python BLER simulation and save the result table."""
    print(f"Generated {len(config_grid)} configurations.")
    test_result = []

    for config_tuple in config_grid:
        Qm, rv, B, coderate, bgn, crctype = config_tuple
        # This project supports a single code block. BG2 is limited to B <= 3840
        # after CRC, matching the encoder and decoder assumptions.
        if bgn == 2 and B > 3840:
            continue

        ldpc_config = make_ldpc_config(Qm, rv, B, coderate, bgn, crctype)
        bler_list, ber_list = run_bler_sweep(
            ldpc_config, decoder_config_list, snr_power_db_list, test_num
        )
        test_result.append([config_tuple, bler_list, ber_list])

    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("wb") as f:
        # Keep the same top-level metadata as the C simulation. Each Python
        # result row stores [config_tuple, bler_list, ber_list].
        pickle.dump(
            (test_result, decoder_config_list, snr_power_db_list, test_num), f
        )

    return test_result


def main():
    # Build the grid with the same helper used by the C/CUDA scripts so all
    # three entry points reject unsupported combinations consistently.
    config_grid = make_config_grid(
        Qm_LIST, RV_LIST, B_LIST, CODERATE_LIST, BGN_LIST, CRCTYPE_LIST
    )

    run_simulation(
        config_grid=config_grid,
        decoder_config_list=get_decoder_config_list(),
        snr_power_db_list=SNR_POWER_DB_LIST,
        test_num=TEST_NUM,
        result_file=RESULT_FILE,
    )
    print(f"Saved results to {RESULT_FILE}")


if __name__ == "__main__":
    main()
