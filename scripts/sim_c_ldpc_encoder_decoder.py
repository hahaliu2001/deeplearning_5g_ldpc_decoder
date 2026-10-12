# -*- coding: utf-8 -*-
"""BLER simulation parameters for the C LDPC encoder/channel/decoder path."""

import numpy as np

from ldpc_c_sim_common import make_config_grid, run_simulation


RESULT_FILE = "out/c_ldpc_decoder_result_qm1_fixed_min_sum.pkl"

# LDPC test grid. make_config_grid() expands these lists into independent
# sweep points with tuple order: Qm, rv, B, coderate, bgn, crctype.
Qm_LIST = [1]
RV_LIST = [0]
CODERATE_LIST = [1/3]
CRCTYPE_LIST = [0]
# Keep B and BG paired; a Cartesian product would add B=3840/BG1.
BLOCK_CASES = [(3840, 2), (8448, 1)]


# Channel and Monte Carlo settings. Each SNR value runs TEST_NUM random blocks,
# so increasing either list length or TEST_NUM scales runtime linearly.
# noise_power_db = -snr_power_db - 10 * np.log10(ldpc_config["Qm"])
SNR_POWER_DB_LIST = np.arange(-5, 2, 0.2)
TEST_NUM = 2000


def get_decoder_config_list():
    """Return decoder configurations compared in this simulation."""
    return [
        # algo, max_iterations, alpha, beta, filling_bit_value, max_llr, min_llr
        [1, 12, 22936, 0, 1280, 32767, -32767],  # Fixed-point min-sum.
    ]


def main():
    # Build the reusable grid in Python, then hand the whole sweep to the
    # shared C-library simulation runner.
    config_grid = [
        config
        for B, bgn in BLOCK_CASES
        for config in make_config_grid(
            Qm_LIST, RV_LIST, [B], CODERATE_LIST, [bgn], CRCTYPE_LIST,
            strict=True,
        )
    ]

    run_simulation(
        config_grid=config_grid,
        decoder_config_list=get_decoder_config_list(),
        snr_power_db_list=SNR_POWER_DB_LIST,
        test_num=TEST_NUM,
        result_file=RESULT_FILE,
    )


if __name__ == "__main__":
    main()
    pass
