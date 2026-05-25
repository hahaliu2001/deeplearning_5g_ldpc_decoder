# -*- coding: utf-8 -*-
"""BLER simulation parameters for the C LDPC encoder/channel/decoder path."""

import numpy as np

from ldpc_c_sim_common import make_config_grid, run_simulation


RESULT_FILE = "out/c_ldpc_decoder_result_bg1.pkl"

# LDPC test grid. make_config_grid() expands these lists into independent
# sweep points with tuple order: Qm, rv, B, coderate, bgn, crctype.
if 0:
    Qm_LIST = [1,2,4,6,8]
    RV_LIST = [0,1,2,3]
    B_LIST = [128,256,512,1024,2046,3840,4096,6144,8448]
    CODERATE_LIST = [1/5,1/4,1/2,2/3,3/4]
    BGN_LIST = [1,2]
    CRCTYPE_LIST = [0,1,2,3]

if 1:
    Qm_LIST = [1,2,4,6,8]
    RV_LIST = [0]
    B_LIST = [3840]
    CODERATE_LIST = [1/3]
    BGN_LIST = [1]
    CRCTYPE_LIST = [0]


# Channel and Monte Carlo settings. Each SNR value runs TEST_NUM random blocks,
# so increasing either list length or TEST_NUM scales runtime linearly.
# noise_power_db = -snr_power_db - 10 * np.log10(ldpc_config["Qm"])
SNR_POWER_DB_LIST = np.arange(-5, 1.5, 0.2)
TEST_NUM = 2000


def get_decoder_config_list():
    """Return decoder configurations compared in this simulation."""
    return [
        # algo, max_iterations, alpha, beta, filling_bit_value, max_llr, min_llr
        [0, 32, 1, 0, 1280, 20, -32768],         # Floating-point BP.
        [1, 32, 22936, 0, 1280, 32767, -32767],  # Fixed-point min-sum.
        [2, 32, 22936, 0, 1280, 32767, -32767],  # Trained-table min-sum.
        [3, 32, 22936, 0, 1280, 32767, -32767],  # Degree-dependent min-sum.
        [4, 32, 22936, 0, 1280, 32767, -32767],  # Floating-point min-sum.
        
    ]


def main():
    # Build the reusable grid in Python, then hand the whole sweep to the
    # shared C-library simulation runner.
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


if __name__ == "__main__":
    main()
    pass
