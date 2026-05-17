# -*- coding: utf-8 -*-
"""BLER simulation parameters for the C LDPC encoder/channel/decoder path."""

import numpy as np

from ldpc_c_sim_common import make_config_grid, run_simulation


RESULT_FILE = "out/c_ldpc_decoder_result_bg1.pkl"

# LDPC test grid.
Qm_LIST = [1]
RV_LIST = [0]
B_LIST = [256,512]
CODERATE_LIST = [1 / 3]
BGN_LIST = [1]
CRCTYPE_LIST = [1]

# Channel and Monte Carlo settings.
SNR_POWER_DB_LIST = np.arange(-5, -2, 0.2)
TEST_NUM = 400


def get_decoder_config_list():
    """Return decoder configurations compared in this simulation."""
    return [
        # algo, max_iterations, alpha, beta, filling_bit_value, max_llr, min_llr
        [0, 32, 1, 0, 1280, 20, -32768],         # Floating-point BP.
        [1, 32, 22936, 0, 1280, 32767, -32767],  # Min-sum, alpha ~= 0.7.
        [2, 32, 22936, 0, 1280, 32767, -32767],  # Min-sum, alpha ~= 0.7.
        [3, 32, 22936, 0, 1280, 32767, -32767],  # Min-sum, alpha ~= 0.7.
        [4, 32, 22936, 0, 1280, 32767, -32767],  # Min-sum, alpha ~= 0.7.
        
    ]


def main():
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
