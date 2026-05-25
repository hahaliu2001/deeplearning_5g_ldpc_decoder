# -*- coding: utf-8 -*-
"""BLER simulation parameters for the CUDA LDPC encoder/decoder path."""

import numpy as np

from ldpc_c_sim_common import (
    CUDA_LIB_PATH,
    LIB_PATH,
    make_config_grid,
    run_simulation,
)


RESULT_FILE = "out/cuda_ldpc_decoder_result_bg1.pkl"

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

# Channel and Monte Carlo settings. Small TEST_NUM is useful while validating
# CUDA kernels; increase it for smoother BLER/BER estimates.
# noise_power_db = -snr_power_db - 10 * np.log10(ldpc_config["Qm"])
SNR_POWER_DB_LIST = np.arange(-5, 1.5, 0.2)
TEST_NUM = 2000


def get_decoder_config_list():
    """Return CUDA decoder configurations compared in this simulation."""
    return [
        # algo, max_iterations, alpha, beta, filling_bit_value, max_llr, min_llr
        [4, 32, 22936, 0, 1280, 32767, -32767],  # CUDA floating-point BP.
        [1, 32, 22936, 0, 1280, 32767, -32767],  # CUDA fixed min-sum.
        
    ]


def main():
    # CUDA provides the encoder/decoder symbols, while tx_rx_process still
    # comes from the C shared library through channel_lib_path below.
    config_grid = make_config_grid(
        Qm_LIST, RV_LIST, B_LIST, CODERATE_LIST, BGN_LIST, CRCTYPE_LIST
    )

    run_simulation(
        config_grid=config_grid,
        decoder_config_list=get_decoder_config_list(),
        snr_power_db_list=SNR_POWER_DB_LIST,
        test_num=TEST_NUM,
        result_file=RESULT_FILE,
        lib_path=CUDA_LIB_PATH,
        encode_symbol="cuda_ldpc_5g_encode",
        decode_symbol="cuda_ldpc_5g_decode",
        channel_lib_path=LIB_PATH,
    )


if __name__ == "__main__":
    main()
