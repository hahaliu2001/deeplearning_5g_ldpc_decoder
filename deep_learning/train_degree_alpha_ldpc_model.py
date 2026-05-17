"""Train/evaluate the LDPC model with one alpha per unique HBG row degree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from deep_learning.ldpc_training_workflow import run_ldpc_training_workflow


ALPHA_MODE = "degree"
OUTPUT_DIR = Path("deeplearning_out") / ALPHA_MODE
C_FLOAT_LLR_CLIP = 32767.0 / 128.0
INIT_ALPHA_BY_DEGREE = (
    0.8125,  # degree 1
    0.8125,  # degree 2
    0.8125,  # degree 3
    0.8125,  # degree 4
    0.75,  # degree 5
    0.75,  # degree 6
    0.75,  # degree 7
    0.75,  # degree 8
    0.6875,  # degree 9
    0.6875,  # degree 10
    0.6875,  # degree 11
    0.6875,  # degree 12
    0.6875,  # degree 13
    0.6875,  # degree 14
    0.6875,  # degree 15
    0.6875,  # degree 16
    0.6875,  # degree 17
    0.6875,  # degree 18
    0.6875,  # degree 19
)

TRAINING_CONFIG = {
    "bgn": 1,
    "A": 232 + 256,
    "crctype": 1,
    "E": 256 * 3 * 2,
    "rv": 0,
    "Qm": 1,
    "noise_power_db": 5,
    "HARQ_on": 0,
    "num_iterations": 32,
    "epochs": 100,
    "batch_size": 20,
    "lr": 1e-3,
    "init_alpha": 0.75,
    "init_beta": 0.0,
    "learnable_scaling": True,
    "learnable_beta": False,
    "alpha_mode": ALPHA_MODE,
    "alpha_min": 0.5,
    "alpha_max": 1.0 - 1e-6,
    "init_alpha_degree_values": INIT_ALPHA_BY_DEGREE,
    "training_noise_power_db_range": (4.3, 4.7),
    "eval_noise_power_db_list": (3.5, 4.0, 4.5),
    "eval_num_test_blocks": 800,
    "eval_batch_size": 4,
    "loss_type": "bce",
    "llr_margin": 5.0,
    "clip_llr": C_FLOAT_LLR_CLIP,
    "detach_check_update_inputs": True,
    "cpu_threads": 4,
    "intermediate_loss_weight": 0.1,
    "intermediate_loss_stride": 4,
    "model_save_path": str(OUTPUT_DIR / "trained_c_table_ldpc_decoder.pth"),
    "result_save_path": str(OUTPUT_DIR / "trained_c_table_ldpc_decoder_result.pth"),
    "lib_path": None,
    "c_minsum_alpha": 22936,
    "c_minsum_beta": 0,
    "device": None,
    "verbose": True,
}


if __name__ == "__main__":
    run_ldpc_training_workflow(**TRAINING_CONFIG)
