"""Top-level training/evaluation workflow for the neural LDPC decoder.

The heavy lifting lives in smaller modules:
  - ldpc_training_data.py builds channel/rate-recovered batches.
  - ldpc_model_trainer.py owns the optimization loop.
  - ldpc_c_baseline_evaluation.py compares the trained model with C decoders.
  - ldpc_training_utils.py holds alpha initialization and reporting helpers.

This module keeps the shared training/evaluation workflow used by the
mode-specific training entry points.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from deep_learning.ldpc_c_baseline_evaluation import evaluate_trained_model_with_c_baselines
from deep_learning.ldpc_model_evaluation import (
    evaluate_c_table_ldpc_model,
    evaluate_c_table_ldpc_models_paired,
)
from deep_learning.ldpc_model_trainer import train_c_table_ldpc_model
from deep_learning.ldpc_table_config import load_ldpc_table_config
from deep_learning.ldpc_training_data import make_c_table_training_batch
from deep_learning.ldpc_training_utils import (
    CRC_TYPE_TO_POLY,
    alpha_init_from_degree,
    get_unique_row_degrees,
    infer_model_device,
    load_trained_row_alpha_init,
    make_degree_alpha_init,
    make_degree_seeded_row_alpha_init,
    summarize_model_weights,
)

DEFAULT_OUTPUT_DIR = Path("deeplearning_out")


def run_ldpc_training_workflow(
    bgn: int,
    A: int,
    crctype: int,
    E: int,
    rv: int,
    Qm: int,
    noise_power_db: float,
    HARQ_on: int = 0,
    num_iterations: int = 32,
    epochs: int = 100,
    batch_size: int = 20,
    lr: float = 1e-3,
    init_alpha: Union[float, Sequence[Sequence[float]], torch.Tensor] = 0.75,
    init_beta: float = 0.0,
    learnable_scaling: bool = True,
    learnable_beta: bool = False,
    alpha_mode: str = "degree",
    alpha_min: float = 0.5,
    alpha_max: float = 1.0 - 1e-6,
    init_alpha_degree_values: Optional[Sequence[float]] = None,
    training_noise_power_db_range: Optional[Tuple[float, float]] = (5.0, 6.0),
    eval_noise_power_db_list: Sequence[float] = (4.5, 5.0, 5.5),
    eval_num_test_blocks: int = 800,
    eval_batch_size: int = 4,
    loss_type: str = "bce",
    llr_margin: float = 5.0,
    clip_llr: Optional[float] = None,
    detach_check_update_inputs: bool = False,
    cpu_threads: Optional[int] = 4,
    intermediate_loss_weight: float = 0.25,
    intermediate_loss_stride: int = 4,
    model_save_path: Optional[str] = str(DEFAULT_OUTPUT_DIR / "trained_c_table_ldpc_decoder_bg1.pth"),
    result_save_path: Optional[str] = str(DEFAULT_OUTPUT_DIR / "trained_c_table_ldpc_decoder_result.pth"),
    lib_path: Optional[str] = None,
    c_minsum_alpha: int = 22936,
    c_minsum_beta: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a model, evaluate it against C decoder baselines, and save results.

    This is the main function intended for simulation scripts: it creates the
    LDPC table config, trains the model, runs paired evaluation against the C
    ``layered_minsum_ldpc_decode`` and ``layered_degree_minsum_ldpc_decode``
    baselines, saves optional artifacts, and returns a compact result
    dictionary.

    Inputs:
        bgn: LDPC base graph number, 1 or 2.
        A: information-bit length before transport-block CRC.
        crctype: numeric CRC type used by the C/Python LDPC path:
            0=CRC16, 1=CRC24A, 2=CRC24B, 3=CRC24C.
        E: rate-matched codeword length in bits.
        rv: redundancy version, 0, 1, 2, or 3.
        Qm: modulation order in bits per symbol.
        noise_power_db: default AWGN noise power used for training batches.
        HARQ_on: HARQ combining flag passed to the LDPC table config.
        num_iterations: number of unrolled layered decoder iterations.
        epochs: number of optimizer epochs.
        batch_size: training batch size.
        lr: Adam optimizer learning rate.
        init_alpha/init_beta: initial normalized min-sum alpha and offset beta.
        learnable_scaling/learnable_beta: select whether alpha and beta are
            trained or kept fixed.
        alpha_mode: alpha sharing mode, one of "single", "checkpoint", or
            "degree".
        alpha_min/alpha_max: bounds applied to trainable alpha after sigmoid.
        init_alpha_degree_values: optional degree-mode alpha values. Provide
            either values ordered by sorted unique HBG row degree, or a
            1-based exact-degree table where degree d reads entry d - 1. If
            omitted in degree mode, defaults are generated from
            make_degree_alpha_init().
        training_noise_power_db_range: optional (low, high) range used to draw
            a random training noise power per batch.
        eval_noise_power_db_list: noise powers used for post-training BER/BLER
            evaluation.
        eval_num_test_blocks/eval_batch_size: evaluation workload controls.
        loss_type: "bce" or "margin".
        llr_margin: margin value used when loss_type is "margin".
        clip_llr: optional LLR clip applied inside the neural decoder.
        detach_check_update_inputs: truncate backpropagation through previous
            layered decoder states while still training current alpha/beta
            updates.
        cpu_threads: optional torch CPU thread limit.
        intermediate_loss_weight: optional supervision weight for intermediate
            iteration LLRs.
        intermediate_loss_stride: interval for intermediate-iteration losses.
        model_save_path: optional path for the best trained model checkpoint.
        result_save_path: optional path for the training/evaluation result file.
        lib_path: optional path to libnrLDPC.so.
        c_minsum_alpha/c_minsum_beta: fixed-point alpha and beta passed to the
            C layered min-sum baseline. The degree min-sum baseline uses its
            built-in degree alpha table and the same beta value.
        device: optional torch device. If omitted, CUDA is used when available.
        verbose: print progress and result summaries when true.

    Returns:
        A dictionary containing the LDPC config, training loss, alpha/beta
        summary, C-paired evaluation metrics, and output paths.
    """
    table_config = load_ldpc_table_config(
        bgn=bgn,
        A=A,
        crctype=crctype,
        E=E,
        rv=rv,
        Qm=Qm,
        HARQ_on=HARQ_on,
        lib_path=lib_path,
    )
    unique_degrees = get_unique_row_degrees(table_config)
    # Degree mode needs one initial alpha per unique row degree. If the caller
    # does not provide the list, use the local degree-based seed rule.
    if alpha_mode == "degree" and init_alpha_degree_values is None:
        init_alpha_degree_values = make_degree_alpha_init(table_config)

    for output_path in (model_save_path, result_save_path):
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if verbose and init_alpha_degree_values is not None:
        print(
            "degree init_alpha",
            {
                "unique_degrees": unique_degrees,
                "values": tuple(init_alpha_degree_values),
            },
        )

    trained_model, trained_state, trained_loss = train_c_table_ldpc_model(
        bgn=bgn,
        A=A,
        crctype=crctype,
        E=E,
        rv=rv,
        Qm=Qm,
        noise_power_db=noise_power_db,
        HARQ_on=HARQ_on,
        num_iterations=num_iterations,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        init_alpha=init_alpha,
        init_beta=init_beta,
        learnable_scaling=learnable_scaling,
        learnable_beta=learnable_beta,
        alpha_mode=alpha_mode,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        init_alpha_degree_values=init_alpha_degree_values,
        training_noise_power_db_range=training_noise_power_db_range,
        loss_type=loss_type,
        llr_margin=llr_margin,
        clip_llr=clip_llr,
        detach_check_update_inputs=detach_check_update_inputs,
        cpu_threads=cpu_threads,
        intermediate_loss_weight=intermediate_loss_weight,
        intermediate_loss_stride=intermediate_loss_stride,
        save_path=model_save_path,
        lib_path=lib_path,
        device=device,
        verbose=verbose,
    )
    if verbose:
        print("best loss", trained_loss)

    c_eval_results = {}
    for eval_noise_power_db in eval_noise_power_db_list:
        if verbose:
            print(f"C-paired evaluation noise_power_db={eval_noise_power_db:.1f}")
        c_eval_results[float(eval_noise_power_db)] = evaluate_trained_model_with_c_baselines(
            trained_model,
            noise_power_db=eval_noise_power_db,
            num_test_blocks=eval_num_test_blocks,
            batch_size=eval_batch_size,
            lib_path=lib_path,
            device=device,
            verbose=verbose,
            minsum_alpha=c_minsum_alpha,
            minsum_beta=c_minsum_beta,
        )

    weight_summary = summarize_model_weights(trained_model)
    # Store only compact metadata and metrics in the returned result. The full
    # trained weights are saved separately in model_save_path/result_save_path.
    output = {
        "ldpc_config": {
            "bgn": bgn,
            "A": A,
            "B": table_config.B,
            "crctype": crctype,
            "E": E,
            "rv": rv,
            "Qm": Qm,
            "HARQ_on": HARQ_on,
            "Zc": table_config.Zc,
            "num_rows": table_config.num_rows,
            "num_edges": table_config.num_edges,
            "unique_row_degrees": unique_degrees,
        },
        "training": {
            "best_loss": float(trained_loss),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "noise_power_db": noise_power_db,
            "training_noise_power_db_range": training_noise_power_db_range,
            "loss_type": loss_type,
            "intermediate_loss_weight": intermediate_loss_weight,
            "detach_check_update_inputs": detach_check_update_inputs,
        },
        "model": {
            "alpha_mode": alpha_mode,
            "alpha_min": trained_model.alpha_min,
            "alpha_max": trained_model.alpha_max,
            "alpha_parameter_shape": trained_model.alpha_parameter_shape(),
            **weight_summary,
        },
        "evaluation": c_eval_results,
        "paths": {
            "model_save_path": model_save_path,
            "result_save_path": result_save_path,
        },
    }

    if result_save_path is not None:
        torch.save(
            {
                "result": output,
                "trained_state": trained_state,
                "table_config": table_config,
            },
            result_save_path,
        )

    if verbose:
        print("training/evaluation output", output)

    return output


__all__ = [
    "CRC_TYPE_TO_POLY",
    "alpha_init_from_degree",
    "evaluate_c_table_ldpc_model",
    "evaluate_c_table_ldpc_models_paired",
    "evaluate_trained_model_with_c_baselines",
    "get_unique_row_degrees",
    "infer_model_device",
    "load_trained_row_alpha_init",
    "make_c_table_training_batch",
    "make_degree_alpha_init",
    "make_degree_seeded_row_alpha_init",
    "summarize_model_weights",
    "train_c_table_ldpc_model",
    "run_ldpc_training_workflow",
]


if __name__ == "__main__":
    run_ldpc_training_workflow(
        bgn=2,
        A=232 + 256,
        crctype=1,
        E=256 * 4 * 2,
        rv=0,
        Qm=1,
        noise_power_db=5,
    )
