"""Utility helpers for neural LDPC training and reporting.

These helpers are intentionally small and dependency-light. They provide shared
CRC naming, alpha initialization, device detection, checkpoint loading, and
compact reporting used by the trainer and top-level workflow.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from deep_learning.deeplearning_ldpc_model import CTableNeuralLayeredMinSumDecoder


CRC_TYPE_TO_POLY = {
    0: "16",
    1: "24A",
    2: "24B",
    3: "24C",
}


def infer_model_device(model: torch.nn.Module) -> torch.device:
    """Return the device used by a model's first parameter or buffer."""
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device
    return torch.device("cpu")


def make_degree_seeded_row_alpha_init(table_config, num_iterations: int) -> np.ndarray:
    """Create a [iterations, HBG_row] alpha table from each HBG row degree."""
    row_values = []
    for row in range(table_config.num_rows):
        degree = table_config.row_offsets[row + 1] - table_config.row_offsets[row]
        if degree <= 4:
            row_values.append(0.8125)
        elif degree <= 8:
            row_values.append(0.75)
        else:
            row_values.append(0.6875)

    return np.tile(np.asarray(row_values, dtype=np.float32), (num_iterations, 1))


def alpha_init_from_degree(degree: int) -> float:
    """Initial alpha schedule keyed by exact HBG row degree."""
    if degree <= 4:
        return 0.8125
    if degree <= 8:
        return 0.75
    return 0.6875


def make_degree_alpha_init(table_config) -> Tuple[float, ...]:
    """Create alpha initial values ordered by sorted unique HBG row degree."""
    return tuple(alpha_init_from_degree(degree) for degree in get_unique_row_degrees(table_config))


def get_unique_row_degrees(table_config) -> Tuple[int, ...]:
    """Return sorted unique HBG row degrees for reporting and degree mode."""
    return tuple(
        sorted(
            {
                table_config.row_offsets[row + 1] - table_config.row_offsets[row]
                for row in range(table_config.num_rows)
            }
        )
    )


def load_trained_row_alpha_init(
    checkpoint_path: str,
    alpha_min: float,
    alpha_max: float,
    expected_shape: Tuple[int, int],
) -> np.ndarray:
    """Load a trained raw_alpha checkpoint and return the bounded alpha table.

    This helper is useful when a training run wants to seed a new model from an
    existing checkpoint instead of starting from scalar/degree rules.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if "raw_alpha" not in state_dict:
        raise KeyError(f"{checkpoint_path} does not contain state_dict['raw_alpha']")

    raw_alpha = state_dict["raw_alpha"].detach().cpu()
    if tuple(raw_alpha.shape) != expected_shape:
        raise ValueError(
            f"expected raw_alpha shape {expected_shape} from {checkpoint_path}, "
            f"got {tuple(raw_alpha.shape)}"
        )

    alpha = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(raw_alpha)
    return alpha.numpy().astype(np.float32)


def summarize_model_weights(model: CTableNeuralLayeredMinSumDecoder) -> Dict[str, Any]:
    """Return compact alpha/beta statistics for logging and result files."""
    alpha = model.alpha().detach().cpu().numpy()
    beta = model.beta().detach().cpu().numpy()
    summary = {
        "alpha_shape": tuple(alpha.shape),
        "alpha_stats": {
            "min": float(alpha.min()),
            "max": float(alpha.max()),
            "mean": float(alpha.mean()),
            "std": float(alpha.std()),
        },
        "beta_shape": tuple(beta.shape),
        "beta_stats": {
            "min": float(beta.min()),
            "max": float(beta.max()),
            "mean": float(beta.mean()),
            "std": float(beta.std()),
        },
    }
    if model.alpha_mode == "single":
        summary["trained_alpha_values"] = float(alpha.reshape(-1)[0])
    elif model.alpha_mode == "checkpoint":
        summary["trained_alpha_hbg_row_mean_values"] = tuple(
            float(value) for value in alpha.mean(axis=0)
        )
    elif model.alpha_mode == "degree":
        summary["trained_alpha_values"] = {
            int(degree): float(value)
            for degree, value in zip(model.unique_row_degrees, alpha.reshape(-1))
        }
    return summary
