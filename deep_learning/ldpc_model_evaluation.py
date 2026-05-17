"""BER/BLER evaluation helpers for table-driven neural LDPC models."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from deep_learning.deeplearning_ldpc_model import CTableNeuralLayeredMinSumDecoder
from deep_learning.ldpc_training_data import make_c_table_training_batch
from deep_learning.ldpc_training_utils import CRC_TYPE_TO_POLY, infer_model_device
from py5gphy.crc import crc


def decode_c_table_model_with_crc_early_stop(
    model: CTableNeuralLayeredMinSumDecoder,
    llr_batch: torch.Tensor,
) -> torch.Tensor:
    """Return decoded bits from the first iteration that passes CRC per block.

    The model still computes every configured iteration, but evaluation selects
    the earliest CRC-passing hard decision for each block, matching the output
    semantics of the C decoders' early termination.
    """
    _, final_llr, iteration_llrs = model(
        llr_batch,
        return_llr=True,
        return_iteration_llrs=True,
    )
    cfg = model.config
    crcpoly = CRC_TYPE_TO_POLY[cfg.crctype]
    final_bits = (final_llr[:, : cfg.B] < 0).to(torch.int64)
    selected_bits = final_bits.detach().cpu().numpy()
    selected = np.zeros(selected_bits.shape[0], dtype=bool)

    for iteration_llr in iteration_llrs:
        iteration_bits = (
            iteration_llr[:, : cfg.B] < 0
        ).detach().cpu().numpy().astype(np.int8)
        for block_idx in range(iteration_bits.shape[0]):
            if selected[block_idx]:
                continue
            if crc.nr_crc_decode(iteration_bits[block_idx], crcpoly, mask=0)[1] == 0:
                selected_bits[block_idx] = iteration_bits[block_idx]
                selected[block_idx] = True
        if selected.all():
            break

    return torch.from_numpy(selected_bits).to(device=llr_batch.device, dtype=torch.int64)


def evaluate_c_table_ldpc_model(
    model: CTableNeuralLayeredMinSumDecoder,
    noise_power_db: float,
    num_test_blocks: int = 100,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Generate random test blocks and evaluate one decoder without training."""
    if num_test_blocks <= 0:
        raise ValueError("num_test_blocks must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = device or infer_model_device(model)
    model.to(device)
    model.eval()

    cfg = model.config
    total_bits = 0
    bit_errors = 0
    block_errors = 0
    tested_blocks = 0

    with torch.no_grad():
        while tested_blocks < num_test_blocks:
            current_batch_size = min(batch_size, num_test_blocks - tested_blocks)
            llr_batch, target_batch = make_c_table_training_batch(
                model=model,
                batch_size=current_batch_size,
                noise_power_db=noise_power_db,
                device=device,
            )
            decoded_bits = decode_c_table_model_with_crc_early_stop(model, llr_batch)
            target_bits = target_batch.to(torch.int64)
            errors = decoded_bits.ne(target_bits)

            # BER counts all bit mismatches. BLER counts one error when any bit
            # in a decoded block is wrong.
            bit_errors += int(errors.sum().item())
            block_errors += int(errors.any(dim=1).sum().item())
            total_bits += int(target_bits.numel())
            tested_blocks += current_batch_size

    ber = bit_errors / total_bits if total_bits else 0.0
    bler = block_errors / tested_blocks if tested_blocks else 0.0
    results = {
        "num_test_blocks": float(tested_blocks),
        "total_bits": float(total_bits),
        "bit_errors": float(bit_errors),
        "block_errors": float(block_errors),
        "BER": ber,
        "BLER": bler,
    }

    if verbose:
        print(
            "test result: "
            f"blocks={tested_blocks} bit_errors={bit_errors}/{total_bits} "
            f"block_errors={block_errors}/{tested_blocks} "
            f"BER={ber:.6e} BLER={bler:.6e}"
        )

    return results


def evaluate_c_table_ldpc_models_paired(
    models: Dict[str, CTableNeuralLayeredMinSumDecoder],
    noise_power_db: float,
    num_test_blocks: int = 100,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple decoders on identical generated LDPC test blocks.

    The same received LLR batch is reused for every model, which makes paired
    comparisons much less noisy than evaluating each model on separate random
    channels.
    """
    if not models:
        raise ValueError("models must not be empty")
    if num_test_blocks <= 0:
        raise ValueError("num_test_blocks must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    first_model = next(iter(models.values()))
    device = device or infer_model_device(first_model)
    cfg = first_model.config
    for model in models.values():
        if model.config != cfg:
            raise ValueError("all models must use the same LDPC table config")
        model.to(device)
        model.eval()

    totals = {
        name: {
            "total_bits": 0,
            "bit_errors": 0,
            "block_errors": 0,
            "tested_blocks": 0,
        }
        for name in models
    }

    with torch.no_grad():
        tested_blocks = 0
        while tested_blocks < num_test_blocks:
            current_batch_size = min(batch_size, num_test_blocks - tested_blocks)
            llr_batch, target_batch = make_c_table_training_batch(
                model=first_model,
                batch_size=current_batch_size,
                noise_power_db=noise_power_db,
                device=device,
            )
            target_bits = target_batch.to(torch.int64)

            for name, model in models.items():
                # Every model sees the same llr_batch produced by first_model's
                # LDPC configuration. All configs were checked equal above.
                decoded_bits = decode_c_table_model_with_crc_early_stop(model, llr_batch)
                errors = decoded_bits.ne(target_bits)
                totals[name]["bit_errors"] += int(errors.sum().item())
                totals[name]["block_errors"] += int(errors.any(dim=1).sum().item())
                totals[name]["total_bits"] += int(target_bits.numel())
                totals[name]["tested_blocks"] += current_batch_size

            tested_blocks += current_batch_size

    results: Dict[str, Dict[str, float]] = {}
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

        if verbose:
            print(
                f"{name} paired test result: "
                f"blocks={tested} bit_errors={total['bit_errors']}/{total_bits} "
                f"block_errors={total['block_errors']}/{tested} "
                f"BER={ber:.6e} BLER={bler:.6e}"
            )

    return results
