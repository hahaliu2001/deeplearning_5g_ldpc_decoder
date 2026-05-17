"""Core training loop for table-driven neural LDPC models.

This module only trains one model. It does not run post-training evaluation or
format result dictionaries; that orchestration is handled by
``ldpc_training_workflow.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from deep_learning.deeplearning_ldpc_model import CTableNeuralLayeredMinSumDecoder
from deep_learning.ldpc_table_config import load_ldpc_table_config
from deep_learning.ldpc_training_data import make_c_table_training_batch
from deep_learning.ldpc_training_utils import CRC_TYPE_TO_POLY


def _grad_norm_float64(parameters) -> float:
    """Return global gradient norm using float64 accumulation."""
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            return float("inf")
        total_sq += float(torch.sum(grad.double() * grad.double()).cpu())
    return float(total_sq ** 0.5)


def _clip_grad_norm_float64(parameters, max_norm: float) -> float:
    """Clip gradients using a float64 norm to avoid float32 norm overflow."""
    params = [param for param in parameters if param.grad is not None]
    total_norm = _grad_norm_float64(params)
    if not np.isfinite(total_norm):
        return total_norm

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in params:
            param.grad.detach().mul_(clip_coef)
    return total_norm


def train_c_table_ldpc_model(
    bgn: int,
    A: int,
    crctype: int,
    E: int,
    rv: int,
    Qm: int,
    noise_power_db: float,
    HARQ_on: int = 0,
    num_iterations: int = 8,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-2,
    init_alpha: Union[float, Sequence[Sequence[float]], torch.Tensor] = 0.7,
    init_beta: float = 0.0,
    learnable_scaling: bool = True,
    learnable_beta: bool = False,
    alpha_mode: str = "single",
    alpha_min: float = 0.5,
    alpha_max: float = 1.0 - 1e-6,
    init_alpha_degree_values: Optional[Sequence[float]] = None,
    clip_llr: Optional[float] = None,
    detach_check_update_inputs: bool = False,
    training_noise_power_db_range: Optional[Tuple[float, float]] = None,
    loss_type: str = "bce",
    llr_margin: float = 5.0,
    device: Optional[torch.device] = None,
    cpu_threads: Optional[int] = 4,
    save_path: Optional[str] = "trained_c_table_ldpc_decoder.pth",
    lib_path: Optional[str] = None,
    intermediate_loss_weight: float = 0.0,
    intermediate_loss_stride: int = 4,
    verbose: bool = True,
) -> Tuple[CTableNeuralLayeredMinSumDecoder, Dict[str, torch.Tensor], float]:
    """Train alpha/beta for the table-driven layered min-sum decoder.

    Returns the model restored to its best epoch, the best state dict, and the
    best loss value observed during training.
    """
    if crctype not in CRC_TYPE_TO_POLY:
        raise ValueError("crctype must be 0, 1, 2, or 3")
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    if intermediate_loss_weight < 0:
        raise ValueError("intermediate_loss_weight must be non-negative")
    if intermediate_loss_stride <= 0:
        raise ValueError("intermediate_loss_stride must be positive")
    if loss_type not in {"bce", "margin"}:
        raise ValueError("loss_type must be 'bce' or 'margin'")
    if alpha_mode not in {"single", "checkpoint", "degree"}:
        raise ValueError("alpha_mode must be 'single', 'checkpoint', or 'degree'")
    if not 0.0 < alpha_min < alpha_max < 1.0:
        raise ValueError("alpha_min and alpha_max must satisfy 0 < alpha_min < alpha_max < 1")
    if llr_margin <= 0:
        raise ValueError("llr_margin must be positive")
    if training_noise_power_db_range is not None:
        low, high = training_noise_power_db_range
        if low > high:
            raise ValueError("training_noise_power_db_range must be in (low, high) order")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = CTableNeuralLayeredMinSumDecoder(
        table_config,
        num_iterations=num_iterations,
        learnable_scaling=learnable_scaling,
        learnable_beta=learnable_beta,
        init_alpha=init_alpha,
        init_beta=init_beta,
        alpha_mode=alpha_mode,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        init_alpha_degree_values=init_alpha_degree_values,
        clip_llr=clip_llr,
        detach_check_update_inputs=detach_check_update_inputs,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.05,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    def decoder_loss(llr: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the selected supervised loss on information+CRC bits."""
        logits = -llr[:, : table_config.B]
        if loss_type == "bce":
            return criterion(logits, target)

        signed_llr = (1.0 - 2.0 * target) * llr[:, : table_config.B]
        return torch.nn.functional.softplus(llr_margin - signed_llr).mean()

    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    if verbose:
        memory = model.memory_summary(batch_size=batch_size)
        print(
            "training C-table LDPC model: "
            f"bgn={bgn} A={A} B={table_config.B} Zc={table_config.Zc} "
            f"n={model.n} rows={model.num_rows} edges={model.num_edges} "
            f"alpha_mode={alpha_mode} "
            f"message_memory={memory['total_message_MB']:.2f} MB"
        )

    for epoch in range(epochs):
        batch_noise_power_db = noise_power_db
        # Optionally randomize the training SNR/noise level per batch so the
        # learned weights are less tied to a single channel point.
        if training_noise_power_db_range is not None:
            low, high = training_noise_power_db_range
            batch_noise_power_db = float(np.random.uniform(low, high))

        llr_batch, target_batch = make_c_table_training_batch(
            model=model,
            batch_size=batch_size,
            noise_power_db=batch_noise_power_db,
            device=device,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Intermediate supervision can make deep unrolled decoders easier to
        # optimize by applying loss before the final iteration only.
        if intermediate_loss_weight > 0:
            _, final_llr, iteration_llrs = model(
                llr_batch,
                return_llr=True,
                return_iteration_llrs=True,
            )
        else:
            _, final_llr = model(llr_batch, return_llr=True)
            iteration_llrs = []

        if not torch.isfinite(final_llr).all():
            finite_final_llr = final_llr[torch.isfinite(final_llr)]
            if finite_final_llr.numel() > 0:
                min_final_llr = float(finite_final_llr.min().detach().cpu())
                max_final_llr = float(finite_final_llr.max().detach().cpu())
            else:
                min_final_llr = float("nan")
                max_final_llr = float("nan")
            alpha_values = model.alpha().detach()
            raise FloatingPointError(
                "non-finite final_llr detected before loss; "
                f"epoch={epoch + 1}, noise_power_db={batch_noise_power_db:.2f}, "
                f"finite_final_llr_min={min_final_llr:.3e}, "
                f"finite_final_llr_max={max_final_llr:.3e}, "
                f"alpha_min={float(alpha_values.min().cpu()):.6f}, "
                f"alpha_max={float(alpha_values.max().cpu()):.6f}. "
                "Try lower init_alpha_degree_values/alpha_max, enable clip_llr, "
                "or reduce lr/num_iterations."
            )

        final_loss = decoder_loss(final_llr, target_batch)
        if intermediate_loss_weight > 0:
            supervised_llrs = iteration_llrs[intermediate_loss_stride - 1 :: intermediate_loss_stride]
            if supervised_llrs and supervised_llrs[-1] is final_llr:
                supervised_llrs = supervised_llrs[:-1]
            if supervised_llrs:
                intermediate_loss = torch.stack(
                    [
                        decoder_loss(iter_llr, target_batch)
                        for iter_llr in supervised_llrs
                    ]
                ).mean()
            else:
                intermediate_loss = final_loss.new_zeros(())
            loss = final_loss + intermediate_loss_weight * intermediate_loss
        else:
            intermediate_loss = final_loss.new_zeros(())
            loss = final_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "non-finite training loss detected; try lower init_alpha/alpha_max, "
                "enable clip_llr, or reduce lr/num_iterations"
            )

        alpha_before_step = model.alpha().detach()
        loss.backward()
        raw_alpha_grad_norm = 0.0
        if hasattr(model, "raw_alpha") and model.raw_alpha.grad is not None:
            raw_alpha_grad = model.raw_alpha.grad.detach()
            raw_alpha_grad_norm = _grad_norm_float64([model.raw_alpha])
            if not np.isfinite(raw_alpha_grad_norm):
                finite_count = int(torch.isfinite(raw_alpha_grad).sum().cpu())
                total_count = raw_alpha_grad.numel()
                raise FloatingPointError(
                    "non-finite raw_alpha gradient detected; "
                    f"epoch={epoch + 1}, noise_power_db={batch_noise_power_db:.2f}, "
                    f"finite_raw_alpha_grad_values={finite_count}/{total_count}. "
                    "Try enabling/lowering clip_llr, reducing lr, or reducing num_iterations."
                )
        total_grad_norm = _clip_grad_norm_float64(model.parameters(), max_norm=1.0)
        if not np.isfinite(total_grad_norm):
            raise FloatingPointError(
                "non-finite model gradient detected; "
                f"epoch={epoch + 1}, noise_power_db={batch_noise_power_db:.2f}. "
                "Try enabling/lowering clip_llr, reducing lr, or reducing num_iterations."
            )
        optimizer.step()
        scheduler.step()

        # Keep a CPU copy of the best state so training can continue without
        # holding extra GPU memory.
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            alpha_after_step = model.alpha().detach()
            alpha_mean = float(alpha_after_step.mean().cpu())
            alpha_min_value = float(alpha_after_step.min().cpu())
            alpha_max_value = float(alpha_after_step.max().cpu())
            alpha_step_max = float((alpha_after_step - alpha_before_step).abs().max().cpu())
            beta_mean = float(model.beta().detach().mean().cpu())
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"loss={loss_value:.6f} "
                f"final_loss={float(final_loss.item()):.6f} "
                f"intermediate_loss={float(intermediate_loss.item()):.6f} "
                f"alpha_mean={alpha_mean:.4f} "
                f"alpha_min={alpha_min_value:.6f} "
                f"alpha_max={alpha_max_value:.6f} "
                f"alpha_step_max={alpha_step_max:.3e} "
                f"raw_alpha_grad_norm={raw_alpha_grad_norm:.3e} "
                f"beta_mean={beta_mean:.4f} "
                f"noise_power_db={batch_noise_power_db:.2f} "
                f"lr={current_lr:.3e}"
            )

    model.load_state_dict(best_state)
    if verbose and alpha_mode in {"single", "checkpoint", "degree"}:
        alpha = model.alpha().detach().cpu().numpy()
        if alpha_mode == "single":
            trained_alpha_values = float(alpha.reshape(-1)[0])
        elif alpha_mode == "checkpoint":
            trained_alpha_values = tuple(float(value) for value in alpha.mean(axis=0))
        else:
            trained_alpha_values = {
                int(degree): float(value)
                for degree, value in zip(model.unique_row_degrees, alpha.reshape(-1))
            }
        print("trained alpha values", trained_alpha_values)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "table_config": table_config,
                "best_loss": best_loss,
            },
            save_path,
        )

    return model, best_state, best_loss
