"""Train the table-driven neural layered min-sum LDPC decoder.

This trainer is paired with ``c_table_ldpc_model.py``.  It uses the same
C-derived LDPC parameters/tables as the model, then generates training batches
with the existing Python CRC/encoder/channel/rate-recovery path.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from c_table_ldpc_model import CTableNeuralLayeredMinSumDecoder, load_ldpc_table_config
from py_src.ldpc_5g_encoder import cb_ldpc_5g_encode
from py_src.ldpc_utils import tx_rx_process
from py5gphy.crc import crc
from py5gphy.ldpc import nr_ldpc_raterecover


CRC_TYPE_TO_POLY = {
    0: "16",
    1: "24A",
    2: "24B",
    3: "24C",
}


def _infer_model_device(model: torch.nn.Module) -> torch.device:
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
            #row_values.append(0.8125)
            row_values.append(0.6)
        elif degree < 8:
            #row_values.append(0.75)
            row_values.append(0.6)
        else:
            #row_values.append(0.6875)
            row_values.append(0.6)

    return np.tile(np.asarray(row_values, dtype=np.float32), (num_iterations, 1))


def load_trained_row_alpha_init(
    checkpoint_path: str,
    alpha_min: float,
    alpha_max: float,
    expected_shape: Tuple[int, int],
) -> np.ndarray:
    """Load a trained raw_alpha checkpoint and return the bounded alpha table."""
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


def make_c_table_training_batch(
    model: CTableNeuralLayeredMinSumDecoder,
    batch_size: int,
    noise_power_db: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one training batch for ``CTableNeuralLayeredMinSumDecoder``.

    Returns:
        llr_batch: shape ``[batch_size, model.n]``
        target_batch: shape ``[batch_size, model.config.B]``
    """
    cfg = model.config
    crcpoly = CRC_TYPE_TO_POLY[cfg.crctype]

    llrs = []
    targets = []
    for _ in range(batch_size):
        info_bits = np.random.randint(0, 2, size=cfg.A)
        target = crc.nr_crc_encode(info_bits, crcpoly, mask=0).astype(np.float32)

        codeword = cb_ldpc_5g_encode(
            info_bits=info_bits,
            E=cfg.E,
            Qm=cfg.Qm,
            rv=cfg.rv,
            crcpoly=crcpoly,
            bgn=cfg.bgn,
        )
        channel_llr = tx_rx_process(codeword, Qm=cfg.Qm, noise_power_db=noise_power_db)
        recovered_llr = nr_ldpc_raterecover.raterecover_ldpc(
            channel_llr,
            Ncb=cfg.N,
            N=cfg.N,
            k0=cfg.k0,
            Qm=cfg.Qm,
            Zc=cfg.Zc,
            K_apo=cfg.B,
            K=cfg.K,
        )
        decoder_llr = np.concatenate(
            (np.zeros(2 * cfg.Zc, dtype=recovered_llr.dtype), recovered_llr)
        ).astype(np.float32)

        if decoder_llr.size != model.n:
            raise RuntimeError(f"expected LLR length {model.n}, got {decoder_llr.size}")

        llrs.append(torch.from_numpy(decoder_llr))
        targets.append(torch.from_numpy(target))

    llr_batch = torch.stack(llrs, dim=0).to(device)
    target_batch = torch.stack(targets, dim=0).to(device)
    return llr_batch, target_batch


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
    alpha_mode: str = "row",
    alpha_min: float = 0.5,
    alpha_max: float = 1.0 - 1e-6,
    init_alpha_degree_values: Optional[Tuple[float, float, float]] = None,
    clip_llr: Optional[float] = None,
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
    """Train alpha/beta for the table-driven layered min-sum decoder."""
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
    if alpha_mode not in {"row", "degree"}:
        raise ValueError("alpha_mode must be 'row' or 'degree'")
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
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.05,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    def decoder_loss(llr: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
            raw_alpha_grad_norm = float(model.raw_alpha.grad.detach().norm().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

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
    if save_path is not None:
        torch.save(
            {
                "state_dict": best_state,
                "table_config": table_config,
                "best_loss": best_loss,
            },
            save_path,
        )

    return model, best_state, best_loss


def evaluate_c_table_ldpc_model(
    model: CTableNeuralLayeredMinSumDecoder,
    noise_power_db: float,
    num_test_blocks: int = 100,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Generate test blocks and evaluate decoder performance without training.

    Metrics:
        BER: bit error rate over the first ``B`` information+CRC bits.
        BLER: block error rate, one error if any of the first ``B`` bits fail.
    """
    if num_test_blocks <= 0:
        raise ValueError("num_test_blocks must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = device or _infer_model_device(model)
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
            decoded_bits = model(llr_batch)[:, : cfg.B]
            target_bits = target_batch.to(torch.int64)
            errors = decoded_bits.ne(target_bits)

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
    """Evaluate multiple decoders on identical generated LDPC test blocks."""
    if not models:
        raise ValueError("models must not be empty")
    if num_test_blocks <= 0:
        raise ValueError("num_test_blocks must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    first_model = next(iter(models.values()))
    device = device or _infer_model_device(first_model)
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
                decoded_bits = model(llr_batch)[:, : cfg.B]
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


if __name__ == "__main__":
    main_bgn = 1
    main_A = 232 + 256
    main_crctype = 1
    main_E = 256 * 4 * 2
    main_rv = 0
    main_Qm = 1
    main_num_iterations = 32
    main_alpha_min = 0.5
    main_alpha_max = 1.0 - 1e-6
    trained_alpha_init_path = "opt1_trained_c_table_ldpc_decoder_bg1.pth"
    main_table_config = load_ldpc_table_config(
        bgn=main_bgn,
        A=main_A,
        crctype=main_crctype,
        E=main_E,
        rv=main_rv,
        Qm=main_Qm,
        HARQ_on=0,
    )
    row_init_alpha = make_degree_seeded_row_alpha_init(
        main_table_config,
        num_iterations=main_num_iterations,
    )
    if os.path.exists(trained_alpha_init_path):
        row_init_alpha = load_trained_row_alpha_init(
            trained_alpha_init_path,
            alpha_min=main_alpha_min,
            alpha_max=main_alpha_max,
            expected_shape=(main_num_iterations, main_table_config.num_rows),
        )
        print(
            f"loaded init_alpha from {trained_alpha_init_path}: "
            f"shape={row_init_alpha.shape} "
            f"min={row_init_alpha.min():.6f} "
            f"max={row_init_alpha.max():.6f} "
            f"mean={row_init_alpha.mean():.6f}"
        )

    trained_model, trained_state, trained_loss = train_c_table_ldpc_model(
        bgn=main_bgn,
        A=main_A,
        crctype=main_crctype,
        E=main_E,
        rv=main_rv,
        Qm=main_Qm,
        noise_power_db=5,
        num_iterations=main_num_iterations,
        epochs=100,
        batch_size=20,
        lr=1e-3,
        init_alpha=row_init_alpha,
        init_beta=0.0,
        learnable_scaling=True,
        learnable_beta=False,
        alpha_mode="row",
        alpha_min=main_alpha_min,
        alpha_max=main_alpha_max,
        init_alpha_degree_values=None,
        training_noise_power_db_range=(5.0, 6.0),
        loss_type="bce",
        llr_margin=5.0,
        clip_llr=None,
        cpu_threads=4,
        intermediate_loss_weight=0.25,
        intermediate_loss_stride=4,
        save_path="trained_c_table_ldpc_decoder_bg1.pth",
    )
    print("best loss", trained_loss)

    alpha = trained_model.alpha().detach().cpu().numpy()
    beta = trained_model.beta().detach().cpu().numpy()
    print("alpha shape", alpha.shape)
    print(
        "alpha stats",
        {
            "min": float(alpha.min()),
            "max": float(alpha.max()),
            "mean": float(alpha.mean()),
            "std": float(alpha.std()),
        },
    )
    print("beta shape", beta.shape)
    print(
        "beta stats",
        {
            "min": float(beta.min()),
            "max": float(beta.max()),
            "mean": float(beta.mean()),
            "std": float(beta.std()),
        },
    )

    fixed_model = CTableNeuralLayeredMinSumDecoder(
        trained_model.config,
        num_iterations=trained_model.num_iterations,
        learnable_scaling=False,
        learnable_beta=False,
        init_alpha=0.75,
        init_beta=0.0,
        alpha_mode="degree",
        alpha_min=trained_model.alpha_min,
        alpha_max=trained_model.alpha_max,
        init_alpha_degree_values=(0.8125, 0.75, 0.6875),
    ).to(_infer_model_device(trained_model))

    test_results = {}
    for eval_noise_power_db in (4.5, 5.0, 5.5):
        print(f"paired evaluation noise_power_db={eval_noise_power_db:.1f}")
        test_results[eval_noise_power_db] = evaluate_c_table_ldpc_models_paired(
            {
                "fixed_init_alpha": fixed_model,
                "trained_alpha": trained_model,
            },
            noise_power_db=eval_noise_power_db,
            num_test_blocks=800,
            batch_size=4,
        )
    print(test_results)

    raw = trained_model.raw_alpha.detach()
    trained_alpha = (
        trained_model.alpha_min
        + (trained_model.alpha_max - trained_model.alpha_min) * torch.sigmoid(raw)
    )
    print(trained_alpha.cpu().numpy())
