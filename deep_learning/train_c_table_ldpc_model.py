"""Train the table-driven neural layered min-sum LDPC decoder.

This trainer is paired with ``c_table_ldpc_model.py``.  It uses the same
C-derived LDPC parameters/tables as the model, then generates training batches
with the existing Python CRC/encoder/channel/rate-recovery path.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

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

        scale = np.max(np.abs(decoder_llr))
        if scale > 0:
            decoder_llr = decoder_llr / scale * 128.0
            decoder_llr = np.clip(decoder_llr, -32768, 32767).astype(np.float32)

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
    init_alpha: float = 0.7,
    init_beta: float = 0.0,
    learnable_scaling: bool = True,
    learnable_beta: bool = False,
    clip_llr: Optional[float] = None,
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
        clip_llr=clip_llr,
    ).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.05,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

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
            f"message_memory={memory['total_message_MB']:.2f} MB"
        )

    for epoch in range(epochs):
        llr_batch, target_batch = make_c_table_training_batch(
            model=model,
            batch_size=batch_size,
            noise_power_db=noise_power_db,
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

        final_loss = criterion(-final_llr[:, : table_config.B], target_batch)
        if intermediate_loss_weight > 0:
            supervised_llrs = iteration_llrs[intermediate_loss_stride - 1 :: intermediate_loss_stride]
            if supervised_llrs and supervised_llrs[-1] is final_llr:
                supervised_llrs = supervised_llrs[:-1]
            if supervised_llrs:
                intermediate_loss = torch.stack(
                    [
                        criterion(-iter_llr[:, : table_config.B], target_batch)
                        for iter_llr in supervised_llrs
                    ]
                ).mean()
            else:
                intermediate_loss = final_loss.new_zeros(())
            loss = final_loss + intermediate_loss_weight * intermediate_loss
        else:
            intermediate_loss = final_loss.new_zeros(())
            loss = final_loss
        loss.backward()
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
            alpha_mean = float(model.alpha().detach().mean().cpu())
            beta_mean = float(model.beta().detach().mean().cpu())
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"loss={loss_value:.6f} "
                f"final_loss={float(final_loss.item()):.6f} "
                f"intermediate_loss={float(intermediate_loss.item()):.6f} "
                f"alpha_mean={alpha_mean:.4f} "
                f"beta_mean={beta_mean:.4f} "
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

    device = device or next(model.parameters()).device
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


if __name__ == "__main__":
    trained_model, trained_state, trained_loss = train_c_table_ldpc_model(
        bgn=1,
        A=232 + 256,
        crctype=1,
        E=256 * 4 * 2,
        rv=0,
        Qm=1,
        noise_power_db=5.5,
        num_iterations=32,
        epochs=100,
        batch_size=20,
        lr=2e-3,
        cpu_threads=4,
        intermediate_loss_weight=0.25,
        intermediate_loss_stride=4,
        save_path="trained_c_table_ldpc_decoder_bg1.pth",
    )
    print("best loss", trained_loss)

    alpha = trained_model.alpha().detach().cpu().numpy()
    beta = trained_model.beta().detach().cpu().numpy()
    print("alpha shape", alpha.shape)
    print("beta shape", beta.shape)

    test_results = evaluate_c_table_ldpc_model(
        trained_model,
        noise_power_db=5.5,
        num_test_blocks=400,
        batch_size=4,
    )
    print(test_results)
