"""Training helpers for the neural 5G LDPC min-sum decoder.

This file keeps the training loop small and explicit:
- generate random information blocks
- CRC/LDPC encode and pass through the channel
- rate-recover to decoder LLR length
- train the learnable min-sum alpha/beta parameters
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Type
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from codex_ldpc_model import NeuralLayeredFloatpointMinSumDecoder
from py_src.ldpc_5g_encoder import cb_ldpc_5g_encode
from py_src.ldpc_utils import tx_rx_process
from py5gphy.crc import crc
from py5gphy.ldpc import ldpc_info
from py5gphy.ldpc import nr_ldpc_ratematch
from py5gphy.ldpc import nr_ldpc_raterecover


def _crc_len(crcpoly: str) -> int:
    if crcpoly == "16":
        return 16
    if crcpoly in {"24A", "24B", "24C"}:
        return 24
    raise ValueError("crcpoly must be one of '16', '24A', '24B', or '24C'")


def make_training_batch(
    B: int,
    E: int,
    Qm: int,
    noise_power_db: float,
    rv: int,
    crcpoly: str,
    bgn: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create one training batch.

    Returns:
        llr_batch: shape ``[batch_size, n]`` for the neural decoder
        target_batch: shape ``[batch_size, B]`` with information+CRC bits
    """
    crc_bits = _crc_len(crcpoly)
    A = B - crc_bits
    if A <= 0:
        raise ValueError("B must be larger than the selected CRC length")
    if A % 8 != 0:
        raise ValueError("B minus CRC length must be a multiple of 8")

    _, cbz, L, _, K, Zc = ldpc_info.get_cbs_info(B, bgn)
    K_apo = cbz + L
    N = (66 if bgn == 1 else 50) * Zc
    k0 = nr_ldpc_ratematch.get_k0(N, bgn, rv, Zc)

    targets = []
    llrs = []
    for _ in range(batch_size):
        info_bits = np.random.randint(0, 2, size=A)
        target = crc.nr_crc_encode(info_bits, crcpoly, mask=0).astype(np.float32)

        codeword = cb_ldpc_5g_encode(
            info_bits=info_bits,
            E=E,
            Qm=Qm,
            rv=rv,
            crcpoly=crcpoly,
            bgn=bgn,
        )
        channel_llr = tx_rx_process(codeword, Qm=Qm, noise_power_db=noise_power_db)
        recovered_llr = nr_ldpc_raterecover.raterecover_ldpc(
            channel_llr,
            Ncb=N,
            N=N,
            k0=k0,
            Qm=Qm,
            Zc=Zc,
            K_apo=K_apo,
            K=K,
        )
        decoder_llr = np.concatenate(
            (np.zeros(2 * Zc, dtype=recovered_llr.dtype), recovered_llr)
        ).astype(np.float32)

        targets.append(torch.from_numpy(target))
        llrs.append(torch.from_numpy(decoder_llr))

    llr_batch = torch.stack(llrs, dim=0).to(device)
    target_batch = torch.stack(targets, dim=0).to(device)
    return llr_batch, target_batch


def train_ldpc_decoder_model(
    B: int,
    E: int,
    Qm: int,
    noise_power_db: float,
    rv: int,
    crcpoly: str,
    bgn: int,
    model_cls: Type[NeuralLayeredFloatpointMinSumDecoder] = NeuralLayeredFloatpointMinSumDecoder,
    num_iterations: int = 8,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-2,
    learnable_scaling: bool = True,
    learnable_beta: bool = True,
    init_alpha: float = 0.7,
    init_beta: float = 0.0,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    cpu_threads: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[NeuralLayeredFloatpointMinSumDecoder, Dict[str, torch.Tensor], float]:
    """Train a neural layered min-sum LDPC decoder.

    The defaults are intentionally modest because the current decoder stores
    dense message tensors and can consume a lot of memory for 5G LDPC matrices.
    Increase ``batch_size`` and ``num_iterations`` only after checking RAM/VRAM.
    """
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, _, _, Zc = ldpc_info.get_cbs_info(B, bgn)
    iLS = ldpc_info.find_iLS(Zc)
    H = ldpc_info.getH(Zc, bgn, iLS)

    model = model_cls(
        H,
        num_iterations=num_iterations,
        learnable_scaling=learnable_scaling,
        learnable_beta=learnable_beta,
        init_alpha=init_alpha,
        init_beta=init_beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state: Dict[str, torch.Tensor] = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

    for epoch in range(epochs):
        llr_batch, target_batch = make_training_batch(
            B=B,
            E=E,
            Qm=Qm,
            noise_power_db=noise_power_db,
            rv=rv,
            crcpoly=crcpoly,
            bgn=bgn,
            batch_size=batch_size,
            device=device,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, final_llr = model(llr_batch, return_llr=True)
        loss = criterion(-final_llr[:, :B], target_batch)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            print(f"epoch {epoch + 1}/{epochs} loss={loss_value:.6f}")

    model.load_state_dict(best_state)

    if save_path is not None:
        torch.save(best_state, save_path)

    return model, best_state, best_loss


if __name__ == "__main__":
    trained_model, trained_state, trained_loss = train_ldpc_decoder_model(
        B=256,
        E=256 * 4,
        Qm=1,
        noise_power_db=4,
        rv=0,
        crcpoly="24A",
        bgn=2,
        num_iterations=8,
        epochs=20,
        batch_size=4,
        lr=1e-2,
        cpu_threads=4,
        save_path="trained_ldpc_decoder2.pth",
    )
    print("best loss", trained_loss)
