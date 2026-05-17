"""Training and evaluation batch generation for neural LDPC models.

Each generated block follows the same path as the Python LDPC simulation:
random information bits, CRC attachment, LDPC encoding/rate matching, AWGN
channel, demodulation, and LDPC rate recovery. The returned LLR layout matches
the C decoder layout expected by ``CTableNeuralLayeredMinSumDecoder``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from deep_learning.deeplearning_ldpc_model import CTableNeuralLayeredMinSumDecoder
from deep_learning.ldpc_training_utils import CRC_TYPE_TO_POLY
from py5gphy.crc import crc
from py5gphy.ldpc import nr_ldpc_raterecover
from py5gphy.ldpc_5g_encoder import cb_ldpc_5g_encode
from py5gphy.ldpc_utils import tx_rx_process


def make_c_table_training_batch(
    model: CTableNeuralLayeredMinSumDecoder,
    batch_size: int,
    noise_power_db: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one training/evaluation batch for the table-driven decoder.

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
        # The training target is the LDPC input block: information bits plus CRC.
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
        # Convert the rate-matched channel LLRs back to the full LDPC decoder
        # memory layout before feeding the neural model.
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
