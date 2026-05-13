"""Utilities to train a NeuralLayeredFloatpointMinSumDecoder.

Provides a simple training function that optimizes the decoder's learnable
parameters on noisy all-zero codewords (AWGN) using BCEWithLogitsLoss as in
the example in codex_ldpc_model.py.
"""
from typing import Optional, Tuple, Dict, Any
import sys
import numpy as np
import os
import pickle

sys.path.append(os.getcwd()) 

from py_src.ldpc_5g_decoder import cb_ldpc_5g_decode
from py_src.ldpc_utils import tx_rx_process
from py_src.ldpc_5g_encoder import cb_ldpc_5g_encode
from py5gphy.crc import crc
from py5gphy.ldpc import ldpc_info
from py5gphy.ldpc import nr_ldpc_raterecover
from py5gphy.ldpc import nr_ldpc_ratematch

import torch

from codex_ldpc_model import NeuralLayeredFloatpointMinSumDecoder


def train_neural_min_sum_decoder( B,E, Qm, noise_power_db, rv, crcpoly, bgn,
    Model: NeuralLayeredFloatpointMinSumDecoder,
    num_iterations = 32,
    learnable_scaling = True,
    learnable_beta = True,
    init_alpha = 0.7,
    init_beta = 0.0,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-2,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[NeuralLayeredFloatpointMinSumDecoder, Optional[Dict[str, torch.Tensor]], float]:
    """Train a NeuralLayeredFloatpointMinSumDecoder on noisy all-zero codewords.

    Args:
        B:informatioin bit size including crc
        E: rate matching out bit size
        Qm: modulation [1,2,4,6,8] map to BPSK,QPSK,16QAM,64QAM,256QAM
        noise_power_db: noise power
        rv: redundency value[0,1,2,3]
        crcpoly:["16","24A","24B","24C"]
        bgn: base graph [1,2], 
        
        Model: an instance of NeuralLayeredFloatpointMinSumDecoder
        epochs: number of training epochs
        batch_size: training batch size
        lr: learning rate for Adam optimizer
        device: torch.device to use (defaults to CUDA if available)
        verbose: print periodic progress

    Returns:
        (model, best_state_dict_cpu, best_loss)
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    _, cbz, L, F, K, Zc = ldpc_info.get_cbs_info(B, bgn)
    K_apo = cbz + L #filler bit location of CB segment output is from K_apo to K
    iLS = ldpc_info.find_iLS(Zc)
    H = ldpc_info.getH(Zc, bgn, iLS)

    model = Model(H, num_iterations=num_iterations, 
            learnable_scaling=learnable_scaling, learnable_beta=learnable_beta, init_alpha=init_alpha, init_beta=init_beta)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    n = model.n
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None    

    for epoch in range(epochs):
        #generate LLR data
        if crcpoly == "16":
            A = B - 16
        else:
            A = B - 24
        
        # generate a batch of blkandcrc and llr tensors
        blkandcrc_list = []
        llr_list = []
        for _ in range(batch_size):
            info_bits = np.random.randint(0, 2, size=A)
            blkandcrc = crc.nr_crc_encode(info_bits, crcpoly, mask=0)
            # ensure torch tensor
            if not isinstance(blkandcrc, torch.Tensor):
                blkandcrc = torch.tensor(blkandcrc, dtype=torch.float32)
            else:
                blkandcrc = blkandcrc.float()
            blkandcrc_list.append(blkandcrc)
            codeword = cb_ldpc_5g_encode(info_bits=info_bits, E=E, Qm=Qm, rv=rv, crcpoly=crcpoly, bgn=bgn)
            LLRin = tx_rx_process(codeword, Qm=Qm, noise_power_db=noise_power_db)
            E = len(LLRin)
            if bgn == 1:
                N = 66 * Zc
            else:
                N = 50 * Zc
            k0 = nr_ldpc_ratematch.get_k0(N, bgn, rv, Zc)
            llr = nr_ldpc_raterecover.raterecover_ldpc(LLRin, Ncb=N, N=N, k0=k0, Qm=Qm,Zc=Zc,K_apo=K_apo,K=K)
            llr = np.concatenate((np.zeros(2 * Zc, dtype=llr.dtype), llr))

            if not isinstance(llr, torch.Tensor):
                llr = torch.tensor(llr, dtype=torch.float32)
            llr = llr.to(device)
            llr_list.append(llr)

        # stacked tensors: blkandcrc_batch shape [batch_size, B], llr_batch shape [batch_size, len(llr)]
        blkandcrc_batch = torch.stack(blkandcrc_list, dim=0).to(device)  # [batch_size, B]
        llr_batch = torch.stack(llr_list, dim=0).to(device)              # [batch_size, len(llr)]
        
        model.train()
        opt.zero_grad()
        _, final_llr = model(llr_batch, return_llr=True)
        targets = blkandcrc_batch.float()
        # final_llr expected shape [batch_size, n]; compare only first B bits
        loss = criterion(-final_llr[:, 0:B], targets)
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            a = model.alpha().detach().cpu().numpy()
            b = model.beta().detach().cpu().numpy()
            print(f"epoch {epoch+1}/{epochs} loss={loss.item():.4f} alpha={a} beta={b}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_state, best_loss


if __name__ == "__main__":
    # quick demo: instantiate, train, and save best state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)

    B,E, Qm, noise_power_db, rv, crcpoly, bgn = 256,256*4,1,6,0,'24A',2

    model, state, loss = train_neural_min_sum_decoder(B,E, Qm, noise_power_db, rv, crcpoly, bgn,
            NeuralLayeredFloatpointMinSumDecoder, num_iterations=16,epochs=200, batch_size=4, lr=1e-2, device=device)
    #if state is not None:
    # torch.save(state, "trained_ldpc_decoder.pth")
    print("best loss", loss)
