# -*- coding:utf-8 -*-

import sys
import numpy as np
import os

sys.path.append(os.getcwd()) 

from py5gphy.crc import crc
from py5gphy.ldpc import ldpc_info, nr_ldpc_decode
from py5gphy.ldpc import ldpc_decoder_bit_flipping
from py5gphy.ldpc import nr_ldpc_raterecover
from py5gphy.ldpc import nr_ldpc_ratematch


def ldpc_5g_decode(LLRin, B, Qm, rv, crcpoly, bgn, ldpcdecoder_config,HARQ_on=False, HARQ_buffer=None):
    """ LDPC decode following to TS38.212 5.3.2
    
    input: 
        LLRin: E length LDPC decoder LLR input data
        B: information bit size before LDPC encoding, which is also the input bit size for CRC decoding
        Qm: modulation order, should be 1 for BPSK, 2 for QPSK, 4 for 16QAM, 6 for 64QAM, 8 for 256QAM
        rv: redundancy version: 0, 1, 2, 3, which corresponds to the starting point shift for parity bits in LDPC encoding and rate matching
        crcpoly: cyclic redundancy check type, should be "24A", "24B", "24C", only support 24-bit CRC for now
        bgn: base graph value 1 or 2
        ldpcdecoder_config: a dictionary containing the configuration for LDPC decoder, including:
            L: iteration size
            algo: decoding algorithm, one of ["BF","BP", "min-sum"]
            alpha: (0:1] used for normalized min-sum. 
    output:
        ck: B length LDPC decoded sequence
    
    """
    
    E = len(LLRin) # codeword length after LDPC encoding and rate matching, which is also the input bit size E
    
    assert E % Qm == 0, "the codeword length after LDPC encoding and rate matching should be a multiple of modulation order Qm"
    assert bgn in [1, 2], "bgn should be 1 or 2"
    assert crcpoly in ["24A", "24B", "24C"], "crc should be one of '24A', '24B', '24C'"
    assert B % 8 == 0, "the information bit size after CRC should be a multiple of 8"
    assert Qm in [1, 2, 4, 6, 8], "Qm should be one of 1, 2, 4, 6, 8"
    assert rv in [0, 1, 2, 3], "rv should be one of 0, 1, 2, 3"

    #support only single code block, which means the information bit size after CRC should be less than or equal to 8448 for base graph 1 and 3840 for base graph 2
    if bgn == 1:
        assert B <= 8448, "for base graph 1, the information bit size after CRC should be less than or equal to 8448"
    else:        
        assert B <= 3840, "for base graph 2, the information bit size after CRC should be less than or equal to 3840"
    assert E >= B, "the codeword length after LDPC encoding and rate matching should be greater than or equal to the information bit size after CRC"

    #Get information of code block segments
    #C is the number of code blocks, cbz is the code block size, 
    # L is the number of CRC bits for codeblock, F is the number of filler bits, 
    # K is the code block size after inserting CRC and filler bits, 
    # Zc is the lifting size
    # in this code, C=1, L=0, cbz = B
    C, cbz, L, F, K, Zc = ldpc_info.get_cbs_info(B, bgn)

    K_apo = cbz + L #filler bit location of CB segment output is from K_apo to K

    if bgn == 1:
        N = 66 * Zc
    else:
        N = 50 * Zc
    
    k0 = nr_ldpc_ratematch.get_k0(N, bgn, rv, Zc)

    LLr_dn = nr_ldpc_raterecover.raterecover_ldpc(LLRin, Ncb=N, N=N, k0=k0, Qm=Qm,Zc=Zc,K_apo=K_apo,K=K)

    if HARQ_on:
        if HARQ_buffer is not None:
            LLr_dn = LLr_dn + HARQ_buffer
        HARQ_buffer = LLr_dn.copy() # save the LLR for HARQ combining in next transmission
    
    blkandcrc, ck, status = nr_ldpc_decode.nr_decode_ldpc(LLr_dn, Zc, bgn,crcpoly, ldpcdecoder_config)

    return blkandcrc, status, HARQ_buffer

if __name__ == "__main__":
    import ldpc_5g_encoder
    import py_src.ldpc_utils as ldpc_utils
    
    Qm,B, noise_power_db = 4, 256, -20
    info_bits=np.random.randint(0, 2, size=B-24)
    codeword = ldpc_5g_encoder.ldpc_5g_encode(info_bits=info_bits, E=1024, Qm=Qm, rv=0, crcpoly="24A", bgn=1)
    LLrout = ldpc_utils.tx_rx_process(codeword, Qm=Qm, noise_power_db=noise_power_db)
    blkandcrc, status, _ = ldpc_5g_decode(LLrout, B=B, Qm=Qm, rv=0, crcpoly="24A", bgn=1, ldpcdecoder_config={"L": 32, "algo": "min-sum", "alpha": 0.7, "beta": 0.3})

    codeword = ldpc_5g_encoder.ldpc_5g_encode(info_bits=np.random.randint(0, 2, size=8448-24), E=10240, Qm=2, rv=0, crcpoly="24A", bgn=1)
    codeword = ldpc_5g_encoder.ldpc_5g_encode(info_bits=np.random.randint(0, 2, size=3840-24), E=5120, Qm=4, rv=0, crcpoly="24A", bgn=2)
    pass