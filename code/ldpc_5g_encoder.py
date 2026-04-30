# -*- coding:utf-8 -*-

import sys
import numpy as np
import os

sys.path.append(os.getcwd()) 

from py5gphy.ldpc import nr_ldpc_ratematch
from py5gphy.ldpc import nr_ldpc_encode
from py5gphy.ldpc import ldpc_info
from py5gphy.crc import crc

def nr_encode_ldpc(info_bits, E, Qm, rv, crcpoly, bgn):
    """ 
    summary: 5G LDPC encoding and rate matching
    By 5G spec, bgn selection is based on block block size and code rate. In this code, we indicate bgn as an input parameter for flexibility
    this code support only single code block, which means the information bit size after CRC should be less than or equal to 8448 for base graph 1 and 3840 for base graph 2

    Args:
        info_bits (_type_): input bits before CRC
        E: the codeword length after LDPC encoding and rate matching
        Qm: modulation order, should be 1 for BPSK, 2 for QPSK, 4 for 16QAM, 6 for 64QAM, 8 for 256QAM
            Qm affect rate matching, which is defined in 3GPP TS38.212 section
        rv: redundancy version: 0, 1, 2, 3, which corresponds to the starting point shift for parity bits in LDPC encoding and rate matching
        crc: cyclic redundancy check type, should be "24A", "24B", "24C", only support 24-bit CRC for now
        bgn: base graph number, should be 1 or 2
    Returns:
        codeword: the codeword after LDPC encoding and rate matching, should be a 1D numpy array of size E
    """

    A = len(info_bits) # information bit size before CRC
    B = A + 24 # information bit size after CRC, which is also the input bit size for LDPC encoding

    assert A % 8 == 0, "the information bit size before CRC should be a multiple of 8"
    assert bgn in [1, 2], "bgn should be 1 or 2"
    assert crcpoly in ["24A", "24B", "24C"], "crc should be one of '24A', '24B', '24C'"
    assert Qm in [1, 2, 4, 6, 8], "Qm should be one of 1, 2, 4, 6, 8"
    assert rv in [0, 1, 2, 3], "rv should be one of 0, 1, 2, 3"
    assert E % Qm == 0, "the codeword length after LDPC encoding and rate matching should be a multiple of modulation order Qm"

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

    # perform CRC encoding and insert the information bits to the code block, the remaining bits are filler bits which are set to -1
    cb = -1 * np.ones((1, K),'i1')  # -1 is filler bits
    cb[0,0:B] = crc.nr_crc_encode(info_bits,    crcpoly, mask=0) # perform CRC encoding, the output is the concatenation of information bits and CRC bits, and the size is K=Zc*22 for base graph 1 and K=Zc*10 for base graph 2

    # perform LDPC encoding
    dn = nr_ldpc_encode.encode_ldpc(cb[0,:], bgn)
    # perform LDPC rate matching
    k0 = nr_ldpc_ratematch.get_k0(Ncb=dn.size, bgn=bgn, rv=rv, Zc=Zc)
    codeword = nr_ldpc_ratematch.ratematch_ldpc(dn, dn.size, E, k0, Qm)

    return codeword

if __name__ == "__main__":
    "I have verified the LDPC encoder and rate matching in https://github.com/hahaliu2001/python_5gtoolbox.git."
    "The code in this file is the same as the code in that repository"
    print("run LDPC 5G encoder")

    codeword = nr_encode_ldpc(info_bits=np.random.randint(0, 2, size=256), E=1024, Qm=4, rv=0, crcpoly="24A", bgn=1)
    codeword = nr_encode_ldpc(info_bits=np.random.randint(0, 2, size=8448-24), E=10240, Qm=2, rv=0, crcpoly="24A", bgn=1)
    codeword = nr_encode_ldpc(info_bits=np.random.randint(0, 2, size=3840-24), E=5120, Qm=4, rv=0, crcpoly="24A", bgn=2)

    pass
    