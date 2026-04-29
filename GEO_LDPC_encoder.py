# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from LDPC_parameter import gen_LDPC_parameter
from py5gphy.ldpc import nr_ldpc_encode
from py5gphy.ldpc import nr_ldpc_ratematch
from py5gphy.ldpc import nr_ldpc_cbsegment
from py5gphy.crc import crc

"""
04/06/2026 Jie Liu
About GEO LDPC encoder.
GEO uses a subset of the LDPC code defined in 3GPP TS38.212 section 5.3.2, which is also used for 5G NR.
The code supports only base graph 2, and code rate 1/4 or 1/5, and it support rv(redundancy version) 0 only

I have generated three LDPC encoder methods to verify the correctness of the LDPC encoder implementation
version 1:
    https://github.com/NVlabs/sionna. 
    The code call open source Sionna LDPC encoder to perform LDPC code block segmentation, LDPC encoding and LDPC rate matching
    Sionna LDPCencoder code has some issues. It doesn;t fully match with 3GPP specification. But it has no issue for Xona LDPC configuration
    The code use tensorflow which is not easy to use and debug. I have removed this code to avoid confusion.
    The code is in ./not_used/GEO_LDPC_encoder.py
    
version 2:
    https://github.com/hahaliu2001/python_5gtoolbox/tree/main/py5gphy/ldpc
    This open source code is implemented by myself, and it is fully compliant with 3GPP specification. It has been verified with the 60k+ test vectors provided by 3GPP.
    I have used this code for the LDPC encoder and compared the results with Sionna LDPCencoder. Both of them give the same results for Xona LDPC configuration.  

version 3:
    Xona LDPC encoder is a small subset of 3GPP 5G LDPC. 
    I have implemented a simplified LDPC encoder for Xona LDPC configuration to make it easier to understand and debug. The code is based on the paper 'Flexible 5G New Radio LDPC Encoder Optimized for High Hardware Usage Efficiency' (https://www.mdpi.com/2079-9292/10/9/1106) and the optimization is also explained in docs/LDPC_encoder_optimization.pdf
    I will verify the code with the version 2 LDPC encoder
"""

def use_py5gphy_LDPC_encoder(info_bits, code_rate):
    """
    this is version 2 that use the methods in ldpc package in https://github.com/hahaliu2001/python_5gtoolbox/tree/main/py5gphy/ldpc
    the code is used to encode the information bits using LDPC encoder defined in 3GPP TS38.212 section 5.3.2
    the code call the methods in ldpc package to perform LDPC code block segmentation, LDPC encoding and LDPC rate matching
    it support only base graph 2, and code rate 1/4 or 1/5
    and it support rv(redundancy version) 0 only, which means no starting point shift for parity bits
    input:
        info_bits: the information bits to be encoded, should be a 1D numpy array of size B
        
    output:
        encoded_bits: the encoded bits after LDPC encoding, should be a 1D numpy array of size E
    """
    
    A = len(info_bits)
    blkandcrc = crc.nr_crc_encode(info_bits,'24A') # perform CRC encoding, 
    B = A + 24 # 24bits CRC bits

    n = int(B / code_rate) # codeword length after LDPC encoding and rate matching

    cb = nr_ldpc_cbsegment.ldpc_cbsegment(blkandcrc, bgn=2)[0][0,:] # code block segmentation

    dn = nr_ldpc_encode.encode_ldpc(cb, bgn=2)
    encoded_bits = nr_ldpc_ratematch.ratematch_ldpc(dn, dn.size,n,k0=0,Qm=1)

    return encoded_bits

def Xona_LDPC_encoder(info_bits, code_rate):
    """
    This is version 3 that implement a simplified LDPC encoder for Xona LDPC configuration to 
    make it easier to understand and debug. 
    
    it support only base graph 2, and code rate 1/4 or 1/5
    and it support rv(redundancy version) 0 only, which means no starting point shift for parity bits
    input:
        info_bits: the information bits to be encoded, should be a 1D numpy array of size B
        
    output:
        encoded_bits: the encoded bits after LDPC encoding, should be a 1D numpy array of size E
    """
    
    A = len(info_bits)
    blkandcrc = crc.nr_crc_encode(info_bits,'24A') # perform CRC encoding, the output is the concatenation of information bits and CRC bits, and the size is K=Zc*10
    B = A + 24 # 24bits CRC bits

    [Es, N, Zc, F, H],_ = gen_LDPC_parameter(B, code_rate)

    #add filler bits to the information bits to make the code block size after inserting filler bits to be cbz
    #filling bits are set to 0 for LDPC encoding
    cb = np.zeros(Zc*10, dtype=np.int8) # initialize the code block with -1, which means filler bits
    cb[0:B] = blkandcrc # insert the information bits to the code block, the remaining bits are filler bits which are set to 0
    
    wn = _gen_ldpc_parity_bit(H, cb, Zc, 10*Zc, N)

    #rate matching
    encoded_bits = np.concatenate((cb[2*Zc:B], wn[0:Es-(B-2*Zc)]))

    return encoded_bits

def _gen_ldpc_parity_bit(H, ck, Zc, K, N):
    """ generate LDPC paraty bit"""
    #optimization based on 'Flexible 5G New Radio LDPC Encoder Optimized for High Hardware Usage Efficiency'
    #https://www.mdpi.com/2079-9292/10/9/1106
    #can also refer to docs/LDPC_encoder_optimization.pdf

    A = H[0:4*Zc, 0:K]
    B = H[0:4*Zc, K:K+4*Zc]
    C = H[4*Zc:N+2*Zc, 0:K+4*Zc]

    L1 = A @ ck.T
    L1=L1.reshape((4,-1))
    L2 = np.sum(L1,axis=0) % 2

    #below code is for bgn2, for bgn1
    zeros_idx = np.nonzero(B[2*Zc,0:Zc])[0][0]
    pc1 = np.roll(L2,zeros_idx)
    pc2 = (L1[0,:] + B[0:Zc,0:Zc] @pc1.T) % 2
    pc4 = (L1[3,:] + B[3*Zc:4*Zc,0:Zc] @pc1.T) % 2
    pc3 = (L1[1,:] + B[1*Zc:2*Zc,1*Zc:2*Zc] @pc2.T) % 2

    pc = np.array([pc1,pc2,pc3,pc4]).reshape((-1,1))
    pc = pc[:,0]

    combined = np.concatenate((ck,pc))
    pe = (C @ combined.T) % 2
    wn = np.concatenate((pc,pe))
    
    return wn


if __name__ == "__main__":
    # try to encode a random information bits using GEO LDPC encoder
    
    for code_rate in [1/4, 1/5]:
        print(f"Testing code rate: {code_rate}")
        for B in [320,400,480,640,800,960,1280,3840,256,384,512,768]:
            print(f"Testing information bit length: {B}")
            info_bits = np.random.randint(0, 2, size=B-24) # generate random information bits of size B
            
            xona_encoded_bits = Xona_LDPC_encoder(info_bits, code_rate)

            use_py5gphy_encoded_bits = use_py5gphy_LDPC_encoder(info_bits, code_rate)
            
            assert np.array_equal(xona_encoded_bits, use_py5gphy_encoded_bits)

    pass