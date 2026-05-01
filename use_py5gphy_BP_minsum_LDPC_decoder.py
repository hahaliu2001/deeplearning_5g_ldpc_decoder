# -*- coding:utf-8 -*-

import numpy as np

from py5gphy.ldpc import nr_ldpc_raterecover
from py5gphy.ldpc import nr_ldpc_decode
from py5gphy.ldpc import ldpc_info

def use_py5gphy_LDPC_decoder(LLR_bits, code_rate, LDPC_decoder_config):
    """
    the code is used to decode the LLR bits using LDPC decoder defined in 3GPP TS38.212 section 5.3.2
    I am using this code to compare with the layered min-sum LDPC decoder implemented in GEO_layered_min_sum_ldpc_decoder.py to verify the correctness of the layered min-sum LDPC decoder.
    the code call the methods in ldpc package to perform LDPC rate recovery and LDPC decoding
    it support only base graph 2, and code rate 1/4 or 1/5
    and it support rv(redundancy version) 0 only, which means no starting point shift for parity bits
    input:
        LLR_bits: the LLR bits input to the LDPC decoder, should be a 1D numpy array of size E, which is the codeword length after LDPC encoding and rate matching
        code_rate: the code rate of the LDPC code, should be 1/4 or 1/5
        LDPC_decoder_config: the configuration of the LDPC decoder, should be a dictionary with the following keys:
            L: iteration number
            algo: decoding algorithm, one of ["BF", "BP", "min-sum"]
            alpha: (0:1] used for normalized min-sum. 1 means no normalized
            beta: >=0 used for offset min-sum, 0 means no offset
    output:
        decoded_bits: the decoded bits after LDPC decoding, should be a 1D numpy array of size B, which is the information bit size before LDPC encoding
        status: decoding status, true for success, false for failure
    """
    # unpack LDPC parameters
    BG = 2 # only support base graph 2 for GEO LDPC spec
    
    E = len(LLR_bits) # codeword length after LDPC encoding and rate matching, which is also the input bit size E
    B = int(E * code_rate) # information bit size before LDPC encoding

    #Get information of code block segments
    #C is number of code block, cbz is code block size after inserting filler bits, L is number of filler bits, F is number of information bits in last code block, K is number of information bits in each code block, Zc is lifting factor
    #in Xona LDPC spec, 
    # C =1, 
    C, cbz, L, F, K, Zc = ldpc_info.get_cbs_info(B, BG)
    N = 50 * Zc # code block size after LDPC encoding, which is also the input bit size of LDPC decoder

    
    LLr_dn = nr_ldpc_raterecover.raterecover_ldpc(LLR_bits, Ncb=N, N=N, k0=0, Qm=1,Zc=Zc,K_apo=cbz,K=K)

    blkandcrc, ck, status = nr_ldpc_decode.nr_decode_ldpc(LLr_dn, Zc, BG, LDPC_decoder_config["L"], algo=LDPC_decoder_config["algo"],alpha=LDPC_decoder_config["alpha"], beta=LDPC_decoder_config["beta"])

    return blkandcrc, status

if __name__ == "__main__":
    from scipy import io
    import os
    from GEO_LDPC_encoder import Xona_LDPC_encoder
    
    print("test LDPC rate recover and decode")  
        
    #LDPC decoder config
    LDPC_decoder_config = {
        "L": 32, # iteration number
        "algo": "min-sum", # decoding algorithm, one of ["BF", "BP", "min-sum"]
        "alpha": 0.7, # used for normalized min-sum. 1 means no normalized
        "beta": 0.3, # used for offset min-sum, 0 means no offset
    }

    #noise power in dB
    noise_power_db = -20
    noise_power = 10 ** (noise_power_db / 10)
    for code_rate in [1/4, 1/5]:
        for B in [320,400,480,640,800,960,1280,3840,256,384,512,768]:
            print(f"Testing code rate: {code_rate}, Testing information bit length: {B}")
            info_bits = np.random.randint(0, 2, size=B-24) # generate random information bits of size B
                        
            xona_encoded_bits = Xona_LDPC_encoder(info_bits, code_rate)
            
            #BPSK modulation of xona_encoded_bits
            mod_bits = 1 - 2 * xona_encoded_bits  # BPSK modulation

            #add noise to the encoded bits, and convert them to LLR bits
            noise = np.sqrt(noise_power) * np.random.randn(len(xona_encoded_bits))
            received_bits = mod_bits + noise
            LLR_bits = 2 * received_bits / noise_power # assuming BPSK modulation
            
            decoded_bits, status = use_py5gphy_LDPC_decoder(LLR_bits, code_rate, LDPC_decoder_config)
    
            print("LDPC decode status: ", status)
            assert np.array_equal(decoded_bits[0:B-24], info_bits), "The encoded bits from sionna and Xona should be the same"
    
    pass