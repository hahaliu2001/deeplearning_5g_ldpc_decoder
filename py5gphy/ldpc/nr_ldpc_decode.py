# -*- coding:utf-8 -*-

import numpy as np
from py5gphy.crc import crc
from py5gphy.ldpc import ldpc_info
from py5gphy.ldpc import nr_ldpc_decoder
from py5gphy.ldpc import ldpc_decoder_bit_flipping
from py5gphy.ldpc import nr_ldpc_raterecover


def nr_decode_ldpc(LLRin, B, Qm, rv, crcpoly, bgn, ldpcdecoder_config):
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

def _min_sum_process(sel_Lq, A, Lr,m, alpha, beta):
    """ mixed min-sim processing
    alpha = 1, beta =0: min-sum
    alpha < 1, beta = 0: normalized min-sum
    alpha = 1, beta >0: offset min-sum
    alpha <1, beta >0: mixed min-sum
    """
    
    zero_value_list = list(np.where(sel_Lq==0)[0])
    
    if len(zero_value_list) == 0:
        #normal case, no zero value
        sign_prod = np.prod(np.sign(sel_Lq))
        sorted_abs_v = np.sort(np.abs(sel_Lq))

        first_min = sorted_abs_v[0]
        second_min = sorted_abs_v[1]
        for idx, n in enumerate(A[m]):
            if np.abs(sel_Lq[idx]) == first_min:
                minv = second_min
            else:
                minv = first_min
                
            sel_minv = max(minv - beta, 0) #offset min-sum  
            Lr[m,n] = alpha * sign_prod * np.sign(sel_Lq[idx]) * sel_minv
    elif len(zero_value_list) == 1:
        #only one zero value, only zero value related Lr is non-zero
        for idx, n in enumerate(A[m]):
            Lr[m,n] = 0
        
        zero_idx = zero_value_list[0]
        
        if zero_idx == 0:
            sign_prod = np.prod(np.sign(sel_Lq[1:]))
            min_v = np.min(np.abs(sel_Lq[1:]))
        elif zero_idx == sel_Lq.size-1: #last bit is zero
            sign_prod = np.prod(np.sign(sel_Lq[0:zero_idx]))
            min_v = np.min(np.abs(sel_Lq[0:zero_idx]))
        else:
            sign_prod = np.prod(np.sign(sel_Lq[0:zero_idx]))*np.prod(np.sign(sel_Lq[zero_idx+1:]))
            min_v = min(np.min(np.abs(sel_Lq[0:zero_idx])),np.min(np.abs(sel_Lq[zero_idx+1:])))
        
        sel_minv = max(min_v - beta, 0) #offset min-sum  
        Lr[m,A[m][zero_idx]] = alpha * sign_prod *sel_minv 
    else:
        #>=2 zero values. all Lr are zero
        for idx, n in enumerate(A[m]):
            Lr[m,n] = 0
    
    return Lr

def for_test_5g_ldpc_encoder(Zc, bgn, snr_db,crcpoly='24A'):
    """ generate 5G LLRin for LDPC decoder test using BPSK modulation
    """
    assert bgn in [1, 2]
    assert crcpoly in ['24A', '24B', '16']
    if bgn == 1:
        K = Zc * 22
        N = Zc * 66
    else:
        K = Zc * 10
        N = Zc * 50

    if crcpoly in ['24A', '24B']:
        crc_len = 24
    else:
        crc_len = 16

    #generate input bit
    inbits = np.random.randint(2, size = K-crc_len)
    blkandcrc = crc.nr_crc_encode(inbits, crcpoly)
    dn = nr_ldpc_encode.encode_ldpc(blkandcrc, bgn)      

    #LLR generation
    en = 1 - 2*dn #BPSK modulation, 0 -> 1, 1 -> -1
    fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
    
    #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
    noise_power = 10**(-snr_db/10)
    LLRin = 2*fn/noise_power
    

    return blkandcrc, dn, LLRin

if __name__ == "__main__":
    from scipy import io
    import os
    from py5gphy.crc import crc
    from py5gphy.ldpc import nr_ldpc_encode
    
    print("test LDPC soft decision decoder")
    from tests.ldpc import test_ldpc_decoder_soft
    file_lists = test_ldpc_decoder_soft.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_ldpc_decoder_soft.test_nr_ldpc_decode_BP(filename)
        test_ldpc_decoder_soft.test_nr_ldpc_decode_min_sum(filename)
        test_ldpc_decoder_soft.test_nr_ldpc_decode_NMS(filename)
        test_ldpc_decoder_soft.test_nr_ldpc_decode_OMS(filename)
        test_ldpc_decoder_soft.test_nr_ldpc_decode_mixed_min_sum(filename)



    pass