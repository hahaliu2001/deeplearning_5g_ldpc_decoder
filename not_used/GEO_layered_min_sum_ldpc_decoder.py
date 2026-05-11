# -*- coding:utf-8 -*-

import numpy as np

from LDPC_parameter import gen_LDPC_parameter
from py5gphy.crc import crc

def GEO_layered_min_sum_ldpc_decoder(LLR_bits, code_rate, LDPC_decoder_config):
    """
    the code is used to decode the LLR bits using layered min-sum LDPC decoder 
    """
    
    E = len(LLR_bits) # codeword length after LDPC encoding and rate matching, which is also the input bit size E
    B = int(E * code_rate) # information bit size before LDPC encoding

    #get information used for LDPC rate recover and LDPC decoding, 
    [Es, N, Zc, F, H], _ = gen_LDPC_parameter(B, code_rate)
    assert Es == E, "The codeword length after LDPC encoding and rate matching should be the same as the input LLR bit size of LDPC decoder"
    
    #rate recovery
    #format of output of rate recovery:
    # first 2*Zc bits are for the punctured information bits, which are set to 0
    # next B-2*Zc bits are the LLR bits for the information bits
    # next F bits are for the filler bits, which are set to max_LLR
    # last E-(B-2*Zc) bits are for the parity bits, which are the remaining LLR bits after removing the information bits 
    
    max_LLR = np.max(np.abs(LLR_bits)) * 10 # set the LLR for filler bits to be a large value to ensure they are decoded as 0
    #max_LLR = 127
    
    LLr_dn = np.zeros(2*Zc+N,dtype=np.float32)
    LLr_dn[2*Zc:B] = LLR_bits[0:B-2*Zc]
    LLr_dn[B:B+F] = max_LLR
    LLr_dn[B+F:(B+F)+E-(B-2*Zc)] = LLR_bits[B-2*Zc:]
    
    # get LDPC decoder configuration
    L = LDPC_decoder_config["L"]
    alpha = LDPC_decoder_config["alpha"]
    beta = LDPC_decoder_config["beta"]

    decoded_bits, status = _layered_min_sum_decoder(LLr_dn, B, H, L=L, alpha=alpha, beta=beta)
    
    return decoded_bits[0:B], status


def _layered_min_sum_decoder(LLRin, B, H, L=16, alpha=1, beta=0):
    """ layered min-sum LDPC decoder
    input: 
        LLRin: N length LDPC decoder LLR input data
        B: information bit size before LDPC encoding
        H: LDPC matrix, 
        L: iteration size
        alpha: (0:1] used for normalized min-sum. 1 means no normalized
        beta: >=0 used for offset min-sum, 0 means no offset
    output:
        decoded_bits: B length LDPC decoded bit sequence
        status: decoding status, true for success, false for failure
    
    """

    #for soft decision decoder
    M = H.shape[0] #total number of layers, which is also the number of parity check equations
    N = H.shape[1] #total number of variable nodes, which is also the code block size after LDPC encoding

    #validate
    assert LLRin.size == N

    #generate A list and B list
    # A list contain sublist of all variable nodes index that connect to check node j
    # correspondent to each line of H
    #for examnple, A[0] is a list of variable nodes index that connect to check node 0
    Alist=[]
    for m in range(M):
        idxlist = list(np.where(H[m,:] == 1)[0])
        Alist.append(idxlist)
        
    #init Lq, LQ, Lr
    LQ = LLRin #decoded variable bits LLR

    #Variable nodes to checking nodes LLR, Lq[m,n] = 0 if H[m,n]=0
    #m is check node index, n is variable index, 
    Lq = np.zeros((M,N)) 
    
    #checking nodes to variable nodes, Lr[i,j] = 0 if H[i,j]=0
    Lr = np.zeros((M,N))     

    for iter in range(L):        
        #for each layer, where M is number of check nodes, which is also the number of layers in layered decoding
        for m in range(M):
            # A[m] is the list of variable node indices that connect to check node m

            #generate layer m variable node -> check node LLR
            sel_Lq = LQ[Alist[m]] - Lr[m,Alist[m]]
            
            #update Lr for layer m using min-sum processing
            Lr[m,Alist[m]] = _min_sum_process(sel_Lq, Alist[m], alpha, beta)
        
            #update LQ with index that connect to layer m
            LQ[Alist[m]] = sel_Lq + Lr[m,Alist[m]]

        if 1:
            ck, status = _do_parity_check(LQ, H)
            if status:
                return ck[0:B-24], status
        else:
            #support CRC check after each iteration, if CRC check pass, return the decoded bits and status
            ck = np.zeros(B,'i1') #hard coded bits
            ck[LQ[0:B] < 0] = 1
            if not np.any(ck): 
                #if all variables bits are 0, decoding failed, return False
                return ck, False
            blk,status = crc.nr_crc_decode(ck, '24A')
            if status==0:
                return blk, status
        
        
    return ck[0:B], status

def _min_sum_process(sel_Lq, V_m, alpha, beta):
    """ mixed min-sim processing
    alpha = 1, beta =0: min-sum
    alpha < 1, beta = 0: normalized min-sum
    alpha = 1, beta >0: offset min-sum
    alpha <1, beta >0: mixed min-sum
    """
    sel_Lr = np.zeros(sel_Lq.shape, dtype=np.float32)
    zero_value_list = list(np.where(sel_Lq==0)[0])
    
    if len(zero_value_list) == 0:
        #normal case, no zero value
        sign_prod = np.prod(np.sign(sel_Lq))
        sorted_abs_v = np.sort(np.abs(sel_Lq))

        first_min = sorted_abs_v[0]
        second_min = sorted_abs_v[1]
        for idx, n in enumerate(V_m):
            if np.abs(sel_Lq[idx]) == first_min:
                minv = second_min
            else:
                minv = first_min
                
            sel_minv = max(minv - beta, 0) #offset min-sum  
            sel_Lr[idx] = alpha * sign_prod * np.sign(sel_Lq[idx]) * sel_minv
    elif len(zero_value_list) == 1:
        #only one zero value, only zero value related Lr is non-zero
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
        sel_Lr[zero_idx] = alpha * sign_prod *sel_minv 
    else:
        #>=2 zero values. all sel_Lr are zero
        pass
    
    return sel_Lr

def _do_parity_check(LQ, H):
    N = LQ.size
    #hard decision
    ck = np.zeros(N,'i1') #hard coded bits
    ck[LQ >= 0] = 0
    ck[LQ < 0] = 1
    
    if not np.any(ck): 
        #if all variables bits are 0, decoding failed, return False
        return ck, False
    
    #parity check first
    S = (H @ ck.T) % 2    
    if not np.any(S): 
        #if S is all zero sequence, decoding success, return Tue
        return ck, True
    else:
        return ck, False
    

if __name__ == "__main__":
    from scipy import io
    import os
    from GEO_LDPC_encoder import Xona_LDPC_encoder
    
    print("test layered min-sum LDPC rate recover and decode")
    
    #LDPC decoder config
    LDPC_decoder_config = {
        "L": 32, # iteration number
        "alpha": 0.7, # used for normalized min-sum. 1 means no normalized
        "beta": 0.3, # used for offset min-sum, 0 means no offset
    }

    LDPC_decoder_config2 = {
        "L": 32, # iteration number
        "algo": "min-sum", # decoding algorithm, one of ["BF", "BP", "min-sum"]
        "alpha": 0.7, # used for normalized min-sum. 1 means no normalized
        "beta": 0.3, # used for offset min-sum, 0 means no offset
    }

    #noise power in dB
    noise_power_db = -20
    noise_power = 10 ** (noise_power_db / 10)
    for code_rate in [1/4, 1/5]:
    #for code_rate in [1/4]:
        for B in [320,400,480,640,800,960,1280,3840,256,384,512,768]:
        #for B in [320,3840]:
            print(f"Testing code rate: {code_rate}, Testing information bit length: {B}")
            info_bits = np.random.randint(0, 2, size=B-24) # generate random information bits of size B
            
            xona_encoded_bits = Xona_LDPC_encoder(info_bits, code_rate)
            
            #BPSK modulation of xona_encoded_bits
            mod_bits = 1 - 2 * xona_encoded_bits  # BPSK modulation

            #add noise to the encoded bits, and convert them to LLR bits
            noise = np.sqrt(noise_power) * np.random.randn(len(xona_encoded_bits))
            received_bits = mod_bits + noise
            LLR_bits = 2 * received_bits / noise_power # assuming BPSK modulation
                        
            decoded_bits, status = GEO_layered_min_sum_ldpc_decoder(LLR_bits, code_rate, LDPC_decoder_config)
    
            print("LDPC decode status: ", status)
            assert np.array_equal(decoded_bits[0:B-24], info_bits), "The encoded bits from sionna and Xona should be the same"
    
    pass