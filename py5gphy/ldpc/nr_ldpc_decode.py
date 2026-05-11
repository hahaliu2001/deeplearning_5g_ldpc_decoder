# -*- coding:utf-8 -*-
import numpy as np
import scipy as sp
import time

from py5gphy.ldpc import ldpc_info
from py5gphy.ldpc import ldpc_decoder_bit_flipping
from py5gphy.crc import crc
from py5gphy.ldpc import nr_ldpc_encode

def nr_decode_ldpc(LLRin, Zc, bgn, crcpoly,B, ldpcdecoder_config={"L": 32,"algo": "min-sum","alpha": 0.8,"beta": 0}):
    """ LDPC decode following to TS38.212 5.3.2
    ck = encode_ldpc(dn, bgn)
    input: 
        LLRin: N length LDPC decoder LLR input data
        bgn: base graph value 1 or 2
        crcpoly: one of ['24A','24B','16']
        B: payload size after CRC
        algo: ["BF", "BP", "min-sum"]
    output:
        ck: K length LDPC decoded sequence
    
    """
    L = ldpcdecoder_config["L"]
    algo = ldpcdecoder_config["algo"]
    alpha = ldpcdecoder_config["alpha"]
    beta = ldpcdecoder_config["beta"]

    assert bgn in [1,2]
    assert algo in ['BF', 'BP', 'min-sum', 'layered-BP', 'layered-min-sum']
    
    if bgn == 1:
        N = Zc * 66
        K = Zc * 22
    else:
        N = Zc * 50
        K = Zc * 10
    
    assert N == LLRin.size
    
    #step 1: Find the set with index iLS in Table 5.3.2-1 which contains Zc
    iLS = ldpc_info.find_iLS(Zc)
    assert iLS < 8

    #step 2 : get H
    H = ldpc_info.getH(Zc, bgn, iLS)

    #add punctured 2*Zc information bits to get full ldpc decoder input
    newLLRin = np.concatenate([np.zeros(2*Zc), LLRin])

    if algo in [ "BF"]:
         blkWithoutCrc, status = ldpc_decoder_bit_flipping.ldpc_decoder_BF(newLLRin, H, crcpoly,B, L)
    elif algo in ["BP", "min-sum"]:
        blkWithoutCrc, status =  decode_ldpc(newLLRin, H, L, crcpoly,B,algo, alpha, beta)
    elif algo in ["layered-BP", "layered-min-sum"]:
        blkWithoutCrc, status =  layered_decode_ldpc(newLLRin, H, crcpoly,B,L, algo, alpha, beta)
    
    return blkWithoutCrc, status

def decode_ldpc(LLRin, H, L, crcpoly, B, algo='min-sum', alpha=1, beta=0):
    """ LDPC decoder, support BF(bit flipping), BP(belief propagation), min-sum algorithms
    BP is also named sum-product algorithm
    input: 
        LLRin: N length LDPC decoder LLR input data
        H: LDPC matrix, 
        L: iteration size
        crcpoly: crc type
        B: payload size after CRC
        algo: one of ["BF", "BP", "min-sum"]
        alpha: (0:1] used for normalized min-sum. 1 means no normalized
        beta: >=0 used for offset min-sum, 0 means no offset
    output:
        ck: N length LDPC decoded sequence
        status: True or False
    """     
    
    #for soft decision decoder
    M = H.shape[0]
    N = H.shape[1]
    
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
    
    # B list contain sublist of all check nodes index that connect to variable node i
    # correspondent to each column of H
    #for examnple, B[1] is a list of check nodes index that connect to variable node 1
    Blist=[]
    for n in range(N):
        idxlist = list(np.where(H[:,n] == 1)[0])
        Blist.append(idxlist)
    
    #init Lq, LQ, Lr
    LQ = LLRin.copy() #decoded variable bits LLR

    #Variable nodes to checking nodes LLR, Lq[m,n] = 0 if H[m,n]=0
    #m is check node index, n is variable index, 
    Lq = H*LLRin  
    
    #checking nodes to variable nodes, Lr[i,j] = 0 if H[i,j]=0
    Lr = np.zeros((M,N)) 
    
    for iter in range(L):
        #variable nodes to checking nodes update
        for m in range(M):
            sel_Lq = Lq[m,Alist[m]]
            
            if algo == 'BP':
                Lr[m,Alist[m]] = _BP_process(sel_Lq, Alist[m])
            else:
                Lr[m,Alist[m]] = _min_sum_process(sel_Lq, Alist[m], alpha, beta)
        
        #update LQ
        LQ = LLRin + Lr.sum(axis=0)

        #update Lq
        for n in range(N):
            for m in Blist[n]:
                Lq[m,n] = LQ[n] - Lr[m,n]

        ck = np.zeros(B,'i1') #hard coded bits
        ck[LQ[0:B] < 0] = 1
        blk,status = crc.nr_crc_decode(ck, crcpoly)
        if status==0:
            return blk, status
    
    return blk, False

def layered_decode_ldpc(LLRin, H, crcpoly, B, L, algo='min-sum', alpha=1, beta=0):
    """ LDPC decoder, support BP(belief propagation), min-sum algorithms
    BP is also named sum-product algorithm
    input: 
        LLRin: N length LDPC decoder LLR input data
        H: LDPC matrix, 
        L: iteration size
        algo: one of ["BF", "BP", "min-sum"]
        alpha: (0:1] used for normalized min-sum. 1 means no normalized
        beta: >=0 used for offset min-sum, 0 means no offset
    output:
        ck: N length LDPC decoded sequence
        status: True or False
    """     
    
    #for soft decision decoder
    M = H.shape[0]
    N = H.shape[1]
    
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
    LQ = LLRin.copy() #decoded variable bits LLR

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
            if algo == 'layered-min-sum':
                Lr[m,Alist[m]] = _min_sum_process(sel_Lq, Alist[m], alpha, beta)
            elif algo == 'layered-BP':
                Lr[m,Alist[m]] = _BP_process(sel_Lq, Alist[m])
            else:
                assert False, "unsupported algorithm for layered decoding, should be one of 'layered-BP' and 'layered-min-sum'"
        
            #update LQ with index that connect to layer m
            LQ[Alist[m]] = sel_Lq + Lr[m,Alist[m]]

        #support CRC check after each iteration, if CRC check pass, return the decoded bits and status
        ck = np.zeros(B,'i1') #hard coded bits
        ck[LQ[0:B] < 0] = 1
        blk,status = crc.nr_crc_decode(ck, crcpoly)
        if status==0:
            return blk, status
    
    return ck, False

def _BP_process(sel_Lq, V_m):
    #calculate Lr with BP processing
    sel_Lr = np.zeros(sel_Lq.shape, dtype=np.float32)

    zero_value_list = list(np.where(sel_Lq==0)[0])
    
    if len(zero_value_list) == 0:
        #normal case, no zero value
        tanh_Lq = np.tanh(sel_Lq/2)
        #tanh_Lq[tanh_Lq==0] = 10**(-22)
        prod_v = np.prod(tanh_Lq)
        for idx, n in enumerate(V_m):
            #tmp2 = np.prod(sel_Lq[0:idx]) * np.prod(sel_Lq[idx+1:])
            tmp2 = prod_v/tanh_Lq[idx]
            if tmp2 >= 1:
                #print("Warning: tanh value is greater than or equal to 1, which may cause numerical issue for BP processing. Set Lr to maximum value")
                sel_Lr[idx] = 2 * 19.07 #atanh(1) = infinite, set 19.07 as maximum value
            elif tmp2 <= -1:
                #print("Warning: tanh value is less than or equal to -1, which may cause numerical issue for BP processing. Set Lr to minimum value")
                sel_Lr[idx] = -2 * 19.07 #atanh(-1) = negative infinite, set 19.07 as maximum value
            else:
                sel_Lr[idx] = 2 * np.arctanh(tmp2)
    elif len(zero_value_list) == 1:
        tanh_Lq = np.tanh(sel_Lq/2)
        #only one zero value, only zero value related Lr is non-zero
        zero_idx = zero_value_list[0]
        
        #np.prod(tanh_Lq[m:m) return 1, so we can directly calculate the product of all tanh_Lq without zero value and get the same result
        sel_Lr[zero_idx] = np.prod(tanh_Lq[0:zero_idx]) * np.prod(tanh_Lq[zero_idx+1:])
    else:
        #>=2 zero values. all Lr are zero
        pass    
    
    return sel_Lr

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
        #>=2 zero values. all Lr are zero
        pass
    
    return sel_Lr

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

def _do_parity_check(LQ, H):
    """_summary_ This function is used for parity check in LDPC decoding, 
        which can be used for early stopping. 
        But we use CRC check for early stopping in our implementation, so this function is not used in the current code.

    Args:
        LQ (_type_): _description_
        H (_type_): _description_

    Returns:
        _type_: _description_
    """
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