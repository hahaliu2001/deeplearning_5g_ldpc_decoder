# -*- coding:utf-8 -*-
import numpy as np
import time

from py5gphy.crc import crc

def ldpc_decoder_BF(LLRin, H, crcpoly, B, L):
    """bit flipping decoding algorithm
       input:
            LLRin: N length LDPC decoder LLR input data
            H: M*N parity check matrix
            L: maximum iterations
            crcpoly: crc
            B: payload size after CRC
        output:
            ck: N size of decoded bits, 
            status: True or False
    """

    M = H.shape[0]
    N = H.shape[1]
    K = N - M

    #validate
    assert LLRin.size == N

    #generate A list and B list
    # A list contain sublist of all variable nodes index that connect to check node j
    # correspondent to each line of H
    #for examnple, A[0] is a list of variable nodes index that connect to check node 0
    A=[]
    for m in range(M):
        idxlist = list(np.where(H[m,:] == 1)[0])
        A.append(idxlist)
    
    # B list contain sublist of all check nodes index that connect to variable node i
    # correspondent to each column of H
    #for examnple, B[1] is a list of check nodes index that connect to variable node 1
    B=[]
    for n in range(N):
        idxlist = list(np.where(H[:,n] == 1)[0])
        B.append(idxlist)
    
    #hard coded LLRin to generate ck bit sequence, LLR >0 ->0, LLR<0 -> 1
    ck = np.copy(LLRin)
    ck[ck > 0] = 0
    ck[ck < 0 ] = 1

    #main loop
    for iter in range(L):
        S = (H @ ck.T) % 2
        #most of paper provide below S calculation which is exact the same with above matrix multiply.
        #for python, matrix multiply is much faster than below implementation
        #S = np.zeros(M,'i1')
        #for m in range(M):
        #    S[m] = sum(ck[A[m]]) % 2
        
        if not np.any(S): 
            #if S is all zero sequence, decoding success, return Tue
            return ck, True
        
        #process if S is not all zero
        
        #En = S @ H #some paper use this equation, but it only works for regular LDPC, 5G use non-regular LDPC matrix
        En = (2*S-1) @ H #this is correct equation
        #most of paper provide below S calculation which is exact the same with above matrix multiply.
        #for python, matrix multiply is much faster than below implementation
        #En = np.zeros(N, 'i1')
        #for n in range(N):
        #    En[n] = sum(2*S[B[n]]-1)
        max_value = np.max(En)

        #bit flip any ck bits with En value==max_value
        ck[En==max_value] = 1- ck[En==max_value]

        blk,status = crc.nr_crc_decode(ck, crcpoly)
        if status == 0:
            return blk, True
    
    #reach max iterations, and S still not all zero
    return blk, False

if __name__ == "__main__":
    from py5gphy.crc import crc
    from py5gphy.ldpc import nr_ldpc_encode
    from py5gphy.ldpc import ldpc_info
    import time

    print("test LDPC BF decoder")
    

