# -*- coding:utf-8 -*-
import numpy as np
import scipy as sp
import time

from py5gphy.ldpc import ldpc_info

def encode_ldpc(ck, bgn):
    """ LDPC encode following to TS38.212 5.3.2
    dn = encode_ldpc(ck, bgn)
    input: 
        ck: K length code block
        bgn: base graph value 1 or 2
    output:
        dn: N length LDPC encoded sequence
    
    """
    assert bgn in [1,2]
    K = ck.size
    if bgn == 1:
        Zc = K // 22
        N = 66 * Zc
    else:
        Zc = K // 10
        N = 50 * Zc
    
    #step 1: Find the set with index iLS in Table 5.3.2-1 which contains Zc
    iLS = ldpc_info.find_iLS(Zc)
    assert iLS < 8

    #step 2: set dn[0:K-2*Zc]
    dn = -np.ones(N,'i1')
    for k in range(2*Zc,K):
        if ck[k] == -1:
            ck[k] = 0
        else:
            dn[k - 2*Zc] = ck[k]
    
    #step 3: generate wn
    #start = time.time()
    H = ldpc_info.getH(Zc, bgn, iLS)
    #print("getH elpased time: {:6.2f}".format(time.time() - start))

    wn = _gen_ldpc_parity_bit(H, ck, Zc, K, N,bgn)

    #step 4
    for k in range(K,N + 2*Zc):
        dn[k - 2*Zc] = wn[k - K]

    return dn

def _gen_ldpc_parity_bit(H, ck, Zc, K, N,bgn):
    """ generate LDPC paraty bit"""
    choose_opt = 1

    if choose_opt == 0:
        #below code is not used. it does LDPC encoder without optimization
        # for bgn1, 46ZcX66Zc H matrix can be split to 46ZcX22Zc H1 matrix and 46ZcX46Zc H2 matrix
        # for bgn2, 42ZcX52Zc H matrix can be split to 42ZcX10Zc H1 matrix and 42ZcX42Zc H2 matrix
        # the matrix H*[c w]' = 0 -> H1*c + H2*w = 0 -> H2*w = -H1*c -> w = H2_inverse *(-H1*c)

        Hrowsize = H.shape[0]
        Hcolsize = H.shape[1]
        H1 = H[:,0:Hcolsize-Hrowsize]
        H2 = H[:,Hcolsize-Hrowsize:Hcolsize]

        #start = time.time()
        L1 = -H1 @ ck.T  #matrix multiply
        #print("H1 @ck.T elpased time: {:6.2f}".format(time.time() - start))

        #start = time.time()
        #this H2\L1 calcualtion takes very long of the elapsed time
        #for Zc = 382m bgn=2, getH and -H1 @ ck.T operation each take around 0.20 second, 
        # while below calculation takes 51.43 seconds
        outd = np.linalg.solve(H2, L1) #outd = H2\L1;
        #print("H1\L1 elpased time: {:6.2f}".format(time.time() - start))
        ref_wn = np.round(outd) % 2
        ref_wn = ref_wn.astype('i1')

        #return wn
    
    #optimization based on 'Flexible 5G New Radio LDPC Encoder Optimized for High Hardware Usage Efficiency'
    #https://www.mdpi.com/2079-9292/10/9/1106
    #can also refer to docs/LDPC_encoder_optimization.pdf

    #start = time.time() below code use at most 0.4second
    #for optimized method
    A = H[0:4*Zc, 0:K]
    B = H[0:4*Zc, K:K+4*Zc]
    C = H[4*Zc:N+2*Zc, 0:K+4*Zc]

    L1 = A @ ck.T
    L1=L1.reshape((4,-1))
    L2 = np.sum(L1,axis=0) % 2
    if bgn == 1:
        zeros_idx = np.nonzero(B[Zc,0:Zc])[0][0]
        pc1 = np.roll(L2,zeros_idx)
        pc2 = (L1[0,:] + B[0:Zc,0:Zc] @pc1.T) % 2
        pc4 = (L1[3,:] + B[3*Zc:4*Zc,0:Zc] @pc1.T) % 2
        pc3 = (L1[2,:] + B[2*Zc:3*Zc,3*Zc:4*Zc] @pc4.T) % 2
    else:
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
    #print("encode_ldpc elpased time: {:6.2f}".format(time.time() - start))
    return wn


if __name__ == "__main__":
    print("test LDPC encode")
    from tests.ldpc import test_ldpc_encoder
    file_lists = test_ldpc_encoder.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_ldpc_encoder.test_nr_ldpc_encode(filename)