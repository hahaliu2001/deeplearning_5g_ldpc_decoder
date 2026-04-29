# -*- coding:utf-8 -*-
import numpy as np
import time

def ldpc_decoder_BF(LLRin, H, L):
    """bit flipping decoding algorithm
       input:
            LLRin: N length LDPC decoder LLR input data
            H: M*N parity check matrix
            L: maximum iterations
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
    
    #reach max iterations, and S still not all zero
    return ck, False

def for_test_ldpc_encoder(K, H, snr_db):
    ck = np.random.randint(2, size=K)
    #ldpc encoder to get parity bits
    Hrowsize = H.shape[0]
    Hcolsize = H.shape[1]
    H1 = H[:,0:Hcolsize-Hrowsize]
    H2 = H[:,Hcolsize-Hrowsize:Hcolsize]

    L1 = -H1 @ ck.T  #matrix multiply
    outd = np.linalg.solve(H2, L1) #outd = H2\L1;
    wn = np.round(outd) % 2
    wn = wn.astype('i1')

    dn = np.concatenate((ck, wn))

    en = 1 - 2*dn #BPSK modulation, 0 -> 1, 1 -> -1
    fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
    
    #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
    noise_power = 10**(-snr_db/10)
    LLRin = 2*fn/noise_power

    return dn, LLRin


if __name__ == "__main__":
    from py5gphy.crc import crc
    from py5gphy.ldpc import nr_ldpc_encode
    from py5gphy.ldpc import ldpc_info
    import time

    print("test LDPC BF decoder")
    from tests.ldpc import test_ldpc_decoder_BF
    file_lists = test_ldpc_decoder_BF.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_ldpc_decoder_BF.test_nr_ldpc_decode_BF(filename)

    H = np.array([[1,1,0,1,0,0],
                  [0,1,1,0,1,0],
                  [1,0,0,0,1,1],
                  [0,0,1,1,0,1]])
    dn, LLRin = for_test_ldpc_encoder(2, H, 255)

    dn_decoded, status = ldpc_decoder_BF(LLRin, H, 8)
    assert status == True
    assert np.array_equal(dn,dn_decoded)

    #test if one bit flip, and decoder can correctly decode it
    for m in range(H.shape[1]):
        dn, LLRin = for_test_ldpc_encoder(2, H, 255)
        LLRin[m] = -LLRin[m]
        dn_decoded, status = ldpc_decoder_BF(LLRin, H, 8)
        assert status == True
        assert np.array_equal(dn,dn_decoded)

    #test two bits flip, the decoder can not correctly decode it, but still return True
    failed_count = 0
    for m in range(H.shape[1]):
        dn, LLRin = for_test_ldpc_encoder(2, H, 255)
        LLRin[m] = -LLRin[m]
        LLRin[(m+1) % LLRin.size] = -LLRin[(m+1) % LLRin.size]
        dn_decoded, status = ldpc_decoder_BF(LLRin, H, 8)
        assert status == True
        if not np.array_equal(dn,dn_decoded):
            failed_count += 1
    print("2 bit flip failed test count {} over total {} test".format(failed_count,H.shape[1]))

    #choose 5G LDPC parameters, test pure LDPC decoder
    L = 32
    N = 660
    bgn = 1
    max_count = 2000
    
    H, K, Zc = ldpc_info.gen_ldpc_para(N, bgn)
    
    snr_range = np.arange(1, 5, 0.5).tolist()  
    snr_range = np.arange(4.5, 6, 0.5).tolist()  
        
    for snr_db in snr_range:
        failed_count = 0
        false_count = 0 # ldpc decoder status = True, but bit sequence can not match
        start = time.time()
        for m in range(max_count):
            dn, LLRin = for_test_ldpc_encoder(K, H, snr_db)
            dn_decoded, status = ldpc_decoder_BF(LLRin, H, L)
            if not np.array_equal(dn,dn_decoded):
                failed_count += 1
                if status == True:
                    false_count += 1

        #print("elpased time: {:6.6f}".format(time.time() - start))
        print("BF,N={},snr={} L= {},BF BLER={:3.2f}%,false_count = {:3.6f}%".format(N, snr_db, L, failed_count/max_count*100,false_count/max_count*100))
        
        
    #test 5G LDPC encoder and decoder
    L = 32
    N = 660
    bgn = 1
    crcpoly = '24A'
    max_count = 300

    H, K, Zc = ldpc_info.gen_ldpc_para(N, bgn)
    snr_range = np.arange(2, 5, 0.5).tolist()  
        
    for snr_db in snr_range:
        failed_count = 0
        for m in range(max_count):
            inbits = np.random.randint(2, size=K-24)
            blkandcrc = crc.nr_crc_encode(inbits, crcpoly)
            dn = nr_ldpc_encode.encode_ldpc(blkandcrc, bgn)      
            en = 1 - 2*dn #BPSK modulation, 0 -> 1, 1 -> -1
            fn = en + np.random.normal(0, 10**(-snr_db/20), dn.size) #add noise
    
            #LLR is log(P(0)/P(1)) = (-(x-1)^2+(x+1)^2)/(2*noise_power) = 4x/(2*noise_power) = 2x/noise_power
            noise_power = 10**(-snr_db/10)
            LLRin = 2*fn/noise_power

            #dn occupy c[2Zc:K], w[0:N+2Zc-K], LLRin need add 2Zc '0' ahead of data
            LLRin = np.concatenate((np.zeros(2*Zc), LLRin))

            dn_decoded, status = ldpc_decoder_BF(LLRin, H, L)
            if not np.array_equal(blkandcrc,dn_decoded[0:K]):
                failed_count += 1
                if status == True:
                    false_count += 1
        print("N={},snr={} L= {},BF BLER={:3.2f}%,false_count = {:3.6f}%".format(N, snr_db, L, failed_count/max_count*100,false_count/max_count*100))
    pass

