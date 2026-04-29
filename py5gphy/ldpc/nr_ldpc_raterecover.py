# -*- coding:utf-8 -*-

import numpy as np
import math

def raterecover_ldpc(LLr_fe, Ncb, N, k0, Qm,Zc,K_apo,K):
    """ does one code block LDPC rate recover by 38.212 5.4.2
    the function include bit de-interleaving and bit selection.
    the calculation of rate match related parameters(such as Nref, Er,k0..) is not in this function
    dn = raterecover_ldpc(fe, E, Qm):
    input:
        LLr_fe: E length input LLR sequence
        N: rate recover output length
        Ncb: circular buffer length
        E: rate match output length
        k0: starting position, refer to table 5.4.2.1-2
        Qm: modulation order in range[1,2,4,6,8]
        K_apo: first filler bit location defined in 5.2.2 CB segment,
             LDPC rate matching filler position = [K_apo: K] - 2Zc
    output:
        LLr_dn: N length sequence
    """

    E = LLr_fe.size
    #38.212 5.4.2.2 de-bit interleaving
    d1=LLr_fe.reshape(E // Qm, Qm)
    d2=d1.T
    LLr_ek = d2.reshape(E)

    max_LLR = np.max(np.abs(LLr_fe)) * 10

    #38.212 5.4.2.1 de-bit selection
    #refer to 5.3.2, filler bits in dn is left shift by 2Zc
    dn_filler_bits_pos = np.arange(K_apo,K) - 2*Zc
    
    #for de-RM, if no repetition, set 0 to not-transmitted data
    #if repetition, average of all re-tx data
    size = Ncb - dn_filler_bits_pos.size #number of non-filler bits in dn seq
    rep_num = int(np.ceil(E/size)) #maximum number of repetition

    tmp_buf = np.zeros((rep_num,Ncb)) #rearange fe 
    rep_buf = 10000*np.ones(Ncb) #save repetition times for each bit
    rep_idx = -1

    k = 0
    j = 0
    while k < E:
        pos = (k0+j) % Ncb
        if pos == k0:
            rep_idx += 1 #increase rep num
        
        if pos not in dn_filler_bits_pos:
            tmp_buf[rep_idx,pos] = LLr_ek[k]
            k += 1
        
        if rep_buf[pos] == 10000:
            rep_buf[pos] = 1
        else:
            rep_buf[pos] += 1
        j += 1

    LLr_dn = np.zeros(N)
    LLr_dn[0:Ncb] = np.sum(tmp_buf,axis=0)/rep_buf
    LLr_dn[dn_filler_bits_pos] = max_LLR
    return LLr_dn            

if __name__ == "__main__":
    from scipy import io
    import os
    print("test LDPC rate recover")
    from tests.ldpc import test_ldpc_raterecover
    file_lists = test_ldpc_raterecover.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_ldpc_raterecover.test_nr_ldpc_raterecover(filename)
        
    pass