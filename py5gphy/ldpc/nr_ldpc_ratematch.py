# -*- coding:utf-8 -*-
import numpy as np
import math

def get_Er_ldpc(G, C, Qm, NL):
    """ get LDPC rate matching output sequence length for each coded block
     following to 38.212 5.4.2
     Er_list = get_Er_ldpc(G, C, Qm, NL)
    input:
        NL is the number of transmission layers that the transport block is mapped onto;
        Qm is the modulation order;
        G is the total number of coded bits available for transmission of the transport block;
        C number of code block // do not support CBGTI here
    output:
        Er_list: rate matching output len for each coded block
    """
    Er_list = [0] * C #init
    
    j = 0
    for r in range(C):
        if j <= (C - ((G/(NL*Qm)) % C) - 1):
             Er_list[j] = NL*Qm*math.floor(G/(NL*Qm*C))
        else:
             Er_list[j] = NL*Qm*math.ceil(G/(NL*Qm*C))
        j += 1

    return Er_list

def get_k0(Ncb, bgn, rv, Zc):
    """ get LDPC rate match k0 value folowing to 38.212 5.4.2
    k0 = get_k0(Ncb, bgn, rv, Zc)
    input:
        Ncb: circular buffer of length
        bgn: LDPC base graph
        rv: redundancy versions
        Zc: minimum value of Z in all sets of lifting sizes in Table 5.3.2-1
    output:
        k0: Starting position
    """
    assert rv in [0, 1, 2, 3]
    assert bgn in [1, 2]

    if rv == 0:
       k0 = 0
    elif rv == 1:
       if bgn == 1:
           k0 = math.floor(17*Ncb/(66*Zc)) * Zc      
       else:
           k0 = math.floor(13*Ncb/(50*Zc)) * Zc
    elif rv == 2:
       if bgn == 1:
           k0 = math.floor(33*Ncb/(66*Zc)) * Zc      
       else:
           k0 = math.floor(25*Ncb/(50*Zc)) * Zc
    elif rv == 3:
       if bgn == 1:
           k0 = math.floor(56*Ncb/(66*Zc)) * Zc      
       else:
           k0 = math.floor(43*Ncb/(50*Zc)) * Zc

    return k0


def ratematch_ldpc(dn, Ncb, E, k0, Qm):
    """ does one code block LDPC rate match by 38.212 5.4.2
    the function include bit selection and bit interleaving.
    the calculation of rate match related parameters(such as Nref, Er,k0..) is not in this function
    fe = ratematch_ldpc(dn, Ncb, E, k0, Qm)::
    input:
        dn: N length input sequence
        Ncb: circular buffer length
        E: rate match output length
        k0: starting position, refer to table 5.4.2.1-2
        Qm: modulation order in range[1,2,4,6,8]
    output:
        fe: E length sequence
    """

    N = dn.size
    assert N >= Ncb
    ek = np.zeros(E)
    #bit selection
    k = 0
    j = 0
    while k < E:
        if dn[(k0 + j) % Ncb] != -1: #not filler bit
            ek[k] = dn[(k0 + j) % Ncb]
            k += 1
        j += 1
    
    #bit interleaving
    d1 = ek.reshape(Qm, E // Qm)
    d2=d1.T
    fe = d2.reshape(1,E)[0]
    fe = fe.astype('i1')

    return fe

if __name__ == "__main__":
    from scipy import io
    import os
    print("test LDPC rate match")
    from tests.ldpc import test_ldpc_ratematch
    file_lists = test_ldpc_ratematch.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_ldpc_ratematch.test_nr_ldpc_ratematch(filename)