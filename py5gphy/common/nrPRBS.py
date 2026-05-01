# -*- coding:utf-8 -*-
import numpy as np

# TS 38.211 Section 5.2.1
def gen_nrPRBS(c_init, N):
    """ generate pseudo-random sequences refer to 38.211 5.2.1
    seq = gen_nrPRBS(c_init, N)
    input:
        c_init: init value
        N: output seq bit length
    output:
        seq: N length [0,1] sequence
    """
    assert N > 0
    Nc = 1600
    #initial x1 and x2 m-sequence
    x1_seq = np.zeros(Nc + N,'i1')
    x1_seq[0] = 1
    x2_seq = np.zeros(Nc + N,'i1')
    x2_seq[0:31] = [c_init >> i & 1 for i in range(31)]
    
    for m in range(Nc+N-31):
        x1_seq[m+31] = (x1_seq[m+3]+x1_seq[m]) % 2
        x2_seq[m+31] = (x2_seq[m+3]+x2_seq[m+2]+x2_seq[m+1]+x2_seq[m]) % 2
    
    seq = (x1_seq[1600:]+x2_seq[1600:]) % 2
    
    return seq


if __name__ == "__main__":
    from scipy import io
    import os
    
    print("test nr PBRS encoding")
    from tests.common import test_nrPRBS
    testcases = test_nrPRBS.get_testvectors()
    count = 1
    for ins,  expected in testcases:
        test_nrPRBS.test_nrPRBS(ins,  expected)
