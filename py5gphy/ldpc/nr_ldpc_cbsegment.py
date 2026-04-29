# -*- coding:utf-8 -*-
import numpy as np

from py5gphy.crc import crc
from py5gphy.ldpc import ldpc_info

def ldpc_cbsegment(inbits, bgn):
    """ LDPC code block segment by 38.212 5.2.2
    cbs = ldpc_cbsegment(inbits, bgn)
    input:
        inbits: B length input bits
        bgn: base graph value 1 or 2
    output:
        cbs: C by K code blocks
        where C is number of code block, K is each code block length
        filler bits inserted to end of each code block is -1
    """
    B = inbits.size
    assert bgn in [1,2]

    #Get information of code block segments
    C, cbz, L, F, K, Zc = ldpc_info.get_cbs_info(B, bgn)

    cbs = -1 * np.ones((C, K),'i1')  # -1 is filler bits
    #Perform code block segmentation and CRC encoding
    if C == 1:
        cbs[0,0:cbz] = inbits
    else:
        for c in range(C):
            cb = inbits[c*cbz : (c+1)*cbz]
            blkandcrc = crc.nr_crc_encode(cb, '24B', 0)
            cbs[c,0 : cbz+L] = blkandcrc
    return cbs, Zc


if __name__ == "__main__":
    from scipy import io
    import os
    print("test LDPC CB segment")
    from tests.ldpc import test_ldpc_cb_segment
    file_lists = test_ldpc_cb_segment.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_ldpc_cb_segment.test_nr_ldpc_cb_segment(filename)
    