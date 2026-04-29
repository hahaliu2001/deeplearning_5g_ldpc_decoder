# -*- coding:utf-8 -*-
import numpy as np

def nr_crc_encode(blk, poly, mask=0):
    """blkandcrc = nr_crc_encode(blk,poly,mask=0)
    calculate CRC for input blk and return a copy of blk and CRC parity bits.
      blk: 1xA int type 0/1 bit array
      poly: ('6','11','16','24A','24B','24C')
      mask: integer value usually RNTI, to xor mask CRC bits. mask value is applied to CRC bits MSB first
           if mask value is greater than CRC bit length, then LSB 'L' bits are used for mask
      blkandcrc: output 1x(A+L) int8 type 0/1 bit array
      example:
      blkandcrc = nr_crc_encode(np.array([1,1,1,1,0,0,0,0]), '16', 12345)
    """
    # make sure all input data value are 0 or 1
    assert (not np.any(np.nonzero(blk < 0))) and (
        not np.any(np.nonzero(blk > 1))
    )

    crcpoly = _get_crcpoly(poly)
    L = crcpoly.size  # CRC size

    A = blk.size  #  input data size

    blkandcrc = np.zeros(A + L, "i1")
    blkandcrc[0:A] = blk

    remainder = blkandcrc[0:L]
    for idx in range(0, A):
        firstbit = remainder[0]
        remainder = np.append(remainder[1:], blkandcrc[idx + L])
        if firstbit:
            remainder = np.bitwise_xor(remainder, crcpoly)

    if mask:
        bitseq = [mask >> (23 - i) & 1 for i in range(0, 24)]  #'MSB first
        bitseq = bitseq[24 - L : 24]  # select L LSB bits
        remainder = np.bitwise_xor(remainder, bitseq)

    blkandcrc[A:] = remainder
    return blkandcrc

def nr_crc_decode(blkandcrc, poly, mask=0):
    """[blk, err] = nr_crc_decode(blk,poly,mask=0)
    check CRC error for input blk with associated CRC bits, and return a copy of blk and CRC error .
      blkandcrc: 1x(A+L) int type 0/1 bit array, data + CRC bits
      poly: ('6','11','16','24A','24B','24C')
      mask: integer value usually RNTI, to xor mask CRC bits. mask value is applied to CRC bits MSB first
           if mask value is greater than CRC bit length, then LSB 'L' bits are used for mask
      blk: output 1xA int8 type 0/1 bit array
      err: 0: no error, 1:error
      example:
      blk, err = nr_crc_decode(np.array([1,1,1,1,0,0,0,0]), '16', 12345)
    """
    # make sure all input data value are 0 or 1
    assert (not np.any(np.nonzero(blkandcrc < 0))) and (
        not np.any(np.nonzero(blkandcrc > 1))
    )

    blkandcrc = blkandcrc.astype('i1')
    
    crcpoly = _get_crcpoly(poly)
    L = crcpoly.size  # CRC size

    A = blkandcrc.size - L  # input data size

    blk = np.zeros(A, "i1")
    blk = blkandcrc[0:A]

    remainder = blkandcrc[0:L]
    for idx in range(0, A):
        firstbit = remainder[0]
        remainder = np.append(remainder[1:], blkandcrc[idx + L])
        if firstbit:
            remainder = np.bitwise_xor(remainder, crcpoly)

    if mask:
        bitseq = [mask >> (23 - i) & 1 for i in range(0, 24)]  #'MSB first
        bitseq = bitseq[24 - L : 24]  # select L LSB bits
        remainder = np.bitwise_xor(remainder, bitseq)

    # remainder should be all zero if CRC pass
    if np.count_nonzero(remainder):
        err = 1
    else:
        err = 0

    return blk, err

def _get_crcpoly(poly):
    """return CRC polynomial"""

    # fmt: off
    match poly.upper():
        case '6':
            crcpoly = np.array([1, 0, 0, 0, 0, 1],'i1')
        case '11':
            crcpoly = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],'i1')
        case '16':
            crcpoly = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],'i1')
        case '24A':
            crcpoly = np.array([1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],'i1')
        case '24B':
            crcpoly = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],'i1')
        case '24C':
            crcpoly = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],'i1')
        case _:
            assert False
    # fmt: on

    return crcpoly

def gen_crc_mask(K, padCRC, rnti):
    """ generate CRC mask: CRC value with input [np.ones(24), np.zeros(K)] 
        K: information bit size, not include CRC size
        padCRC: 1: padding 24 '1' ahead of CRC data for CRC encoder which is used by DL DCI
                0: no padding which is used by DL BCH
        rnti: used by DL DCI CRC mask
    """
    if padCRC == 1:
        #get CRC mask: CRC value with [np.ones(24), np.zeros(K-24)] 
        data = [1]*24 + list(np.zeros(K , 'i1'))
        blkandcrc = nr_crc_encode(np.array(data, 'i1'), '24C', rnti)
        crc_mask = blkandcrc[-24:]
    else:
        crc_mask = np.zeros(24, 'i1')
    return crc_mask

def check_distributed_CRC24C(ck, crc_bit_loc, crcidx_matrix,crc_mask):
    """ verify distributed CRC 24C at one CRC bit location
        ck:  bit sequence without polar interleaving
        crc_bit_loc: CRC bit location from 0 to 23
        
    """
    K = np.array(ck).size
    
    #get bits locations that is used to calculate CRC bit=phase
    column = crcidx_matrix[:,crc_bit_loc]
    locs = column[column>=0]
    
    #add data together with mod 2
    cal_crc_bit = ((sum(ck[locs]) % 2) + crc_mask[crc_bit_loc]) %2

    polar_crc_bit = ck[K-24+crc_bit_loc] #CRC bit calculated by polar decoder

    return cal_crc_bit == polar_crc_bit
    
def gen_CRC24C_encoding_matrix(K):
    """ gen CRC encoding matrix for CRC24C
        K: informartion bit size, not include CRC 
        crcidx_matrix: CRC index matrix, -1 means not involved
            each column show that CRC input data indexs that are involed for this bit CRC calculation
    """
    L = 24  # CRC size
    crcidx_matrix = np.zeros((K,L),'i2')
    for n in range(K):
        inbits = np.zeros(K,'i1')
        inbits[n] = 1
        blkandcrc = nr_crc_encode(inbits, '24C')
        crc_bits = blkandcrc[-L:]
        crcidx_matrix[n,:] = crc_bits*(n+1)   

    crcidx_matrix = crcidx_matrix -1
    return crcidx_matrix


if __name__ == "__main__":
    print("test crc encode")
    from tests.crc import test_crc
    testlists = [([1,1,1,1], 0, [1, 1, 1, 1, 0, 0, 1, 0, 1, 0]),
     ([1,0,1,1,0,1,1,0,1], 0, [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]),
     ([1,0,1,1,0,1,1,0,1], 1, [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]),
     ([1,0,1,1,0,1,1,0,1], 45678, [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]),
     ([1,0,1,1,0,1,1,0], 200, [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1]),
     ]
    for blk, mask, expected in testlists:
        test_crc.test_crc6(blk, mask, expected)
    
    testlists = [([1, 0, 1, 1], 0, [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]),
     ([1, 0, 1, 1], 12345, [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0]),
     ([0, 0, 1, 1, 0, 0, 0, 1], 0, [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]),
     ([0, 0, 1, 1, 0, 0, 0, 1], 24, [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0]),
     ([0, 0, 1, 1, 0, 0, 0, 1, 1,1,0,1], 0, [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]),
     ]
    for blk, mask, expected in testlists:
        test_crc.test_crc11(blk, mask, expected)

    testlists = [([0, 0, 1, 1, 0, 0, 0, 1, 1,1,0,1], 0, [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1]),
     ([0, 0, 1, 1, 0, 0, 0, 1, 1,1,0,1], 12345, [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0]),
     ]
    for blk, mask, expected in testlists:
        test_crc.test_crc16(blk, mask, expected)
    
    testlists = [([0, 0, 1, 1, 0, 0, 0, 1], 0, [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]),
     ([0, 0, 1, 1, 0, 0, 0, 1], 45678, [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1]),
     ]
    for blk, mask, expected in testlists:
        test_crc.test_crc24A(blk, mask, expected)
    
    testlists = [([0, 0, 1, 1, 0, 0, 0, 1], 0, [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
     ([0, 0, 1, 1, 0, 0, 0, 1], 45678, [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]),
     ]
    for blk, mask, expected in testlists:
        test_crc.test_crc24B(blk, mask, expected)
    
    testlists = [([0, 0, 1, 1, 0, 0, 0, 1], 0, [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]),
     ([0, 0, 1, 1, 0, 0, 0, 1], 45678, [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1]),
     ]
    for blk, mask, expected in testlists:
        test_crc.test_crc24C(blk, mask, expected)
    
