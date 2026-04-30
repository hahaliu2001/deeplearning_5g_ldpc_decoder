# -*- coding:utf-8 -*-
import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer

from py5gphy.crc import crc

# the file is to test crc24A implementation in c

lib_path = os.path.abspath("c_lib/libGeoLDPC.so")
libGeoLDPC = ctypes.CDLL(lib_path)

# generate input data for crc24A test
A = 256 - 24 # information bit size
info_bits = np.random.randint(0, 2, A).astype(np.uint8)

# call crc encoder from python code
py_blkandcrc_bits = crc.nr_crc_encode(info_bits,'24A') 

# convert info_bits to c array bytes array
info_bytes = np.packbits(info_bits, bitorder='big')
# prepare output array for crc24A test, the size is A+24
blkandcrc_bytes = np.zeros((A + 24)//8, dtype=np.uint8)
blkandcrc_bytes[0:len(info_bytes)] = info_bytes

libGeoLDPC.calculate_crc24a_fast.argtypes = [
    ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"), # uint8_t* pointer
    ctypes.c_int                                    # array length
]
libGeoLDPC.calculate_crc24a_fast.restype = ctypes.c_uint32
# call the crc24A function in c
c_crc = libGeoLDPC.calculate_crc24a_fast(info_bytes, A//8)

blkandcrc_bytes[A//8:] = np.array([(c_crc >> i) & 0xFF for i in range(16, -8, -8)], dtype=np.uint8)

#unpack the crc result from c to bits
c_crc_bits = np.array([(c_crc >> i) & 1 for i in range(23, -1, -1)], dtype=np.uint8)
assert np.array_equal(py_blkandcrc_bits[A:], c_crc_bits), "CRC24A bits from C implementation do not match Python implementation"
print("CRC24A test passed, CRC bits from C implementation match Python implementation")


pass
