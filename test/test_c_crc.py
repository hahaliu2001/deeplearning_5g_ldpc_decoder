# -*- coding: utf-8 -*-
import sys
import numpy as np
import pytest

import ctypes
import os
sys.path.append(os.getcwd()) 
    
from numpy.ctypeslib import ndpointer

from py5gphy.crc import crc

## the file is to test crc implementation in c
# get the path of the c library
lib_path = os.path.abspath("c_lib/libnrLDPC.so")
libnrLDPC = ctypes.CDLL(lib_path)

libnrLDPC.calculate_crc.argtypes = [
    ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS") # uint8_t* pointer
    ,ctypes.c_int                                    # array length
    ,ctypes.c_int                                    # crc type
]
libnrLDPC.calculate_crc.restype = ctypes.c_uint32

def test_crc16():
    A = 256 - 16 # information bit size
    info_bits = np.random.randint(0, 2, A).astype(np.uint8)
    py_blkandcrc_bits = crc.nr_crc_encode(info_bits,'16')
    info_bytes = np.packbits(info_bits, bitorder='big')
    c_crc = libnrLDPC.calculate_crc(info_bytes, A//8, 0) # 0 for crc16
    c_crc_bits = np.array([(c_crc >> i) & 1 for i in range(15, -1, -1)], dtype=np.uint8)
    assert np.array_equal(py_blkandcrc_bits[A:], c_crc_bits), "CRC16 bits from C implementation do not match Python implementation"
    print("CRC16 test passed, CRC bits from C implementation match Python implementation")

def test_crc24a():
    A = 256 - 24 # information bit size
    info_bits = np.random.randint(0, 2, A).astype(np.uint8)
    py_blkandcrc_bits = crc.nr_crc_encode(info_bits,'24A')
    info_bytes = np.packbits(info_bits, bitorder='big')
    c_crc = libnrLDPC.calculate_crc(info_bytes, A//8, 1) # 1 for crc24A
    c_crc_bits = np.array([(c_crc >> i) & 1 for i in range(23, -1, -1)], dtype=np.uint8)
    assert np.array_equal(py_blkandcrc_bits[A:], c_crc_bits), "CRC24A bits from C implementation do not match Python implementation"
    print("CRC24A test passed, CRC bits from C implementation match Python implementation")

def test_crc24b():
    A = 256 - 24 # information bit size
    info_bits = np.random.randint(0, 2, A).astype(np.uint8)
    py_blkandcrc_bits = crc.nr_crc_encode(info_bits,'24B')
    info_bytes = np.packbits(info_bits, bitorder='big')
    c_crc = libnrLDPC.calculate_crc(info_bytes, A//8, 2) # 2 for crc24B
    c_crc_bits = np.array([(c_crc >> i) & 1 for i in range(23, -1, -1)], dtype=np.uint8)
    assert np.array_equal(py_blkandcrc_bits[A:], c_crc_bits), "CRC24B bits from C implementation do not match Python implementation"
    print("CRC24B test passed, CRC bits from C implementation match Python implementation")

def test_crc24c():
    A = 256 - 24 # information bit size
    info_bits = np.random.randint(0, 2, A).astype(np.uint8)
    py_blkandcrc_bits = crc.nr_crc_encode(info_bits,'24C')
    info_bytes = np.packbits(info_bits, bitorder='big')
    c_crc = libnrLDPC.calculate_crc(info_bytes, A//8, 3) # 3 for crc24C
    c_crc_bits = np.array([(c_crc >> i) & 1 for i in range(23, -1, -1)], dtype=np.uint8)
    assert np.array_equal(py_blkandcrc_bits[A:], c_crc_bits), "CRC24C bits from C implementation do not match Python implementation"
    print("CRC24C test passed, CRC bits from C implementation match Python implementation")

if __name__ == "__main__":
    test_crc16()
    test_crc24a()
    test_crc24b()
    test_crc24c()

    pass