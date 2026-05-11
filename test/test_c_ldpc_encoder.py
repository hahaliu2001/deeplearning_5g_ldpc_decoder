# -*- coding: utf-8 -*-
from logging import config
import sys
import numpy as np
import pytest

import ctypes
import os

sys.path.append(os.getcwd()) 
    
from numpy.ctypeslib import ndpointer
from py5gphy.crc import crc
from py_src import ldpc_5g_encoder

## the file is to test crc implementation in c
# get the path of the c library
lib_path = os.path.abspath("c_lib/libnrLDPC.so")
libnrLDPC = ctypes.CDLL(lib_path)

class Ldpc_in_config(ctypes.Structure):
        _fields_ = [
            ("bgn", ctypes.c_uint16),
            ("A", ctypes.c_uint16),
            ("crctype", ctypes.c_uint16),
            ("E", ctypes.c_uint16),
            ("rv", ctypes.c_uint16),
            ("Qm", ctypes.c_uint16),
            ("HARQ_on", ctypes.c_uint16),
            
   ]

libnrLDPC.c_ldpc_5g_encode.argtypes = [
    ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS") # uint8_t* pointer infobytes
    ,ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS") # uint8_t* pointer,rm_bytes
    ,ctypes.POINTER(Ldpc_in_config)                 # pointer to ldpc_config_t struct
    
]
libnrLDPC.c_ldpc_5g_encode.restype = ctypes.c_int32

set_flag = 0
def test_nr_ldpc_encoder(A, E, Qm, rv, crcpoly, bgn):
        
    if set_flag:
        info_bits = np.random.randint(0, 2, A).astype(np.uint8)
        info_bytes = np.packbits(info_bits, bitorder='big')
        info_bits.tofile("out/encode_info_bits.bin")
        info_bytes.tofile("out/encode_info_bytes.bin")
    else:
        info_bits = np.fromfile("out/encode_info_bits.bin",dtype=np.uint8)
        info_bytes = np.fromfile("out/encode_info_bytes.bin",dtype=np.uint8)
    
    py_codeword = ldpc_5g_encoder.cb_ldpc_5g_encode(info_bits=info_bits, E=E, Qm=Qm, rv=rv, crcpoly=crcpoly, bgn=bgn)

    crclist ={"16":0,"24A":1,"24B":2,"24C":3}
    ldpc_in_config = Ldpc_in_config()
    ldpc_in_config.bgn = bgn
    ldpc_in_config.E = E
    ldpc_in_config.A = A
    ldpc_in_config.crctype = crclist[crcpoly]
    ldpc_in_config.rv = rv
    ldpc_in_config.Qm = Qm
    ldpc_in_config.HARQ_on = 0

    print(f"DEBUG: Attach GDB to Process ID (PID): {os.getpid()}")
    c_rmbytes = np.zeros(int((E+7)//8), dtype=np.uint8)
    c_ldpc_status = libnrLDPC.c_ldpc_5g_encode(info_bytes,c_rmbytes,ldpc_in_config)
    c_rm_bits = np.unpackbits(c_rmbytes, bitorder='big')

    print(np.array_equal(py_codeword, c_rm_bits[0:E]))
    pass

if __name__ == "__main__":
    print("start test c-code of nr ldpc encoder ")
    A,E, Qm,rv,crcpoly, bgn  = 512-24, 512*3, 1, 0, "24A", 1
    test_nr_ldpc_encoder(A, E, Qm, rv, crcpoly, bgn)
    pass