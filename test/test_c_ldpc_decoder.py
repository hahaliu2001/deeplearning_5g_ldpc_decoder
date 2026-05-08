# -*- coding: utf-8 -*-
import sys
import numpy as np
import pytest

import ctypes
import os

from test_int16_c_ldpc_decoder import LLR_bits
sys.path.append(os.getcwd()) 
    
from numpy.ctypeslib import ndpointer
from py5gphy.ldpc import nr_ldpc_raterecover
from py5gphy.crc import crc
from py_src import ldpc_5g_decoder
from py_src import ldpc_5g_encoder
from py_src.ldpc_utils import tx_rx_process

## the file is to test crc implementation in c
# get the path of the c library
lib_path = os.path.abspath("c_lib/libnrLDPC.so")
libnrLDPC = ctypes.CDLL(lib_path)

libnrLDPC.ldpc_decode.argtypes = [
    ndpointer(dtype=np.int16, flags="C_CONTIGUOUS") # int16_t* pointer
    ,ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS") # uint8_t* pointer
    ,ctypes.POINTER(ctypes.c_uint16)                 # pointer to ldpc_config_t struct
    ,ctypes.POINTER(ctypes.c_uint16)                 # pointer to ldpc_decoder_cfg_t struct]
]
libnrLDPC.ldpc_decode.restype = ctypes.c_uint32

def test_bgn1_nr_ldpc_decoder():
    """ test LDPC decoder for base graph 1, with random input data, and compare the decoded bits with Python implementation
    
    input:
        LLr_fe: E length sequence after de-interleaving, which is also the output of rate recovery
        K_apo: the starting point of filler bits in the code block segment, which is also the number of information bits after CRC in the code block segment
        Ncb: the code block size after inserting CRC and filler bits, which is also the size of code block segment after LDPC encoding but before rate matching
        N: the code block size after LDPC encoding and rate matching, which is also the input bit size for LDPC decoder
        Zc: lifting size for LDPC encoding and decoding
    """
    #test parameters
    A,E, Qm,rv,crcpoly, bgn  = 512-24, 512*5, 1, 0, "24A", 1
    noise_power_db = -20
    
    info_bits = np.random.randint(0, 2, A).astype(np.uint8)
    codeword = ldpc_5g_encoder.ldpc_5g_encode(info_bits=info_bits, E=E, Qm=Qm, rv=rv, crcpoly=crcpoly, bgn=bgn)

    LLrout_5g = tx_rx_process(codeword, Qm=Qm, noise_power_db=noise_power_db)

    #fixed point to 16 bits for C implementation, and also convert to int16 type, which is the input type for C implementation
    LLr_dn_c = np.clip(np.round(LLrout_5g/max(np.abs(LLR_bits)) * 128), -32768, 32767).astype(np.int16) 
    LLr_dn_py = LLrout_5g.astype(np.float32) # keep the original LLR values for Python implementation, which uses float32 type as input for LDPC decoder
    
    #prepare config struct for C implementation
    class Config(ctypes.Structure):
        _fields_ = [
            ("bgn", ctypes.c_uint16),
            ("E", ctypes.c_uint16),
            ("A", ctypes.c_uint16),
            ("crctype", ctypes.c_uint16),
            ("rv", ctypes.c_uint16),
            ("Qm", ctypes.c_uint16),
            ("HARQ_on", ctypes.c_uint16),
            ("B", ctypes.c_uint16),
            ("Zc", ctypes.c_uint16),
            ("Vij_array", ctypes.POINTER(ctypes.c_uint16))
        ]
    config = Config()
    config.bgn = bgn
    config.E = N
    config.A = B
    config.crctype = 1 # 1 for crc24A
    config.rv = rv
    config.Qm = Qm
    config.HARQ_on = 0
    config.B = B
    config.Zc = Zc
    #prepare decoder config struct for C implementation
    class DecoderConfig(ctypes.Structure):
        _fields_ = [
            ("L", ctypes.c_uint16),
            ("algo", ctypes.c_uint16),
            ("alpha", ctypes.c_float),
            ("beta", ctypes.c_float)
        ]
    decoder_config = DecoderConfig()
    decoder_config.L = 32
    decoder_config.algo = 0 # 0 for BF, 1 for BP, 2 for min-sum, 3 for layered-BP, 4 for layered-min-sum
    decoder_config.alpha = 0.7
    decoder_config.beta = 0.3
    hardbits_c = np.zeros(N, dtype=np.uint8) # output bits from C implementation
    libnrLDPC.ldpc_decode(LLr_dn_c, hardbits_c, ctypes.byref(config), ctypes.byref(decoder_config))
    hardbits_py = (LLr_dn_py < 0).astype(np.uint8) # output bits from Python implementation, using hard decision
    assert np.array_equal(hardbits_c, hardbits_py), "Decoded bits from C implementation do not match Python implementation"
    print("LDPC decoder test passed, decoded bits from C implementation match Python implementation")

if __name__ == "__main__":
    test_bgn1_nr_ldpc_decoder()
    pass