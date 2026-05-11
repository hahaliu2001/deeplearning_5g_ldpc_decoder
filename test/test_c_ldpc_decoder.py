# -*- coding: utf-8 -*-
from logging import config
import sys
import numpy as np
import pytest

import ctypes
import os

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

class DecoderConfig(ctypes.Structure):
        _fields_ = [
            ("algo", ctypes.c_uint16),
            ("max_iterations", ctypes.c_uint16),
            ("alpha", ctypes.c_uint16),
            ("beta", ctypes.c_uint16),
            ("filling_bit_value", ctypes.c_int16),
            ("max_llr", ctypes.c_int16),      
            ("min_llr", ctypes.c_int16),    
        ]

libnrLDPC.c_ldpc_5g_decode.argtypes = [
    ndpointer(dtype=np.int16, flags="C_CONTIGUOUS") # int16_t* pointer
    ,ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS") # uint8_t* pointer
    ,ctypes.POINTER(Ldpc_in_config)                 # pointer to ldpc_config_t struct
    ,ctypes.POINTER(DecoderConfig)                 # pointer to ldpc_decoder_cfg_t struct]
]
libnrLDPC.c_ldpc_5g_decode.restype = ctypes.c_int32

set_flag = 0

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
    A,E, Qm,rv,crcpoly, bgn  = 512-24, 512*3, 1, 0, "24A", 1
    noise_power_db = -20
    
    info_bits = np.random.randint(0, 2, A).astype(np.uint8)
    codeword = ldpc_5g_encoder.cb_ldpc_5g_encode(info_bits=info_bits, E=E, Qm=Qm, rv=rv, crcpoly=crcpoly, bgn=bgn)

    LLrout_5g = tx_rx_process(codeword, Qm=Qm, noise_power_db=noise_power_db)

    #fixed point to 16 bits for C implementation, and also convert to int16 type, which is the input type for C implementation
    LLr_dn_c = np.clip(np.round(LLrout_5g/max(np.abs(LLrout_5g)) * 128), -32768, 32767).astype(np.int16) 
    
    if set_flag:
        LLr_dn_c.tofile("out/LLR_bits.bin") # save the LLR bits to binary file for testing C implementation
        info_bytes = np.packbits(info_bits, bitorder='big')
        info_bits.astype(np.uint8).tofile("out/info_bits.bin") # save the information bits to binary file for testing C implementation
        info_bytes.tofile("out/info_bytes.bin") # save the information bytes to binary file for testing C implementation
    else:
        LLr_dn_c = np.fromfile("out/LLR_bits.bin", dtype=np.int16)
        info_bits = np.fromfile("out/info_bits.bin", dtype=np.uint8)
        info_bytes = np.fromfile("out/info_bytes.bin", dtype=np.uint8)

    #python decoder
    ldpcdecoder_config = {"L": 32, "algo": "layered-min-sum", "alpha": 1, "beta": 0}
    blkandcrc_py, status_py, _ = ldpc_5g_decoder.cb_ldpc_5g_decode(LLr_dn_c, B=A+24, Qm=Qm, rv=rv, 
                                        crcpoly=crcpoly, bgn=bgn, ldpcdecoder_config=ldpcdecoder_config)

    #prepare config struct for C implementation
    
    ldpc_in_config = Ldpc_in_config()
    ldpc_in_config.bgn = bgn
    ldpc_in_config.E = E
    ldpc_in_config.A = A
    ldpc_in_config.crctype = 1 # 1 for crc24A
    ldpc_in_config.rv = rv
    ldpc_in_config.Qm = Qm
    ldpc_in_config.HARQ_on = 0
    
    #prepare decoder config struct for C implementation
    
    decoder_config = DecoderConfig()
    decoder_config.L = 32
    decoder_config.algo = 1 # 0 for layered-BP, 1 for layered-min-sum
    decoder_config.alpha = np.uint16(0.7 * 32767)
    decoder_config.beta = np.uint16(0 * 32767)
    decoder_config.filling_bit_value = 128*20 # the LLR value for filling bits in rate recovery, which can be set to 0 for HARQ combining
    decoder_config.max_iterations = 32
    decoder_config.max_llr = 32767 # the maximum LLR value for input to LDPC decoder, which is set to 127 for int8 implementation in C
    decoder_config.min_llr = -32768 # the minimum LLR value for input to LDPC decoder, which is

    hardbits_c = np.zeros(int(A//8+3), dtype=np.uint8) # output bits from C implementation
    print(f"DEBUG: Attach GDB to Process ID (PID): {os.getpid()}")
    #input("Press Enter to start C call...")
    c_ldpc_status = libnrLDPC.c_ldpc_5g_decode(LLr_dn_c, hardbits_c, ctypes.byref(ldpc_in_config), ctypes.byref(decoder_config))
    c_decoded_bits = np.unpackbits(hardbits_c, bitorder='big')

    np.array_equal(info_bits, c_decoded_bits[0:A]), "Decoded bits from C implementation do not match Python implementation"
    print(f"c code LDPC decoder test: {c_ldpc_status}")

if __name__ == "__main__":
    test_bgn1_nr_ldpc_decoder()
    pass