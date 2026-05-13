# -*- coding: utf-8 -*-
from logging import config
import sys
import numpy as np
import pytest
import time
import itertools

import ctypes
import os

sys.path.append(os.getcwd()) 
    
from numpy.ctypeslib import ndpointer
from py5gphy.ldpc import nr_ldpc_raterecover
from py5gphy.crc import crc
from py_src import ldpc_5g_decoder
from py_src import ldpc_5g_encoder
from py_src.ldpc_utils import tx_rx_process

"""the file is to simulate LDPC decoder performance using c-code LDPC encoder and decoder
"""
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

libnrLDPC.c_ldpc_5g_encode.argtypes = [
    ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS") # uint8_t* pointer infobytes
    ,ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS") # uint8_t* pointer,rm_bytes
    ,ctypes.POINTER(Ldpc_in_config)                 # pointer to ldpc_config_t struct
    
]
libnrLDPC.c_ldpc_5g_encode.restype = ctypes.c_int32

def run_ldpc_test(ldpc_in_config, decoder_config_list, noise_power_db):
    """ run test"""

    info_bits = np.random.randint(0, 2, ldpc_in_config.A).astype(np.uint8)
    info_bytes = np.packbits(info_bits, bitorder='big')

    ##Encoding
    c_rmbytes = np.zeros(int((ldpc_in_config.E+7)//8), dtype=np.uint8)
    c_ldpc_status = libnrLDPC.c_ldpc_5g_encode(info_bytes,c_rmbytes,ldpc_in_config)
    c_rm_bits = np.unpackbits(c_rmbytes, bitorder='big')

    ## modulation, add noise, LLR demodulation, 
    LLrout_5g = tx_rx_process(c_rm_bits, Qm=ldpc_in_config.Qm, noise_power_db=noise_power_db)
    #fixed point to 16 bits for C implementation, and also convert to int16 type, which is the input type for C implementation
    LLr_dn_c = np.clip(np.round(LLrout_5g/max(np.abs(LLrout_5g)) * 128), -32768, 32767).astype(np.int16) 

    ## ldpc decoder
    hardbits_c = np.zeros(int(A//8+3), dtype=np.uint8)
    decode_status = np.zeros(len(decoder_config_list))
    for idx, decode_config in enumerate(decoder_config_list):       
        decoder_config = DecoderConfig()
        decoder_config.algo = decode_config[0] # 0 for layered-BP, 1 for layered-min-sum
        decoder_config.max_iterations = decode_config[1]
        decoder_config.alpha = decode_config[2]
        decoder_config.beta = decode_config[3]
        decoder_config.filling_bit_value = decode_config[4] # the LLR value for filling bits in rate recovery, which can be set to 0 for HARQ combining
        decoder_config.max_llr = decode_config[5] # the maximum LLR value for input to LDPC decoder, which is set to 127 for int8 implementation in C
        decoder_config.min_llr = decode_config[6] # the minimum LLR value for input to LDPC decoder, which is

        c_ldpc_status = libnrLDPC.c_ldpc_5g_decode(LLr_dn_c, hardbits_c, ctypes.byref(ldpc_in_config), ctypes.byref(decoder_config))

        if c_ldpc_status == 0:
            decode_status[idx] = 0
        else:
            decode_status[idx] = 1

    return decode_status

#test parameters
#QmList = [1,2,4,6,8]
QmList = [1]
#rvList = [0,1,2,3]
rvList = [0]
#B_list = [256, 512, 1024, 2048, 3840, 5120]
B_list = [512,1024]
#coderate_list = [8/9, 4/5, 2/3, 1/3, 1/5]
coderate_list = [ 1/4]
coderate_map = {8/9:'8/9', 4/5:'4/5', 2/3:'2/3', 1/2: '1/2',1/3:'1/3', 1/4:'1/4',1/5:'1/5'}
#coderate_list = [1/5]
bgn_list = [1,2]
#crctype_list = [0,1,2,3]
crctype_list = [1]
all_configs = list(itertools.product(QmList, rvList, B_list, coderate_list, bgn_list, crctype_list))

decoder_config_list = [
    #algo 0 for layered-BP, 1 for layered-min-sum
    #algo, max_iterations,alpha, beta, filling_bit_value,max_llr,min_llr
    [0,    32,  1,  0,  1280, 20, -32768],
    
    #0.7*32767=22936
    [1,    32,  22936,  0,  1280, 32767, -32768],
    
    [2,    32,  21954,  0,  1280, 32767, -32768],

]

snr_power_db_list = np.arange(-7,-4.5,0.5)
#snr_power_db_list = np.arange(0,2,0.5)

testnum = 2000


result_file_name = "out/c_neural_ldpc_decoder_result.pkl"

print(f"Generated {len(all_configs)} configurations.")

test_result = []
for tstconfig in all_configs:
    Qm = tstconfig[0]
    rv = tstconfig[1]
    B = tstconfig[2]
    coderate = tstconfig[3]
    bgn = tstconfig[4]
    crctype = tstconfig[5]
    A = B-16 if crctype==0 else B-24
    E = int(B/coderate)

    if bgn==2 and B >3840:
        continue

    ldpc_in_config = Ldpc_in_config()
    ldpc_in_config.bgn = bgn
    ldpc_in_config.E = E
    ldpc_in_config.A = A
    ldpc_in_config.crctype = crctype
    ldpc_in_config.rv = rv
    ldpc_in_config.Qm = Qm
    ldpc_in_config.HARQ_on = 0
    #fill the value to
    bler_list = []
    for snr_power_db in snr_power_db_list:
        error_count = np.zeros(len(decoder_config_list))
        for m in range(testnum):
            noise_power_db = -snr_power_db - 10*np.log10(Qm)
            decode_status = run_ldpc_test(ldpc_in_config, decoder_config_list, noise_power_db)
            error_count += decode_status
            
        bler = error_count/testnum
        print(f"Qm {Qm},rv {rv},B {B} coderate {coderate_map[coderate]}, bgn {bgn}, crc {crctype} snr_power_db {snr_power_db}, BLER={bler}")

        bler_list.append(bler)
    
    test_result.append([tstconfig, bler_list])

# write result to pickle
import pickle
with open(result_file_name+'.pkl', 'wb') as f:
    pickle.dump((test_result, decoder_config_list,snr_power_db_list,testnum), f)
pass