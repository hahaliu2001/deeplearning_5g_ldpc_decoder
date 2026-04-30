# -*- coding:utf-8 -*-
import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer

from GEO_LDPC_encoder import Xona_LDPC_encoder

from GEO_LDPC_encoder import Xona_LDPC_encoder
import GEO_layered_min_sum_ldpc_decoder

# this file is to simulate c code LDPC decoder performance and compare with python LDPC decoder

# some work for python to call c lib
lib_path = os.path.abspath("c_lib/libGeoLDPC.so")
libGeoLDPC = ctypes.CDLL(lib_path)

libGeoLDPC.layered_min_sum_decoder.argtypes = [
    ndpointer(dtype=np.int8, flags="C_CONTIGUOUS"), # int8_t* pointer
    ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"), # uint8_t* pointer
    ctypes.c_int16                                    # K
    , ctypes.c_int16                                   # N
    , ctypes.c_int16                                   # max_iter
    , ctypes.c_int16                                   # alpha
    , ctypes.c_int16                                   # beta
]
libGeoLDPC.layered_min_sum_decoder.restype = ctypes.c_uint32


#LDPC decoder config
LDPC_decoder_config = {
    "L": 32, # iteration number
    "alpha": 0.7, # used for normalized min-sum. 1 means no normalized
    #"beta": 0.3, # used for offset min-sum, 0 means no offset
    "beta": 0, # used for offset min-sum, 0 means no offset
}

snr_db_list = np.arange(-4.5, -1, 0.5).tolist()
#snr_db_list = np.arange(-1, 2, 0.5).tolist()

test_num = 200 #numer of test for each test case

set_flag = 1 #0: use save and load from binary files to test the C implementation, 

def run_LDPC_decoder_test(B, code_rate, LDPC_decoder_config, snr_db):
    #get C parameters
    c_L = LDPC_decoder_config["L"]
    c_alpha = (LDPC_decoder_config['alpha']*32767) // 1
    c_beta = LDPC_decoder_config['beta'] // 1

    info_bits = np.random.randint(0, 2, B-24)

    #encode the information bits using Xona LDPC encoder
    xona_encoded_bits = Xona_LDPC_encoder(info_bits, code_rate)

    #BPSK modulation
    transmitted_signal = 1 - 2 * xona_encoded_bits # assuming BPSK modulation, 0->1, 1->-1

    #add noise to the encoded bits
    noise_power = 10 ** (-snr_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(xona_encoded_bits))
    received_signal = transmitted_signal + noise

    LLR_bits = 2 * received_signal / noise_power # assuming BPSK modulation
    LLR_bits = LLR_bits/max(np.abs(LLR_bits)) * 127/4
    LLR_bits = LLR_bits.astype(np.int8)

    #python decoder
    py_decoded_bits, py_status = GEO_layered_min_sum_ldpc_decoder.GEO_layered_min_sum_ldpc_decoder(LLR_bits, code_rate, LDPC_decoder_config)
    py_decoded_status = py_status

    #use c implementation of layered min-sum decoder to decode the LLR bits
    info_bytes = np.packbits(info_bits, bitorder='big')
    c_decoded_bytes = np.zeros_like(info_bytes+4, dtype=np.uint8)
    
    c_status = libGeoLDPC.layered_min_sum_decoder(LLR_bits.astype(np.int8), c_decoded_bytes, 
            B, len(xona_encoded_bits), int(c_L), int(c_alpha), int(c_beta))
    
    c_decoded_status = 0 if np.array_equal(c_decoded_bytes[0:(B-24)//8], info_bytes) else 1

    return py_decoded_status, c_decoded_status

test_config_list = []
test_bler_list = []

for code_rate in [1/4, 1/5]:
#for code_rate in [1/5]:
    for B in [320,400,480,640,800,960,1280,3840,256,384,512,768]:
    #for B in [1280]:
        print(f"Testing code rate: {code_rate}, Testing information bit length: {B}")
        test_config_list.append((code_rate, B, LDPC_decoder_config))
        test_bler = []
        for snr_db in snr_db_list:
            py_error_count, c_error_count = 0, 0
            for i in range(test_num):
                py_decoded_status, c_decoded_status = run_LDPC_decoder_test(B, code_rate, LDPC_decoder_config, snr_db)

                py_error_count += py_decoded_status
                c_error_count +=  c_decoded_status
            
            bler = [py_error_count/test_num, c_error_count/test_num]
            test_bler.append(bler)
            print("rate ", code_rate, " ", B, "bits", LDPC_decoder_config, " SNR: ", snr_db, " BLER: ", bler)

        test_bler_list.append(test_bler)

import pickle
with open("out/int16_c_ldpc_decoder_result.pkl", 'wb') as f:
    pickle.dump((test_bler_list,test_config_list,snr_db_list), f)

pass