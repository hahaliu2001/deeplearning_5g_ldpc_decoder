# -*- coding:utf-8 -*-
import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer

from GEO_LDPC_encoder import Xona_LDPC_encoder

from GEO_LDPC_encoder import Xona_LDPC_encoder
import GEO_layered_min_sum_ldpc_decoder_for_debug

# this file is to test the LDPC decoder implementation in c
# the code is to verify that the C functional implementation is correct, 
# for example, under 10dB SNR, c LDPC function can correctly decode LDPC. it doesn't simulate C code performance.
# which will be done in another file


lib_path = os.path.abspath("c_lib/libGeoLDPC.so")
libGeoLDPC = ctypes.CDLL(lib_path)

libGeoLDPC.layered_min_sum_decoder.argtypes = [
    ndpointer(dtype=np.int8, flags="C_CONTIGUOUS"), # int8_t* pointer
    ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"), # uint8_t* pointer
    ctypes.c_int16                                    # K
    , ctypes.c_int16                                   # N
    , ctypes.c_int16                                   # max_iter
    , ctypes.c_int8                                   # alpha
    , ctypes.c_int8                                   # beta
]
libGeoLDPC.layered_min_sum_decoder.restype = ctypes.c_uint32


#LDPC decoder config
LDPC_decoder_config = {
    "L": 32, # iteration number
    "alpha": 0.7, # used for normalized min-sum. 1 means no normalized
    #"beta": 0.3, # used for offset min-sum, 0 means no offset
    "beta": 0, # used for offset min-sum, 0 means no offset
}

set_flag = 0 #0: use save and load from binary files to test the C implementation, 
#noise power in dB
noise_power_db = 1
noise_power = 10 ** (noise_power_db / 10)
#for code_rate in [1/4, 1/5]:
for code_rate in [1/5]:
    #for B in [320,400,480,640,800,960,1280,3840,256,384,512,768]:
    for B in [1280]:
        print(f"Testing code rate: {code_rate}, Testing information bit length: {B}")
        if set_flag==1:
            info_bits = np.random.randint(0, 2, size=B-24) # generate random information bits of size B
        else:
            info_bits = np.fromfile("out/info_bits.bin", dtype=np.uint8) # load info bits from binary file for testing C implementation
        
        if set_flag==1:
            xona_encoded_bits = Xona_LDPC_encoder(info_bits, code_rate)
        
            #BPSK modulation of xona_encoded_bits
            mod_bits = 1 - 2 * xona_encoded_bits  # BPSK modulation
            #add noise to the encoded bits, and convert them to LLR bits
            noise = np.sqrt(noise_power) * np.random.randn(len(xona_encoded_bits))
            received_bits = mod_bits + noise
            LLR_bits = 2 * received_bits / noise_power # assuming BPSK modulation

            LLR_bits = LLR_bits/max(np.abs(LLR_bits)) * 127/4 # normalize LLR bits to fit in int8 range
            LLR_bits = LLR_bits.astype(np.int8) # convert to int8 for C implementation
        else:
            LLR_bits = np.fromfile("out/LLR_bits.bin", dtype=np.int8) # load LLR bits from binary file for testing C implementation
                    
        decoded_bits, status = GEO_layered_min_sum_ldpc_decoder_for_debug.GEO_layered_min_sum_ldpc_decoder(LLR_bits, code_rate, LDPC_decoder_config)
        print("LDPC decode status: ", status)
        assert np.array_equal(decoded_bits[0:B-24], info_bits), "The decoded bits and original information bits should be the same"

        #use c implementation of layered min-sum decoder to decode the LLR bits
        info_bytes = np.packbits(info_bits, bitorder='big')
        # save LLR_bits and info_bytes to binary files for debugging
        if set_flag==1:
            LLR_bits.tofile("out/LLR_bits.bin")
            info_bits.astype(np.uint8).tofile("out/info_bits.bin")
            info_bytes.tofile("out/info_bytes.bin")

        # load LLR_bits and info_bytes from binary files for debugging
        if 1:
            c_decoded_bytes = np.zeros_like(decoded_bits, dtype=np.uint8)
            c_status = libGeoLDPC.layered_min_sum_decoder(LLR_bits.astype(np.int8), c_decoded_bytes, B, len(LLR_bits), LDPC_decoder_config["L"],89,0)

            print("C layered min-sum LDPC decode status: ", c_status==0)
            #assert np.array_equal(c_decoded_bytes[0:(B-24)//8], info_bytes), "The decoded bits from C implementation should match the original information bits"
            print(f"C layered min-sum LDPC decoder info bits comparison is: {np.array_equal(c_decoded_bytes[0:(B-24)//8], info_bytes)}")
        
        

        

pass