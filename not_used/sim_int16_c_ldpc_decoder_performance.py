# -*- coding:utf-8 -*-
import numpy as np
import ctypes
import os
import pandas as pd
from numpy.ctypeslib import ndpointer

from GEO_LDPC_encoder import Xona_LDPC_encoder

from GEO_LDPC_encoder import Xona_LDPC_encoder
import GEO_layered_min_sum_ldpc_decoder

# this file is to simulate c code LDPC decoder performance and compare with python LDPC decoder

# some work for python to call c lib
lib_path = os.path.abspath("c_lib/libGeoLDPC.so")
libGeoLDPC = ctypes.CDLL(lib_path)

libGeoLDPC.int16_layered_min_sum_decoder.argtypes = [
    ndpointer(dtype=np.int8, flags="C_CONTIGUOUS"), # int8_t* pointer
    ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"), # uint8_t* pointer
    ctypes.c_int16                                    # K
    , ctypes.c_int16                                   # E
    , ctypes.c_int16                                   # max_iter
    , ctypes.c_int16                                   # alpha
    , ctypes.c_int16                                   # beta
    , ctypes.c_int16                                   # filling_LLr
    , ctypes.c_int16                                   # en_crc_check
]
libGeoLDPC.int16_layered_min_sum_decoder.restype = ctypes.c_int32


#LDPC decoder config
LDPC_decoder_config = {
    "L": 32, # iteration number
    "alpha": 0.7, # used for normalized min-sum. 1 means no normalized
    #"beta": 0.3, # used for offset min-sum, 0 means no offset
    "beta": 0, # used for offset min-sum, 0 means no offset
}

#code_rate_list = [1/4, 1/5]  #code rate list
code_rate_list = [1/5]  #code rate list
#info_bit_list = [320,400,480,640,800,960,1280,3840,256,384,512,768] #information bit size list
info_bit_list = [3840] #information bit size list

#simulate parameters
#snr_db_list = np.arange(-3.5, 0, 0.5).tolist()
snr_db_list = np.arange(-4.5, -3, 0.5).tolist()

test_num = 2000 #numer of test for each test case

#choose algorithm
#algo_list = ["min-sum", "BP", "layered-min-sum"]
algo_list = ["layered-min-sum"]
#algo = "BP"

#file name to save the simulation results
result_file_name = f"int16_LDPC_decoder_simulation_results"

#[1,127,127*10] get best performance
testcase_list = [
    #crc check, fixed_llr, fillingbit
    [1,127,127*10],[0,127,127*10],[1,127/2,127/2*10],[1,32,32*10],[1,16,16*10],[1,8,8*10],[1,127/2,127/2*100]
]

def run_LDPC_decoder_test(B, code_rate, LDPC_decoder_config, snr_db,algo="layered-min-sum"):
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

    info_bytes = np.packbits(info_bits, bitorder='big')
    c_decoded_bytes = np.zeros_like(info_bytes+4, dtype=np.uint8)

    c_status_list = []
    for testcase in testcase_list:
        c_en_crc_check = testcase[0]
        c_fixed_LLr = testcase[1]
        c_filling_LLr = testcase[2]

        if c_fixed_LLr < 127:
            c_LLR_bits = LLR_bits/max(np.abs(LLR_bits)) * c_fixed_LLr + 0.5 # scale the LLR bits to int8 range, and add 0.5 for rounding when converting to int
        else:
            c_LLR_bits = LLR_bits/max(np.abs(LLR_bits)) * c_fixed_LLr
        c_LLR_bits = c_LLR_bits.astype(np.int8)
            
        c_status = libGeoLDPC.int16_layered_min_sum_decoder(c_LLR_bits, c_decoded_bytes, 
            B, len(xona_encoded_bits), int(c_L), int(c_alpha), int(c_beta), int(c_filling_LLr), int(c_en_crc_check))
        c_status_list.append(c_status)
       
    return np.array(c_status_list)

test_config_list = []
test_bler_list = []

for algo in algo_list:
    LDPC_decoder_config = {"L": 32,"algo": algo,"alpha": 0.8,"beta": 0}
    for code_rate in code_rate_list:
        for info_bits in info_bit_list:
            test_config_list.append((code_rate, info_bits, LDPC_decoder_config))
            test_bler = []
            for snr_db in snr_db_list:
                print("start simulation for rate ", code_rate, " ", info_bits, "bits", LDPC_decoder_config, " SNR: ", snr_db)
                error_count = np.zeros(len(testcase_list))
                for i in range(test_num):
                    #print("Testing LDPC decoder with code rate: ", code_rate, " and info bits: ", info_bits)
                    status = run_LDPC_decoder_test(info_bits, code_rate, LDPC_decoder_config, snr_db,algo=algo)

                    error_count += status # if status is False, it means decoding failed, which is counted as an error

                bler = error_count / test_num
                test_bler.append(bler)
                print("rate ", code_rate, " ", info_bits, "bits", LDPC_decoder_config, " SNR: ", snr_db, " BLER: ", bler)
            test_bler_list.append(test_bler)                    

# save the test results to a CSV file
results_df = pd.DataFrame(test_bler_list, columns=snr_db_list)
results_df.insert(0, 'LDPC_decoder_config', [str(config) for config in test_config_list])
results_df.to_csv(result_file_name+'.csv', index=False)

##use pickle to save the test results
import pickle
with open(result_file_name+'.pkl', 'wb') as f:
    pickle.dump((test_config_list, test_bler_list), f)
pass