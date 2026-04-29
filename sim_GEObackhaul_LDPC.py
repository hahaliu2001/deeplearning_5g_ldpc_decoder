#!/usr/bin/env python
# coding: utf-8

# The code is to simulate GEO backhaul LDPC decoder performance

# Input parameters:
# - Code rate: 1/4 and 1/5
# - infomation bits size:
# 

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import use_py5gphy_BP_minsum_LDPC_decoder
import GEO_LDPC_encoder
import GEO_layered_min_sum_ldpc_decoder


# In[ ]:


# LDPC parameters list
code_rate_list = [1/4, 1/5]  #code rate list
info_bit_list = [320,400,480,640,800,960,1280,3840,256,384,512,768] #information bit size list

#simulate parameters
#snr_db_list = np.arange(-3.5, 0, 0.5).tolist()
snr_db_list = np.arange(-3, -1, 0.5).tolist()

#LDPC decoder config list
if (0):
    LDPC_decoder_config_list = [
        {"L": 32,"algo": "min-sum","alpha": 0.8,"beta": 0.3},
        {"L": 32,"algo": "min-sum","alpha": 0.7,"beta": 0.3},
        {"L": 16,"algo": "min-sum","alpha": 0.8,"beta": 0.3},
        {"L": 16,"algo": "min-sum","alpha": 0.7,"beta": 0.3}
    ]
else:
    LDPC_decoder_config_list = [
        {"L": 32,"algo": "min-sum","alpha": 0.8,"beta": 0.3},
    ]

#test num
test_num = 200 #numer of test for each test case

#choose algorithm
#algo_list = ["min-sum", "BP", "layered-min-sum"]
algo = "BP"

#file name to save the simulation results
result_file_name = f"LDPC_decoder_simulation_results_{algo}"


# ## main code to run the test

# In[4]:


def run_LDPC_decoder_test(B, code_rate, LDPC_decoder_config, snr_db, algo="layered-min-sum"):

    info_bits = np.random.randint(0, 2, B)

    #encode the information bits using Xona LDPC encoder
    xona_encoded_bits = GEO_LDPC_encoder.Xona_LDPC_encoder(info_bits, code_rate)

    #BPSK modulation
    transmitted_signal = 1 - 2 * xona_encoded_bits # assuming BPSK modulation, 0->1, 1->-1

    #add noise to the encoded bits
    noise_power = 10 ** (-snr_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(xona_encoded_bits))
    received_signal = transmitted_signal + noise

    LLR_bits = 2 * received_signal / noise_power # assuming BPSK modulation

    #decode the received bits using Xona LDPC decoder
    LDPC_decoder_config["algo"] = algo
    if algo == "min-sum":
        decoded_bits, status = use_py5gphy_BP_minsum_LDPC_decoder.jie_LDPC_decoder(LLR_bits, code_rate, LDPC_decoder_config)
    elif algo == "BP":
        decoded_bits, status = use_py5gphy_BP_minsum_LDPC_decoder.jie_LDPC_decoder(LLR_bits, code_rate, LDPC_decoder_config)
    elif algo == "layered-min-sum":
        decoded_bits, status = GEO_layered_min_sum_ldpc_decoder.GEO_layered_min_sum_ldpc_decoder(LLR_bits, code_rate, LDPC_decoder_config)

    #if status == True:
    #    assert np.array_equal(decoded_bits, info_bits), "Decoded bits do not match the original input bits for code rate {}, info bits {}, LDPC decoder config {}, SNR {}".format(code_rate, info_bits, LDPC_decoder_config, snr_db)
    return decoded_bits[0:B], status


# In[ ]:


test_config_list = []
test_bler_list = []

for LDPC_decoder_config in LDPC_decoder_config_list:
    for code_rate in code_rate_list:
        for info_bits in info_bit_list:
            test_config_list.append((code_rate, info_bits, LDPC_decoder_config))
            test_bler = []
            for snr_db in snr_db_list:
                print("start simulation for rate ", code_rate, " ", info_bits, "bits", LDPC_decoder_config, " SNR: ", snr_db)
                error_count = 0
                for i in range(test_num):
                    #print("Testing LDPC decoder with code rate: ", code_rate, " and info bits: ", info_bits)
                    decoded_bits, status = run_LDPC_decoder_test(info_bits, code_rate, LDPC_decoder_config, snr_db)

                    error_count += (not status) # if status is False, it means decoding failed, which is counted as an error

                bler = error_count / test_num
                test_bler.append(bler)
                print("rate ", code_rate, " ", info_bits, "bits", LDPC_decoder_config, " SNR: ", snr_db, " BLER: ", bler)
            test_bler_list.append(test_bler)                    



# In[15]:


# save the test results to a CSV file
results_df = pd.DataFrame(test_bler_list, columns=snr_db_list)
results_df.insert(0, 'LDPC_decoder_config', [str(config) for config in test_config_list])
results_df.to_csv(result_file_name+'.csv', index=False)

##use pickle to save the test results
import pickle
with open(result_file_name+'.pkl', 'wb') as f:
    pickle.dump((test_config_list, test_bler_list), f)

