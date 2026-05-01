# -*- coding:utf-8 -*-

import sys
import numpy as np
import os
import pickle

sys.path.append(os.getcwd()) 

from py_src.ldpc_5g_decoder import ldpc_5g_decode
from py_src.ldpc_utils import tx_rx_process
from py_src.ldpc_5g_encoder import ldpc_5g_encode

################ define simulation parameters here ################
A,E, Qm,rv,crcpoly, bgn  = 3840-24, 3840*5, 1, 0, "24A", 2

default_ldpcdecoder_config = {"L": 32, "algo": "BP", "alpha": 0.8, "beta": 0}

noise_power_db_list = np.arange(7, 8.5, 0.5) # noise power in dB for AWGN noise
algo_list = ["BP", "min-sum"]
test_num = 10
output_dir = "output/floating_algo_sim_results.pkl"

def run_LDPC_decoder_test_no_HARQ(A,E, Qm, noise_power_db, rv, crcpoly, bgn):
    """ run LDPC decoder test for given parameters
    """
    info_bits=np.random.randint(0, 2, size=A)
    codeword = ldpc_5g_encode(info_bits=info_bits, E=E, Qm=Qm, rv=rv, crcpoly=crcpoly, bgn=bgn)
    LLrout = tx_rx_process(codeword, Qm=Qm, noise_power_db=noise_power_db)
    #LLrout = LLrout/2 # to make the noise power after scaling the LLR to be the same as the original noise power, which is also the noise power used for LDPC decoding

    status_list = []
    for algo in algo_list:
        ldpcdecoder_config = default_ldpcdecoder_config.copy()
        ldpcdecoder_config["algo"] = algo
        blkandcrc, status, _ = ldpc_5g_decode(LLrout, B=A+24, Qm=Qm, rv=rv, crcpoly=crcpoly, bgn=bgn, ldpcdecoder_config=ldpcdecoder_config)
        status_list.append(status)
        print(f"LDPC decoder algo: {algo}, decoding status: {status}")
    return status_list

test_bler_list = []
for noise_power_db in noise_power_db_list:
    print(f"Testing noise power (dB): {noise_power_db}")
    bler_sum = np.zeros(len(algo_list))
    for i in range(test_num):
        status_list = run_LDPC_decoder_test_no_HARQ(A,E, Qm, noise_power_db, rv, crcpoly, bgn)
        bler_sum += np.array(status_list)
    bler_list = bler_sum / test_num
    test_bler_list.append(bler_list)

# Save the simulation results
with open(output_dir, "wb") as f:
    pickle.dump(test_bler_list, f)