# -*- coding:utf-8 -*-

import sys
import numpy as np
import os

sys.path.append(os.getcwd()) 
from py5gphy.common import nrModulation
from py5gphy.demodulation import nr_Demodulation

def tx_rx_process(inbits, Qm, noise_power_db):
    """ does modulation, add AWGN noise, and demodulation for test purpose
    input:
        inbits: input bits, should be a 1D numpy array
        Qm: modulation order, should be 1 for BPSK, 2 for QPSK, 4 for 16QAM, 6 for 64QAM, 8 for 256QAM
        noise_power_db: noise power for AWGN noise in dB
    output:
        LLrout: output LLR bits after demodulation, should be a 1D numpy array of the same size as inbits
    """
    # modulation
    mod_table = {1:'BPSK', 2:'QPSK', 4:'16QAM', 6:'64QAM', 8:'256QAM'}
    mod_data = nrModulation.nrModulate(inbits, mod_table[Qm])
    
    # add AWGN noise to the modulated signal
    power_noise = 10 ** (noise_power_db / 10)
    noise = np.random.normal(0, np.sqrt(power_noise), mod_data.shape)
    noisy_signal = mod_data + noise

    # demodulation
    hardbits,LLrout = nr_Demodulation.nrDemodulate(noisy_signal,mod_table[Qm],power_noise*np.ones(noisy_signal.shape))
    
    return LLrout

if __name__ == "__main__":
    # test tx_rx_process
    inbits = np.random.randint(0, 2, size=100) # generate random input bits
    Qm = 4
    noise_var_db = -20
    LLrout = tx_rx_process(inbits, Qm, noise_var_db)
    print("LLR output shape:", LLrout.shape)