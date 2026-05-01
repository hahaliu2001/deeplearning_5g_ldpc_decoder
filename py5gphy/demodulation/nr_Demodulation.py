# -*- coding:utf-8 -*-
import numpy as np
import math
from py5gphy.demodulation import demod_bpsk
from py5gphy.demodulation import demod_pi2_bpsk
from py5gphy.demodulation import demod_qpsk
from py5gphy.demodulation import demod_16qam
from py5gphy.demodulation import demod_64qam
from py5gphy.demodulation import demod_256qam
from py5gphy.demodulation import demod_1024qam

def nrDemodulate(insymbols, modtype,noise_var):
    """ demodulation refer to  38.211 5.1 Modulation mapper
    the LLR demodulation algorithm  is in docs/algorithm/LLR_demodulation_for5G_EN.pdf
    it return both hard bits and LLR soft bits
    input:
    insymbols: np.array of channel equlization output 
    modtype: ["pi/2-bpsk", "bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam"]
    noise_var: noise variance array,noise_var.size should be equal with insymbols.size
    """
    #to make sure the input is one dimension array
    insymbols = insymbols.reshape(insymbols.size)
    noise_var = noise_var.real.reshape(noise_var.size)
    noise_var=noise_var.astype('f')
    
    assert insymbols.size == noise_var.size

    modtype = modtype.lower()
    assert modtype in ["pi/2-bpsk", "bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam"], "modulation type is incorrect"
    if modtype == "bpsk":
        [hardbits, LLR] = demod_bpsk.demod(insymbols,noise_var)
    elif modtype == "pi/2-bpsk":
        [hardbits, LLR] = demod_pi2_bpsk.demod(insymbols,noise_var)
    elif modtype == "qpsk":
        [hardbits, LLR] = demod_qpsk.demod(insymbols,noise_var)
    elif modtype == "16qam":
        [hardbits, LLR] = demod_16qam.demod(insymbols,noise_var)
    elif modtype == "64qam":
        [hardbits, LLR] = demod_64qam.demod(insymbols,noise_var)
    elif modtype == "256qam":
        [hardbits, LLR] = demod_256qam.demod(insymbols,noise_var)
    elif modtype == "1024qam":
        [hardbits, LLR] = demod_1024qam.demod(insymbols,noise_var)
        
    return hardbits, LLR


if __name__ == "__main__":
    from scipy import io
    import os
    
    print("test nr De_modulation")
    from tests.demodulation import test_nr_Demodulation
    file_lists = test_nr_Demodulation.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_nr_Demodulation.test_nr_demodulation(filename)
    
