# -*- coding:utf-8 -*-
"""16QAM soft demodulator."""

import numpy as np
from py5gphy.demodulation._common import hardbits_from_llr

def demod(insymbols,noise_var):
    """Demodulate 16QAM symbols and return hard bits plus LLRs."""
    A = 1 / np.sqrt(10)
    LLR = np.zeros(4*insymbols.size,dtype='f')
    # LLR output follows the NR bit order: I-axis bit, Q-axis bit, repeated by
    # significance level for each modulation symbol.
    LLR[0::4] = LLR_16qam_bit_0_1(insymbols.real,noise_var,A)
    LLR[1::4] = LLR_16qam_bit_0_1(insymbols.imag,noise_var,A)
    
    LLR[2::4] = LLR_16qam_bit_2_3(insymbols.real,noise_var,A)
    LLR[3::4] = LLR_16qam_bit_2_3(insymbols.imag,noise_var,A)

    hardbits = hardbits_from_llr(LLR)
    return hardbits,LLR

def LLR_16qam_bit_0_1(insym, noise_var,A):
    """Calculate MSB LLRs for the I or Q axis of a 16QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -2*A:
            LLR[m] = 8*A*(r+A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 4*A*r/noise_var[m]
        else:
            LLR[m] = 8*A*(r-A)/noise_var[m]
    
    return LLR

def LLR_16qam_bit_2_3(insym, noise_var,A):
    """Calculate LSB LLRs for the I or Q axis of a 16QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < 0:
            LLR[m] = 4*A*(r+2*A)/noise_var[m]
        else:
            LLR[m] = 4*A*(-r+2*A)/noise_var[m]
    
    return LLR
