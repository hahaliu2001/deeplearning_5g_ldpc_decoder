# -*- coding:utf-8 -*-
"""64QAM soft demodulator."""

import numpy as np
from py5gphy.demodulation._common import hardbits_from_llr

def demod(insymbols,noise_var):
    """Demodulate 64QAM symbols and return hard bits plus LLRs."""
    A = 1 / np.sqrt(42)
    LLR = np.zeros(6*insymbols.size,dtype='f')
    # Each helper handles one significance level on one real-valued axis. The
    # same expression is used for I and Q, then interleaved into NR bit order.
    LLR[0::6] = LLR_64qam_bit_0_1(insymbols.real,noise_var,A)
    LLR[1::6] = LLR_64qam_bit_0_1(insymbols.imag,noise_var,A)
    
    LLR[2::6] = LLR_64qam_bit_2_3(insymbols.real,noise_var,A)
    LLR[3::6] = LLR_64qam_bit_2_3(insymbols.imag,noise_var,A)

    LLR[4::6] = LLR_64qam_bit_4_5(insymbols.real,noise_var,A)
    LLR[5::6] = LLR_64qam_bit_4_5(insymbols.imag,noise_var,A)

    hardbits = hardbits_from_llr(LLR)
    return hardbits,LLR

def LLR_64qam_bit_0_1(insym, noise_var,A):
    """Calculate MSB LLRs for the I or Q axis of a 64QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -6*A:
            LLR[m] = 16*A*(r+3*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 12*A*(r+2*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 8*A*(r+A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 4*A*r/noise_var[m]
        elif r < 4*A:
            LLR[m] = 8*A*(r-A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 12*A*(r-2*A)/noise_var[m]
        else:
            LLR[m] = 16*A*(r-3*A)/noise_var[m]
        
    return LLR

def LLR_64qam_bit_2_3(insym, noise_var,A):
    """Calculate middle-bit LLRs for the I or Q axis of a 64QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -6*A:
            LLR[m] = 8*A*(r+5*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 4*A*(r+4*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 8*A*(r+3*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 8*A*(-r+3*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 4*A*(-r+4*A)/noise_var[m]
        else:
            LLR[m] = 8*A*(-r+5*A)/noise_var[m]
    
    return LLR

def LLR_64qam_bit_4_5(insym, noise_var,A):
    """Calculate LSB LLRs for the I or Q axis of a 64QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -4*A:
            LLR[m] = 4*A*(r+6*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 4*A*(-r-2*A)/noise_var[m]
        elif r < 4*A:
            LLR[m] = 4*A*(r-2*A)/noise_var[m]
        else:
            LLR[m] = 4*A*(-r+6*A)/noise_var[m]
    
    return LLR
