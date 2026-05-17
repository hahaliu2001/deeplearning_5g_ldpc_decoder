# -*- coding:utf-8 -*-
"""256QAM soft demodulator."""

import numpy as np
from py5gphy.demodulation._common import hardbits_from_llr

def demod(insymbols,noise_var):
    """Demodulate 256QAM symbols and return hard bits plus LLRs."""
    A = 1 / np.sqrt(170)
    LLR = np.zeros(8*insymbols.size,dtype='f')
    # Compute LLRs by significance level. Each helper is reused for the I and Q
    # axes, then the results are interleaved into the NR bit order.
    LLR[0::8] = LLR_256qam_bit_0_1(insymbols.real,noise_var,A)
    LLR[1::8] = LLR_256qam_bit_0_1(insymbols.imag,noise_var,A)
    
    LLR[2::8] = LLR_256qam_bit_2_3(insymbols.real,noise_var,A)
    LLR[3::8] = LLR_256qam_bit_2_3(insymbols.imag,noise_var,A)

    LLR[4::8] = LLR_256qam_bit_4_5(insymbols.real,noise_var,A)
    LLR[5::8] = LLR_256qam_bit_4_5(insymbols.imag,noise_var,A)

    LLR[6::8] = LLR_256qam_bit_6_7(insymbols.real,noise_var,A)
    LLR[7::8] = LLR_256qam_bit_6_7(insymbols.imag,noise_var,A)

    hardbits = hardbits_from_llr(LLR)
    return hardbits,LLR

def LLR_256qam_bit_0_1(insym, noise_var,A):
    """Calculate MSB LLRs for the I or Q axis of a 256QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -14*A:
            LLR[m] = 32*A*(r+7*A)/noise_var[m]
        elif r < -12*A:
            LLR[m] = 28*A*(r+6*A)/noise_var[m]
        elif r < -10*A:
            LLR[m] = 24*A*(r+5*A)/noise_var[m]
        elif r < -8*A:
            LLR[m] = 20*A*(r+4*A)/noise_var[m]
        elif r < -6*A:
            LLR[m] = 16*A*(r+3*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 12*A*(r+2*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 8*A*(r+1*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 4*A*r/noise_var[m]
        elif r < 4*A:
            LLR[m] = 8*A*(r-1*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 12*A*(r-2*A)/noise_var[m]
        elif r < 8*A:
            LLR[m] = 16*A*(r-3*A)/noise_var[m]
        elif r < 10*A:
            LLR[m] = 20*A*(r-4*A)/noise_var[m]
        elif r < 12*A:
            LLR[m] = 24*A*(r-5*A)/noise_var[m]
        elif r < 14*A:
            LLR[m] = 28*A*(r-6*A)/noise_var[m]
        else:
            LLR[m] = 32*A*(r-7*A)/noise_var[m]
        
    return LLR

def LLR_256qam_bit_2_3(insym, noise_var,A):
    """Calculate second-level LLRs for the I or Q axis of a 256QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -14*A:
            LLR[m] = 16*A*(r+11*A)/noise_var[m]
        elif r < -12*A:
            LLR[m] = 12*A*(r+10*A)/noise_var[m]
        elif r < -10*A:
            LLR[m] = 8*A*(r+9*A)/noise_var[m]
        elif r < -6*A:
            LLR[m] = 4*A*(r+8*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 8*A*(r+7*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 12*A*(r+6*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 16*A*(r+5*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 16*A*(-r+5*A)/noise_var[m]
        elif r < 4*A:
            LLR[m] = 12*A*(-r+6*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 8*A*(-r+7*A)/noise_var[m]
        elif r < 10*A:
            LLR[m] = 4*A*(-r+8*A)/noise_var[m]
        elif r < 12*A:
            LLR[m] = 8*A*(-r+9*A)/noise_var[m]
        elif r < 14*A:
            LLR[m] = 12*A*(-r+10*A)/noise_var[m]
        else:
            LLR[m] = 16*A*(-r+11*A)/noise_var[m]
    
    return LLR

def LLR_256qam_bit_4_5(insym, noise_var,A):
    """Calculate third-level LLRs for the I or Q axis of a 256QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -14*A:
            LLR[m] = 8*A*(r+13*A)/noise_var[m]
        elif r < -10*A:
            LLR[m] = 4*A*(r+12*A)/noise_var[m]
        elif r < -8*A:
            LLR[m] = 8*A*(r+11*A)/noise_var[m]
        elif r < -6*A:
            LLR[m] = 8*A*(-r-5*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 4*A*(-r-4*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 8*A*(-r-3*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 8*A*(r-3*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 4*A*(r-4*A)/noise_var[m]
        elif r < 8*A:
            LLR[m] = 8*A*(r-5*A)/noise_var[m]
        elif r < 10*A:
            LLR[m] = 8*A*(-r+11*A)/noise_var[m]
        elif r < 14*A:
            LLR[m] = 4*A*(-r+12*A)/noise_var[m]
        else:
            LLR[m] = 8*A*(-r+13*A)/noise_var[m]
    
    return LLR

def LLR_256qam_bit_6_7(insym, noise_var,A):
    """Calculate LSB LLRs for the I or Q axis of a 256QAM symbol."""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -12*A:
            LLR[m] = 4*A*(r+14*A)/noise_var[m]
        elif r < -8*A:
            LLR[m] = 4*A*(-r-10*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 4*A*(r+6*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 4*A*(-r-2*A)/noise_var[m]
        elif r < 4*A:
            LLR[m] = 4*A*(r-2*A)/noise_var[m]
        elif r < 8*A:
            LLR[m] = 4*A*(-r+6*A)/noise_var[m]
        elif r < 12*A:
            LLR[m] = 4*A*(r-10*A)/noise_var[m]
        else:
            LLR[m] = 4*A*(-r+14*A)/noise_var[m]
    
    return LLR
