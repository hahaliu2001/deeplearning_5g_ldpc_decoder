# -*- coding:utf-8 -*-
import numpy as np
import math

def demod(insymbols,noise_var):
    """16QAM demodulation
    """
    
    A = 1/math.sqrt(10)
    LLR = np.zeros(4*insymbols.size,dtype='f')
    LLR[0::4] = LLR_16qam_bit_0_1(insymbols.real,noise_var,A)
    LLR[1::4] = LLR_16qam_bit_0_1(insymbols.imag,noise_var,A)
    
    LLR[2::4] = LLR_16qam_bit_2_3(insymbols.real,noise_var,A)
    LLR[3::4] = LLR_16qam_bit_2_3(insymbols.imag,noise_var,A)

    hardbits = np.array([0 if a>0 else 1 for a in LLR])
    return hardbits,LLR

def LLR_16qam_bit_0_1(insym, noise_var,A):
    """ 16QAM b0 and b1 LLR"""
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
    """ 16QAM b2 and b3 LLR"""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < 0:
            LLR[m] = 4*A*(r+2*A)/noise_var[m]
        else:
            LLR[m] = 4*A*(-r+2*A)/noise_var[m]
    
    return LLR