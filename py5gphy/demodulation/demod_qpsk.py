# -*- coding:utf-8 -*-
import numpy as np
import math

def demod(insymbols,noise_var):
    """QPSK demodulation
    """
    
    A = 1/math.sqrt(2)
    LLR = np.zeros(2*insymbols.size,dtype='f')
    LLR[0::2] = 4*insymbols.real*A/noise_var.real
    LLR[1::2] = 4*insymbols.imag*A/noise_var.real
    hardbits = np.array([0 if a>0 else 1 for a in LLR])
    return hardbits,LLR