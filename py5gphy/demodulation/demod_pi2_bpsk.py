# -*- coding:utf-8 -*-
import numpy as np
import math

def demod(insymbols,noise_var):
    """pi/2-bpsk demodulation
    """
    A = 1/math.sqrt(2)
    LLR = np.zeros(insymbols.size,dtype='f')
    LLR[0::2] = 4*(insymbols[0::2].real+insymbols[0::2].imag)*A/noise_var[0::2]
    LLR[1::2] = 4*(-insymbols[1::2].real+insymbols[1::2].imag)*A/noise_var[1::2]
    hardbits = np.array([0 if a>0 else 1 for a in LLR])
    return hardbits,LLR