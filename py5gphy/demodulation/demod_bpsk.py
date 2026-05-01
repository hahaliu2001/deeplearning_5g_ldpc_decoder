# -*- coding:utf-8 -*-
import numpy as np
import math

def demod(insymbols,noise_var):
    """bpsk demodulation
    """
    A = 1/math.sqrt(2)
    LLR = 4*(insymbols.real+insymbols.imag)*A/noise_var
    hardbits = np.array([0 if a>0 else 1 for a in LLR])
    return hardbits,LLR