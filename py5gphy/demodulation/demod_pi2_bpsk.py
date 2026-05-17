# -*- coding:utf-8 -*-
"""pi/2-BPSK soft demodulator."""

import numpy as np
from py5gphy.demodulation._common import hardbits_from_llr

def demod(insymbols,noise_var):
    """Demodulate pi/2-BPSK symbols and return hard bits plus LLRs."""
    A = 1 / np.sqrt(2)
    LLR = np.zeros(insymbols.size,dtype='f')
    # Odd symbols are rotated by pi/2, which flips the sign contribution of I.
    LLR[0::2] = 4*(insymbols[0::2].real+insymbols[0::2].imag)*A/noise_var[0::2]
    LLR[1::2] = 4*(-insymbols[1::2].real+insymbols[1::2].imag)*A/noise_var[1::2]
    hardbits = hardbits_from_llr(LLR)
    return hardbits,LLR
