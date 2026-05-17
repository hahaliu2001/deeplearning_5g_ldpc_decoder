# -*- coding:utf-8 -*-
"""QPSK soft demodulator."""

import numpy as np
from py5gphy.demodulation._common import hardbits_from_llr

def demod(insymbols,noise_var):
    """Demodulate QPSK symbols and return hard bits plus LLRs."""
    A = 1 / np.sqrt(2)
    LLR = np.zeros(2*insymbols.size,dtype='f')
    # QPSK carries one bit on I and one bit on Q for each complex symbol.
    LLR[0::2] = 4*insymbols.real*A/noise_var.real
    LLR[1::2] = 4*insymbols.imag*A/noise_var.real
    hardbits = hardbits_from_llr(LLR)
    return hardbits,LLR
