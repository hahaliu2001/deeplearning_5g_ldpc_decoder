# -*- coding:utf-8 -*-
"""BPSK soft demodulator."""

import numpy as np
from py5gphy.demodulation._common import hardbits_from_llr

def demod(insymbols,noise_var):
    """Demodulate BPSK symbols and return hard bits plus LLRs."""
    A = 1 / np.sqrt(2)
    # BPSK symbols are mapped onto the diagonal, so both I and Q carry the bit.
    LLR = 4*(insymbols.real+insymbols.imag)*A/noise_var
    hardbits = hardbits_from_llr(LLR)
    return hardbits,LLR
