# -*- coding:utf-8 -*-
"""Shared helpers for NR soft demodulators.

The public demodulation entry point accepts arrays in whatever shape the caller
uses. Individual demodulators operate on flat per-symbol arrays, so this module
keeps the input normalization and hard-decision convention in one place.
"""

import numpy as np

SUPPORTED_MODTYPES = (
    "pi/2-bpsk",
    "bpsk",
    "qpsk",
    "16qam",
    "64qam",
    "256qam",
    "1024qam",
)


def normalize_modtype(modtype):
    """Return a normalized modulation name and validate that it is supported."""
    modtype = modtype.lower()
    assert modtype in SUPPORTED_MODTYPES, "modulation type is incorrect"
    return modtype


def prepare_demod_inputs(insymbols, noise_var):
    """Flatten symbols and noise variance arrays for per-symbol demodulation."""
    # The demodulation formulas use only the real noise variance, even if the
    # caller passes a complex-valued array from an equalization pipeline.
    symbols = np.asarray(insymbols).reshape(np.asarray(insymbols).size)
    noise = np.asarray(noise_var).real.reshape(np.asarray(noise_var).size).astype("f")
    assert symbols.size == noise.size
    return symbols, noise


def hardbits_from_llr(llr):
    """Convert soft LLRs to hard bits using the NR convention: positive means 0."""
    return np.where(llr > 0, 0, 1)
