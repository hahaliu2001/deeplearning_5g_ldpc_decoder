# -*- coding:utf-8 -*-
"""Top-level 5G NR soft demodulation dispatcher."""

if __name__ == "__main__" and not __package__:
    from pathlib import Path
    import sys

    # Allow this file to be run directly from an IDE while keeping package
    # imports unchanged for normal library use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from py5gphy.demodulation import demod_bpsk
from py5gphy.demodulation import demod_pi2_bpsk
from py5gphy.demodulation import demod_qpsk
from py5gphy.demodulation import demod_16qam
from py5gphy.demodulation import demod_64qam
from py5gphy.demodulation import demod_256qam
from py5gphy.demodulation import demod_1024qam
from py5gphy.demodulation._common import normalize_modtype, prepare_demod_inputs


DEMODULATORS = {
    "pi/2-bpsk": demod_pi2_bpsk.demod,
    "bpsk": demod_bpsk.demod,
    "qpsk": demod_qpsk.demod,
    "16qam": demod_16qam.demod,
    "64qam": demod_64qam.demod,
    "256qam": demod_256qam.demod,
    "1024qam": demod_1024qam.demod,
}


def nrDemodulate(insymbols, modtype,noise_var):
    """Demodulate NR symbols into hard bits and soft LLRs.

    The demodulation follows TS 38.211 section 5.1. The LLR expressions are
    documented in docs/algorithm/LLR_demodulation_for5G_EN.pdf.

    Args:
        insymbols: Equalized complex input symbols.
        modtype: One of "pi/2-bpsk", "bpsk", "qpsk", "16qam", "64qam",
            "256qam", or "1024qam".
        noise_var: Per-symbol noise variance. It must have the same number of
            elements as ``insymbols``.

    Returns:
        hardbits: Hard decisions derived from the LLR sign.
        LLR: Soft bit log-likelihood ratios.
    """
    # Normalize once here so the individual demodulators can focus only on the
    # modulation-specific LLR expressions.
    insymbols, noise_var = prepare_demod_inputs(insymbols, noise_var)
    modtype = normalize_modtype(modtype)
    return DEMODULATORS[modtype](insymbols, noise_var)


if __name__ == "__main__":
    import numpy as np
    from py5gphy.common.nrModulation import MODULATION_ORDER, nrModulate

    print("NR demodulation examples")

    for modtype, Qm in MODULATION_ORDER.items():
        bits = np.arange(2 * Qm) % 2
        symbols = nrModulate(bits, modtype)
        noise_var = 0.05 * np.ones(symbols.size, dtype="f")
        hardbits, LLR = nrDemodulate(symbols, modtype, noise_var)

        print(f"\n{modtype}")
        print(f"  input bits : {bits.tolist()}")
        print(f"  symbols    : {np.round(symbols, 3).tolist()}")
        print(f"  hard bits  : {hardbits.tolist()}")
        print(f"  first LLRs : {np.round(LLR[:min(6, LLR.size)], 2).tolist()}")

    rng = np.random.default_rng(0)
    bits = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
    symbols = nrModulate(bits, "qpsk")
    noise_var = 0.1 * np.ones(symbols.size, dtype="f")
    noisy_symbols = symbols + 0.05 * (
        rng.standard_normal(symbols.size) + 1j * rng.standard_normal(symbols.size)
    )
    hardbits, LLR = nrDemodulate(noisy_symbols, "qpsk", noise_var)

    print("\nnoisy qpsk")
    print(f"  input bits : {bits.tolist()}")
    print(f"  hard bits  : {hardbits.tolist()}")
    print(f"  LLR        : {np.round(LLR, 2).tolist()}")
