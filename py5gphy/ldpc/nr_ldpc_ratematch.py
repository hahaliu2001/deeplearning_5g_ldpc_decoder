# -*- coding:utf-8 -*-
"""LDPC rate matching helpers from TS 38.212 section 5.4.2."""

import math

import numpy as np


K0_FACTORS = {
    # Numerators from TS 38.212 Table 5.4.2.1-2 for each redundancy version.
    1: {0: 0, 1: 17, 2: 33, 3: 56},
    2: {0: 0, 1: 13, 2: 25, 3: 43},
}

BG_DENOMINATORS = {
    # Encoded LDPC lengths after puncturing are 66*Zc for BG1 and 50*Zc for BG2.
    1: 66,
    2: 50,
}


def get_Er_ldpc(G, C, Qm, NL):
    """Return the rate-matched length for each LDPC code block.

    Args:
        G: Total number of coded bits available for the transport block.
        C: Number of code blocks. CBGTI is not modeled here.
        Qm: Modulation order.
        NL: Number of transmission layers.

    Returns:
        List of length ``C`` containing each code block's rate-match length.
    """
    # Each Er must be a multiple of NL*Qm so the output maps cleanly onto
    # modulation symbols and transmission layers.
    bits_per_symbol_group = NL * Qm
    base = bits_per_symbol_group * math.floor(G / (bits_per_symbol_group * C))
    extra = bits_per_symbol_group * math.ceil(G / (bits_per_symbol_group * C))
    num_base_blocks = C - ((G / bits_per_symbol_group) % C)

    return [
        base if block_idx <= num_base_blocks - 1 else extra
        for block_idx in range(C)
    ]


def get_k0(Ncb, bgn, rv, Zc):
    """Return the circular-buffer starting position ``k0``.

    Args:
        Ncb: Circular-buffer length.
        bgn: LDPC base graph number, either 1 or 2.
        rv: Redundancy version, one of 0, 1, 2, or 3.
        Zc: LDPC lifting size.

    Returns:
        Starting position for bit selection.
    """
    assert rv in [0, 1, 2, 3]
    assert bgn in [1, 2]

    factor = K0_FACTORS[bgn][rv]
    if factor == 0:
        return 0

    # k0 is quantized to a lifting-size boundary.
    return math.floor(factor * Ncb / (BG_DENOMINATORS[bgn] * Zc)) * Zc


def ratematch_ldpc(dn, Ncb, E, k0, Qm):
    """Rate-match one LDPC code block.

    The function performs the TS 38.212 bit-selection step followed by bit
    interleaving. Parameter calculation such as ``Ncb``, ``E``, and ``k0`` is
    intentionally kept outside this function.

    Args:
        dn: Encoded LDPC sequence. Filler bits are marked as ``-1``.
        Ncb: Circular-buffer length.
        E: Desired rate-matched output length.
        k0: Circular-buffer starting position.
        Qm: Modulation order.

    Returns:
        Length-``E`` rate-matched bit sequence.
    """
    assert dn.size >= Ncb
    assert E % Qm == 0

    selected_bits = np.zeros(E, dtype="i1")

    out_idx = 0
    circ_idx = 0
    while out_idx < E:
        pos = (k0 + circ_idx) % Ncb
        if dn[pos] != -1:
            # Filler bits are skipped during circular-buffer selection.
            selected_bits[out_idx] = dn[pos]
            out_idx += 1
        circ_idx += 1

    # Interleave bits so adjacent modulation symbols receive the correct bit
    # positions for the selected Qm.
    return selected_bits.reshape(Qm, E // Qm).T.reshape(E).astype("i1")


if __name__ == "__main__":
    dn = np.array([0, 1, -1, 1, 0, 1, 0, -1], dtype="i1")
    k0 = get_k0(Ncb=dn.size, bgn=1, rv=0, Zc=2)
    fe = ratematch_ldpc(dn, Ncb=dn.size, E=6, k0=k0, Qm=2)

    print("LDPC rate-match example")
    print(f"  k0      : {k0}")
    print(f"  output  : {fe.tolist()}")
