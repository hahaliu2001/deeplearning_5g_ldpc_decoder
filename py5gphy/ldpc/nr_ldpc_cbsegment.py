# -*- coding:utf-8 -*-
"""5G NR LDPC code-block segmentation from TS 38.212 section 5.2.2.

Only the single-code-block path is supported by this module. That is enough for
the current simulation flow and avoids carrying the CB-CRC handling needed only
when a transport block is split into multiple code blocks.
"""

if __name__ == "__main__" and not __package__:
    from pathlib import Path
    import sys

    # Allow direct IDE execution while preserving normal package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from py5gphy.ldpc import ldpc_info

def ldpc_cbsegment(inbits, bgn):
    """Prepare one transport block as a single LDPC code block.

    This implementation supports only the ``C == 1`` case from TS 38.212
    section 5.2.2. Inputs that require multiple code blocks are rejected.

    Args:
        inbits: One-dimensional transport-block bit array with length ``B``.
        bgn: LDPC base graph number, either 1 or 2.

    Returns:
        cbs: Code-block array with shape ``(1, K)``. ``K`` is the LDPC encoder
            input length, and filler bits are written as ``-1`` at the end.
        Zc: Selected lifting size.
    """
    B = inbits.size
    assert bgn in [1,2]

    # Compute the TS 38.212 segmentation parameters. ``cbz`` is the number of
    # input bits copied into the code block, while ``K`` includes filler bits.
    C, cbz, L, F, K, Zc = ldpc_info.get_cbs_info(B, bgn)

    # Multi-code-block transport blocks require a 24B CB-CRC on each segment.
    # That path is intentionally not supported here.
    if C != 1:
        raise NotImplementedError("Only one LDPC code block is supported")

    # Start with all filler bits. The encoder treats -1 positions as NULL bits:
    # they are not transmitted, and they are converted to zero for parity math.
    cbs = -1 * np.ones((1, K),'i1')

    # For C == 1, the input bits are copied directly with no code-block CRC.
    cbs[0,0:cbz] = inbits
    return cbs, Zc


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    for bgn, B in [(1, 1200), (2, 600)]:
        inbits = rng.integers(0, 2, size=B, dtype=np.int8)
        cbs, Zc = ldpc_cbsegment(inbits, bgn)
        filler_count = int(np.sum(cbs == -1))

        print(f"BG{bgn} example")
        print(f"  input bits : {B}")
        print(f"  CB shape   : {cbs.shape}")
        print(f"  Zc         : {Zc}")
        print(f"  fillers    : {filler_count}")
