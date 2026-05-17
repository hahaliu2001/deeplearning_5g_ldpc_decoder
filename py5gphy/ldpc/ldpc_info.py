# -*- coding:utf-8 -*-
"""LDPC dimension and parity-check matrix helpers."""

from pathlib import Path

import numpy as np


MAX_CODE_BLOCK_SIZE = {
    # Maximum code-block sizes Kcb from TS 38.212 section 5.2.2.
    1: 8448,
    2: 3840,
}

BASE_GRAPH_INFO_COLS = {
    # Number of systematic/filler column groups before LDPC parity generation.
    1: 22,
    2: 10,
}

BASE_GRAPH_MATRIX_SHAPE = {
    # Base graph dimensions before lifting by Zc.
    1: (46, 68),
    2: (42, 52),
}

# Allowed lifting sizes from TS 38.212 Table 5.3.2-1.
LIFTING_SIZES = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
    20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 208, 224, 240, 256, 288, 320, 352, 384,
]

LIFTING_SIZE_SETS = [
    # Each list is one iLS set from TS 38.212 Table 5.3.2-1.
    [2, 4, 8, 16, 32, 64, 128, 256],
    [3, 6, 12, 24, 48, 96, 192, 384],
    [5, 10, 20, 40, 80, 160, 320],
    [7, 14, 28, 56, 112, 224],
    [9, 18, 36, 72, 144, 288],
    [11, 22, 44, 88, 176, 352],
    [13, 26, 52, 104, 208],
    [15, 30, 60, 120, 240],
]


def _get_bg2_kb(B):
    """Return the BG2 Kb value from TS 38.212 section 5.2.2."""
    if B > 640:
        return 10
    if B > 560:
        return 9
    if B > 192:
        return 8
    return 6


def get_cbs_info(B, bgn):
    """Return LDPC code-block segmentation parameters.

    Args:
        B: Number of transport-block bits entering code-block segmentation.
        bgn: LDPC base graph number, either 1 or 2.

    Returns:
        C: Number of code blocks.
        cbz: Number of information bits per code block before CB-CRC and
            filler bits.
        L: Code-block CRC length. This is 0 for one code block and 24 for
            multiple code blocks.
        F: Number of filler bits per code block.
        K: LDPC encoder input length per code block, including filler bits.
        Zc: Selected lifting size.
    """
    assert bgn in [1, 2]

    Kcb = MAX_CODE_BLOCK_SIZE[bgn]
    if B <= Kcb:
        # A single code block does not carry a code-block CRC.
        L = 0
        C = 1
        Bd = B
    else:
        # Multiple code blocks each carry a 24B CRC, so Bd includes C CRCs.
        L = 24
        C = int(np.ceil(B / (Kcb - L)))
        Bd = B + C * L

    # The code assumes the transport block splits evenly across code blocks.
    # Existing ULSCH vectors used by this project satisfy this condition.
    assert (B % C) == 0
    assert (Bd % C) == 0

    cbz = B // C
    # Kd is the number of non-filler bits in each LDPC encoder input block.
    Kd = Bd // C

    if bgn == 1:
        Kb = 22
    else:
        Kb = _get_bg2_kb(B)

    # Select the smallest lifting size whose Kb*Zc can hold Kd non-filler bits.
    Zc = next(Z for Z in LIFTING_SIZES if Z * Kb >= Kd)
    K = BASE_GRAPH_INFO_COLS[bgn] * Zc
    F = K - Kd

    return C, cbz, L, F, K, Zc


def find_iLS(Zc):
    """Return the lifting-size set index ``iLS`` that contains ``Zc``."""
    for set_idx, lifting_set in enumerate(LIFTING_SIZE_SETS):
        if Zc in lifting_set:
            return set_idx

    # Preserve the historical sentinel value used by callers before asserting.
    return 255


def _expanded_identity(Zc, shift):
    """Return a ``Zc x Zc`` identity matrix circularly shifted right."""
    return np.roll(np.eye(Zc, dtype="i1"), shift, axis=1)


def getH(Zc, bgn, iLS):
    """Generate the expanded LDPC parity-check matrix.

    The compact base-graph table stores shift values. A non-negative table entry
    expands to a circularly shifted identity matrix; ``-1`` expands to an all
    zero block.
    """
    from scipy import io

    assert bgn in [1, 2]
    row_blocks, col_blocks = BASE_GRAPH_MATRIX_SHAPE[bgn]
    table_path = Path(__file__).resolve().parent / "tables" / f"BG{bgn}S{iLS}.mat1"
    # BG stores one shift value per base-graph block. SciPy is imported lazily
    # so callers that only need size calculations do not require SciPy.
    BG = io.loadmat(table_path)["BG"]

    H = np.zeros((row_blocks * Zc, col_blocks * Zc), dtype="i1")

    for row in range(row_blocks):
        for col in range(col_blocks):
            shift_value = BG[row, col]
            if shift_value < 0:
                # Undefined base-graph entries become all-zero lifted blocks.
                continue

            shift = int(shift_value % Zc)
            row_slice = slice(row * Zc, (row + 1) * Zc)
            col_slice = slice(col * Zc, (col + 1) * Zc)
            # Replace the compact shift value with its lifted circulant block.
            H[row_slice, col_slice] = _expanded_identity(Zc, shift)

    return H


def gen_ldpc_para(N, bgn):
    """Return ``H``, ``K``, and ``Zc`` for an encoded LDPC block length."""
    assert bgn in [1, 2]

    # Encoded length N excludes the first two punctured systematic columns, so
    # the divisor is 66 for BG1 and 50 for BG2.
    if bgn == 1:
        Zc = N // 66
    else:
        Zc = N // 50

    K = BASE_GRAPH_INFO_COLS[bgn] * Zc
    iLS = find_iLS(Zc)
    assert iLS < 8
    H = getH(Zc, bgn, iLS)

    return H, K, Zc


if __name__ == "__main__":
    for B, bgn in [(1200, 1), (600, 2)]:
        C, cbz, L, F, K, Zc = get_cbs_info(B, bgn)
        print(f"BG{bgn} CBS info")
        print(f"  B/C/cbz      : {B}/{C}/{cbz}")
        print(f"  L/F/K/Zc     : {L}/{F}/{K}/{Zc}")
        print(f"  iLS          : {find_iLS(Zc)}")

    try:
        H, K, Zc = gen_ldpc_para(132, 1)
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        print("getH example skipped: scipy is required to load LDPC tables")
    else:
        print(f"getH example: H={H.shape}, K={K}, Zc={Zc}")
