# -*- coding:utf-8 -*-
"""5G NR LDPC encoder from TS 38.212 section 5.3.2."""

if __name__ == "__main__" and not __package__:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from py5gphy.ldpc import ldpc_info


def _get_encoder_sizes(K, bgn):
    """Return lifting size ``Zc`` and encoded length ``N`` for one code block."""
    assert bgn in [1, 2]

    if bgn == 1:
        # Base graph 1 has K = 22*Zc information/filler positions and produces
        # 66*Zc transmitted bits after LDPC encoding.
        assert K % 22 == 0
        Zc = K // 22
        N = 66 * Zc
    else:
        # Base graph 2 has K = 10*Zc information/filler positions and produces
        # 50*Zc transmitted bits after LDPC encoding.
        assert K % 10 == 0
        Zc = K // 10
        N = 50 * Zc

    return Zc, N


def _prepare_systematic_bits(ck, Zc, N):
    """Build the transmitted systematic part and replace filler bits for parity.

    The first ``2*Zc`` information bits are punctured by TS 38.212, so they are
    not copied into the transmitted sequence. Filler bits are represented as
    ``-1`` in ``ck``; they are omitted from ``dn`` and replaced by zero for the
    parity equations.
    """
    parity_input = ck.copy()
    dn = -np.ones(N, dtype="i1")

    for k in range(2 * Zc, ck.size):
        if parity_input[k] == -1:
            # Filler bits do not enter the transmitted sequence, but they are
            # treated as zero while solving the parity equations.
            parity_input[k] = 0
        else:
            # Shift by 2*Zc because the first two systematic columns are
            # punctured and therefore absent from dn.
            dn[k - 2 * Zc] = parity_input[k]

    return dn, parity_input


def _load_parity_check_matrix(Zc, bgn):
    """Load the expanded LDPC parity-check matrix for ``Zc`` and base graph."""
    # The lifting size determines which 38.212 lifting-size set is used.
    iLS = ldpc_info.find_iLS(Zc)
    assert iLS < 8
    return ldpc_info.getH(Zc, bgn, iLS)


def _insert_parity_bits(dn, parity_bits, K, Zc):
    """Insert generated parity bits after the transmitted systematic region."""
    # Parity bits are indexed as if the two punctured systematic columns were
    # still present, so the same 2*Zc shift is used here.
    for k in range(K, dn.size + 2 * Zc):
        dn[k - 2 * Zc] = parity_bits[k - K]
    return dn


def encode_ldpc(ck, bgn):
    """Encode one LDPC code block with the optimized parity-bit method.

    Args:
        ck: Length-``K`` LDPC encoder input block. Filler bits are marked as
            ``-1``.
        bgn: LDPC base graph number, either 1 or 2.

    Returns:
        Length-``N`` encoded sequence after puncturing the first ``2*Zc`` bits.
    """
    K = ck.size
    # Derive all LDPC dimensions from the code-block length and base graph.
    Zc, N = _get_encoder_sizes(K, bgn)

    # Build the transmitted systematic region and the zero-filled vector used
    # by the parity equations.
    dn, parity_input = _prepare_systematic_bits(ck, Zc, N)

    H = _load_parity_check_matrix(Zc, bgn)
    parity_bits = _gen_ldpc_parity_bit(H, parity_input, Zc, K, N, bgn)
    return _insert_parity_bits(dn, parity_bits, K, Zc)


def _gen_ldpc_parity_bit(H, ck, Zc, K, N, bgn):
    """Generate LDPC parity bits with the optimized 5G NR encoder structure."""
    # This optimization follows "Flexible 5G New Radio LDPC Encoder Optimized
    # for High Hardware Usage Efficiency" and docs/LDPC_encoder_optimization.pdf.
    # A and B are the first four check-node block rows used to solve the first
    # four parity sub-blocks; C generates the remaining parity sub-blocks.
    A = H[0 : 4 * Zc, 0:K]
    B = H[0 : 4 * Zc, K : K + 4 * Zc]
    C = H[4 * Zc :, 0 : K + 4 * Zc]

    # L1 is split into four Zc-length rows so the equations match the first
    # four block rows of the lifted parity-check matrix.
    L1 = (A @ ck.T).reshape((4, -1))
    L2 = np.sum(L1, axis=0) % 2

    if bgn == 1:
        # BG1 and BG2 use different pivot rows when solving the first parity
        # sub-block. After pc1 is known, pc2/pc3/pc4 follow from B.
        zeros_idx = np.nonzero(B[Zc, 0:Zc])[0][0]
        pc1 = np.roll(L2, zeros_idx)
        pc2 = (L1[0, :] + B[0:Zc, 0:Zc] @ pc1.T) % 2
        pc4 = (L1[3, :] + B[3 * Zc : 4 * Zc, 0:Zc] @ pc1.T) % 2
        pc3 = (L1[2, :] + B[2 * Zc : 3 * Zc, 3 * Zc : 4 * Zc] @ pc4.T) % 2
    else:
        # BG2 uses the third block row as the pivot for pc1 and derives pc3
        # from pc2 instead of pc4.
        zeros_idx = np.nonzero(B[2 * Zc, 0:Zc])[0][0]
        pc1 = np.roll(L2, zeros_idx)
        pc2 = (L1[0, :] + B[0:Zc, 0:Zc] @ pc1.T) % 2
        pc4 = (L1[3, :] + B[3 * Zc : 4 * Zc, 0:Zc] @ pc1.T) % 2
        pc3 = (L1[1, :] + B[Zc : 2 * Zc, Zc : 2 * Zc] @ pc2.T) % 2

    first_parity = np.array([pc1, pc2, pc3, pc4]).reshape(-1)
    extended_input = np.concatenate((ck, first_parity))

    # Once the first four parity sub-blocks are known, the lower block rows
    # generate the remaining parity bits with one matrix product.
    remaining_parity = (C @ extended_input.T) % 2
    return np.concatenate((first_parity, remaining_parity)).astype("i1")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    for bgn, Zc in [(1, 2), (2, 2)]:
        K = (22 if bgn == 1 else 10) * Zc
        _, N = _get_encoder_sizes(K, bgn)
        ck = rng.integers(0, 2, size=K, dtype=np.int8)
        dn, parity_input = _prepare_systematic_bits(ck, Zc, N)

        print(f"BG{bgn} encoder setup example")
        print(f"  K/Zc/N        : {K}/{Zc}/{N}")
        print(f"  dn shape      : {dn.shape}")
        print(f"  parity input  : {parity_input.shape}")

        try:
            encoded_bits = encode_ldpc(ck, bgn)
        except ModuleNotFoundError as exc:
            if exc.name != "scipy":
                raise
            print("  encode_ldpc   : skipped, scipy is required to load LDPC tables")
        else:
            print(f"  encode_ldpc   : {encoded_bits.shape}")
            print(f"  first bits    : {encoded_bits[:min(12, encoded_bits.size)].tolist()}")
