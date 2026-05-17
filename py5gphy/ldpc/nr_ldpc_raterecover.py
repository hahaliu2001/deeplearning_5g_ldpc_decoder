# -*- coding:utf-8 -*-
"""LDPC rate-recovery helpers from TS 38.212 section 5.4.2."""

import numpy as np


def _deinterleave_llr(LLr_fe, Qm):
    """Undo the LDPC rate-match bit interleaver."""
    E = LLr_fe.size
    assert E % Qm == 0
    # This is the inverse of selected_bits.reshape(Qm, E//Qm).T.reshape(E).
    return LLr_fe.reshape(E // Qm, Qm).T.reshape(E)


def raterecover_ldpc(LLr_fe, Ncb, N, k0, Qm, Zc, K_apo, K):
    """Recover one LDPC code block after rate matching.

    The function reverses bit interleaving and circular-buffer bit selection.
    Parameter calculation such as ``Ncb`` and ``k0`` is intentionally kept
    outside this function.

    Args:
        LLr_fe: Length-``E`` rate-matched LLR sequence.
        Ncb: Circular-buffer length.
        N: Desired rate-recovered output length.
        k0: Circular-buffer starting position.
        Qm: Modulation order.
        Zc: LDPC lifting size.
        K_apo: First filler-bit position in the LDPC encoder input.
        K: LDPC encoder input length, including filler bits.

    Returns:
        Length-``N`` rate-recovered LLR sequence.
    """
    LLr_ek = _deinterleave_llr(LLr_fe, Qm)
    E = LLr_ek.size

    # Filler positions in dn are shifted left by 2*Zc because the first two
    # systematic column groups are punctured.
    filler_positions = np.arange(K_apo, K) - 2 * Zc
    is_filler = np.zeros(Ncb, dtype=bool)
    is_filler[filler_positions] = True

    # The circular buffer may wrap when E is larger than the number of
    # selectable positions. Match the C decoder's HARQ memory behavior by
    # summing repeated LLRs for the same position.
    summed_llr = np.zeros(Ncb)

    out_idx = 0
    circ_idx = 0
    while out_idx < E:
        pos = (k0 + circ_idx) % Ncb
        if not is_filler[pos]:
            # Reverse bit selection by accumulating each deinterleaved LLR back
            # into its circular-buffer position.
            summed_llr[pos] += LLr_ek[out_idx]
            out_idx += 1

        circ_idx += 1

    LLr_dn = np.zeros(N)
    LLr_dn[0:Ncb] = summed_llr

    # Filler bits are known zeros. Use a strong positive LLR so the decoder is
    # biased toward bit 0 at those positions.
    max_LLR = np.max(np.abs(LLr_fe)) * 10
    LLr_dn[filler_positions] = max_LLR
    return LLr_dn


if __name__ == "__main__":
    LLr_fe = np.array([4.0, -3.0, 2.0, 1.0, -2.0, 3.0])
    LLr_dn = raterecover_ldpc(
        LLr_fe,
        Ncb=8,
        N=8,
        k0=0,
        Qm=2,
        Zc=1,
        K_apo=9,
        K=9,
    )

    print("LDPC rate-recovery example")
    print(f"  input   : {LLr_fe.tolist()}")
    print(f"  output  : {np.round(LLr_dn, 2).tolist()}")
