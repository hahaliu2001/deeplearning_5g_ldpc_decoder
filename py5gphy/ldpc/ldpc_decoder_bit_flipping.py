# -*- coding:utf-8 -*-
"""Bit-flipping LDPC decoder, hardbits decoding, worst performance"""

if __name__ == "__main__" and not __package__:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from py5gphy.crc import crc


def _hard_decision(llr):
    """Convert LLR values to hard bits: positive means 0, negative means 1."""
    # A zero LLR is treated as bit 0, matching the convention used by the soft
    # LDPC decoders in this package.
    bits = np.zeros(llr.size, dtype="i1")
    bits[llr < 0] = 1
    return bits


def _syndrome(H, bits):
    """Return the parity-check syndrome for the current hard decisions."""
    # A zero syndrome means all parity checks pass.
    return (H @ bits.T) % 2


def _flip_metric(H, syndrome):
    """Return the bit-flipping metric for each variable node.

    For irregular LDPC matrices, ``S @ H`` alone is not sufficient. The
    ``2*S - 1`` term rewards bits connected to failed checks and penalizes bits
    connected to satisfied checks.
    """
    # Unsatisfied checks have syndrome 1 and contribute +1; satisfied checks
    # have syndrome 0 and contribute -1. Bits with larger metrics are connected
    # to more failed checks and are stronger flip candidates.
    return (2 * syndrome - 1) @ H


def ldpc_decoder_BF(LLRin, H, crcpoly, B, L):
    """Decode one LDPC code block with the bit-flipping algorithm.

    Args:
        LLRin: Length-``N`` decoder input LLR sequence.
        H: ``M x N`` LDPC parity-check matrix.
        crcpoly: CRC polynomial name accepted by ``crc.nr_crc_decode``.
        B: Number of decoded bits to check with CRC.
        L: Maximum number of bit-flipping iterations.

    Returns:
        decoded_bits: CRC-stripped payload bits when CRC passes; otherwise the
            first ``B`` hard-decision bits from the final iteration.
        status: ``True`` if decoding succeeds, otherwise ``False``.
    """
    num_cols = H.shape[1]
    assert LLRin.size == num_cols

    # Start from the channel hard decision before applying parity feedback.
    hard_bits = _hard_decision(LLRin)
    decoded_bits = hard_bits[:B].copy()

    for _ in range(L):
        # Check whether the current hard decision is already a valid codeword.
        syndrome = _syndrome(H, hard_bits)
        if not np.any(syndrome):
            return hard_bits, True

        # Rank all variable nodes by how strongly they are connected to failed
        # parity checks, then flip every bit tied for the largest metric.
        metric = _flip_metric(H, syndrome)
        max_metric = np.max(metric)
        hard_bits[metric == max_metric] = 1 - hard_bits[metric == max_metric]

        # Use CRC as the early-stop criterion for the decoded payload bits.
        decoded_bits = hard_bits[:B].copy()
        blk, status = crc.nr_crc_decode(decoded_bits, crcpoly)
        if status == 0:
            return blk, True

    # Decoding failed within the iteration budget. Return the best current hard
    # decisions for the payload region so callers can inspect the failure.
    return decoded_bits, False


if __name__ == "__main__":
    # Tiny parity-check example, useful for direct IDE execution. The example
    # only exercises the bit-flipping loop; production LDPC matrices are loaded
    # through py5gphy.ldpc.ldpc_info.
    H = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 1],
        ],
        dtype="i1",
    )
    llr = np.array([3.0, -2.0, 2.0, 2.5], dtype=np.float32)
    hard_bits = _hard_decision(llr)
    syndrome = _syndrome(H, hard_bits)
    metric = _flip_metric(H, syndrome)

    print("LDPC bit-flipping helper example")
    print(f"  hard bits : {hard_bits.tolist()}")
    print(f"  syndrome  : {syndrome.tolist()}")
    print(f"  metric    : {metric.tolist()}")
