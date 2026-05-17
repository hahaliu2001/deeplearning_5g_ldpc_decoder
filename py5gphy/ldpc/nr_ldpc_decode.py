# -*- coding:utf-8 -*-
"""5G NR LDPC decoders from TS 38.212 section 5.3.2."""

if __name__ == "__main__" and not __package__:
    from pathlib import Path
    import sys

    # Allow direct IDE execution while keeping package imports unchanged.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from py5gphy.crc import crc
from py5gphy.ldpc import ldpc_decoder_bit_flipping
from py5gphy.ldpc import ldpc_info


DEFAULT_DECODER_CONFIG = {
    # L is the maximum iteration count. alpha/beta configure normalized and
    # offset min-sum behavior when a min-sum algorithm is selected.
    "L": 32,
    "algo": "min-sum",
    "alpha": 0.7,
    "beta": 0,
}

SUPPORTED_ALGOS = ["BF", "BP", "min-sum", "layered-BP", "layered-min-sum"]


def _decoder_config(user_config):
    """Merge caller-provided decoder settings with the defaults."""
    config = DEFAULT_DECODER_CONFIG.copy()
    if user_config is not None:
        config.update(user_config)
    return config


def _ldpc_dimensions(Zc, bgn):
    """Return transmitted length ``N`` and encoder input length ``K``."""
    assert bgn in [1, 2]
    if bgn == 1:
        # BG1 transmits 66*Zc bits after puncturing and has K = 22*Zc.
        return 66 * Zc, 22 * Zc
    # BG2 transmits 50*Zc bits after puncturing and has K = 10*Zc.
    return 50 * Zc, 10 * Zc


def _load_parity_check_matrix(Zc, bgn):
    """Load the expanded parity-check matrix for the selected lifting size."""
    # Zc determines the lifting-size set iLS used by the base-graph table.
    iLS = ldpc_info.find_iLS(Zc)
    assert iLS < 8
    return ldpc_info.getH(Zc, bgn, iLS)


def _prepend_punctured_llr(LLRin, Zc):
    """Restore the two punctured systematic columns with neutral LLR values."""
    # The first two systematic column groups are not transmitted. A zero LLR is
    # neutral, so the decoder can include these variables without bias.
    return np.concatenate([np.zeros(2 * Zc), LLRin])


def _check_to_var_indices(H):
    """Return variable-node indices connected to each check node."""
    return [list(np.where(H[row, :] == 1)[0]) for row in range(H.shape[0])]


def _var_to_check_indices(H):
    """Return check-node indices connected to each variable node."""
    return [list(np.where(H[:, col] == 1)[0]) for col in range(H.shape[1])]


def _hard_bits_from_llr(LLR, size):
    """Convert the first ``size`` LLRs to hard bits."""
    bits = np.zeros(size, dtype="i1")
    bits[LLR[0:size] < 0] = 1
    return bits


def _crc_check_from_llr(LLR, B, crcpoly):
    """Hard-decision the payload region and run CRC."""
    # CRC is used as the early-stop criterion for all soft-decision decoders.
    decoded_bits = _hard_bits_from_llr(LLR, B)
    blk, status = crc.nr_crc_decode(decoded_bits, crcpoly)
    return decoded_bits, blk, status == 0


def nr_decode_ldpc(LLRin, Zc, bgn, crcpoly, B, ldpcdecoder_config=None):
    """Decode one transmitted LDPC code block.

    Args:
        LLRin: Length-``N`` received LLR sequence. It excludes the first
            ``2*Zc`` punctured systematic positions.
        Zc: LDPC lifting size.
        bgn: LDPC base graph number, either 1 or 2.
        crcpoly: CRC polynomial name accepted by ``crc.nr_crc_decode``.
        B: Number of decoded bits to check with CRC.
        ldpcdecoder_config: Optional dict with ``L``, ``algo``, ``alpha``, and
            ``beta`` entries.

    Returns:
        decoded_bits: CRC-stripped payload when CRC passes; otherwise hard
            decisions from the last iteration.
        status: ``True`` when decoding succeeds.
    """
    config = _decoder_config(ldpcdecoder_config)
    L = config["L"]
    algo = config["algo"]
    alpha = config["alpha"]
    beta = config["beta"]

    assert algo in SUPPORTED_ALGOS

    N, _ = _ldpc_dimensions(Zc, bgn)
    assert N == LLRin.size

    H = _load_parity_check_matrix(Zc, bgn)
    full_llr = _prepend_punctured_llr(LLRin, Zc)

    # Dispatch to the requested schedule/algorithm while sharing the same full
    # LLR vector and parity-check matrix.
    if algo == "BF":
        return ldpc_decoder_bit_flipping.ldpc_decoder_BF(full_llr, H, crcpoly, B, L)

    if algo in ["BP", "min-sum"]:
        return decode_ldpc(full_llr, H, L, crcpoly, B, algo, alpha, beta)

    return layered_decode_ldpc(full_llr, H, crcpoly, B, L, algo, alpha, beta)


def decode_ldpc(LLRin, H, L, crcpoly, B, algo="min-sum", alpha=1, beta=0):
    """Decode with flooding-schedule BP or min-sum.

    Args:
        LLRin: Full decoder input LLR sequence, including neutral LLRs for the
            two punctured systematic columns.
        H: Expanded LDPC parity-check matrix.
        L: Maximum iteration count.
        crcpoly: CRC polynomial name.
        B: Number of decoded bits to check with CRC.
        algo: ``"BP"`` or ``"min-sum"``.
        alpha: Normalized min-sum scaling factor.
        beta: Offset min-sum subtraction value.

    Returns:
        decoded_bits: CRC-stripped payload when CRC passes; otherwise hard
            decisions from the final iteration.
        status: ``True`` when CRC passes.
    """
    num_checks, num_vars = H.shape
    assert LLRin.size == num_vars
    assert algo in ["BP", "min-sum"]

    check_to_vars = _check_to_var_indices(H)
    var_to_checks = _var_to_check_indices(H)

    # LQ holds posterior variable-node LLRs. Lq is variable-to-check messages,
    # and Lr is check-to-variable messages.
    LQ = LLRin.copy()
    Lq = H * LLRin
    Lr = np.zeros((num_checks, num_vars))
    decoded_bits = _hard_bits_from_llr(LQ, B)

    for _ in range(L):
        # Check-node update: compute each outgoing message from all incoming
        # variable-node messages connected to the same check node.
        for check_idx, var_indices in enumerate(check_to_vars):
            sel_Lq = Lq[check_idx, var_indices]
            if algo == "BP":
                Lr[check_idx, var_indices] = _BP_process(sel_Lq, var_indices)
            else:
                Lr[check_idx, var_indices] = _min_sum_process(
                    sel_Lq, var_indices, alpha, beta
                )

        # Variable-node posterior LLRs are the channel LLR plus all incoming
        # check-node messages.
        LQ = LLRin + Lr.sum(axis=0)

        # Refresh variable-to-check messages by removing the destination
        # check-node contribution from the posterior LLR.
        for var_idx, check_indices in enumerate(var_to_checks):
            for check_idx in check_indices:
                Lq[check_idx, var_idx] = LQ[var_idx] - Lr[check_idx, var_idx]

        decoded_bits, blk, crc_ok = _crc_check_from_llr(LQ, B, crcpoly)
        if crc_ok:
            return blk, True

    # If CRC never passes, return the last hard decision for the requested
    # decoded region.
    return decoded_bits, False


def layered_decode_ldpc(LLRin, H, crcpoly, B, L, algo="min-sum", alpha=1, beta=0):
    """Decode with layered BP or layered min-sum.

    Layered decoding updates the posterior LLRs immediately after each check
    node, which often converges faster than the flooding schedule.
    """
    num_checks, num_vars = H.shape
    assert LLRin.size == num_vars
    assert algo in ["layered-BP", "layered-min-sum"]

    check_to_vars = _check_to_var_indices(H)

    # Layered decoding keeps only LQ and Lr. Each layer reconstructs its
    # variable-to-check messages from the current posterior LLRs.
    LQ = LLRin.copy()
    Lr = np.zeros((num_checks, num_vars))
    decoded_bits = _hard_bits_from_llr(LQ, B)

    for _ in range(L):
        for check_idx, var_indices in enumerate(check_to_vars):
            # Remove the old check-node message to recover the extrinsic input
            # for this layer, then write the new message back immediately.
            sel_Lq = LQ[var_indices] - Lr[check_idx, var_indices]

            if algo == "layered-min-sum":
                Lr[check_idx, var_indices] = _min_sum_process(
                    sel_Lq, var_indices, alpha, beta
                )
            else:
                Lr[check_idx, var_indices] = _BP_process(sel_Lq, var_indices)

            LQ[var_indices] = sel_Lq + Lr[check_idx, var_indices]

        decoded_bits, blk, crc_ok = _crc_check_from_llr(LQ, B, crcpoly)
        if crc_ok:
            return blk, True

    # If CRC never passes, return the last hard decision for the requested
    # decoded region.
    return decoded_bits, False


def _BP_process(sel_Lq, V_m):
    """Calculate check-node outgoing messages with BP/sum-product."""
    sel_Lr = np.zeros(sel_Lq.shape, dtype=np.float32)
    zero_indices = list(np.where(sel_Lq == 0)[0])

    if len(zero_indices) == 0:
        # Standard BP check-node rule: product of tanh(incoming/2), excluding
        # the destination edge. Clamp atanh inputs at +/-1 to avoid infinities.
        tanh_Lq = np.tanh(sel_Lq / 2)
        tanh_prod = np.prod(tanh_Lq)

        for idx, _ in enumerate(V_m):
            extrinsic_prod = tanh_prod / tanh_Lq[idx]
            if extrinsic_prod >= 1:
                sel_Lr[idx] = 2 * 19.07
            elif extrinsic_prod <= -1:
                sel_Lr[idx] = -2 * 19.07
            else:
                sel_Lr[idx] = 2 * np.arctanh(extrinsic_prod)
    elif len(zero_indices) == 1:
        # With one zero input, every non-zero edge sees an extrinsic product
        # containing that zero. Only the zero edge can receive a non-zero value.
        tanh_Lq = np.tanh(sel_Lq / 2)
        zero_idx = zero_indices[0]
        # Preserve the existing zero-input behavior: only the zero edge can
        # receive a non-zero message.
        sel_Lr[zero_idx] = np.prod(tanh_Lq[0:zero_idx]) * np.prod(
            tanh_Lq[zero_idx + 1 :]
        )

    return sel_Lr


def _min_sum_process(sel_Lq, V_m, alpha, beta):
    """Calculate check-node outgoing messages with mixed min-sum.

    ``alpha=1, beta=0`` gives plain min-sum. Reducing ``alpha`` gives normalized
    min-sum, and increasing ``beta`` gives offset min-sum.
    """
    sel_Lr = np.zeros(sel_Lq.shape, dtype=np.float32)
    zero_indices = list(np.where(sel_Lq == 0)[0])

    if len(zero_indices) == 0:
        # Standard min-sum needs the global sign and the smallest two absolute
        # values. The edge holding the minimum uses the second minimum.
        sign_prod = np.prod(np.sign(sel_Lq))
        sorted_abs = np.sort(np.abs(sel_Lq))
        first_min = sorted_abs[0]
        second_min = sorted_abs[1]

        for idx, _ in enumerate(V_m):
            min_value = second_min if np.abs(sel_Lq[idx]) == first_min else first_min
            min_value = max(min_value - beta, 0)
            sel_Lr[idx] = alpha * sign_prod * np.sign(sel_Lq[idx]) * min_value
    elif len(zero_indices) == 1:
        # With one zero input, only that edge can receive a non-zero message.
        # Its sign and magnitude come from all non-zero neighboring edges.
        zero_idx = zero_indices[0]

        if zero_idx == 0:
            sign_prod = np.prod(np.sign(sel_Lq[1:]))
            min_value = np.min(np.abs(sel_Lq[1:]))
        elif zero_idx == sel_Lq.size - 1:
            sign_prod = np.prod(np.sign(sel_Lq[0:zero_idx]))
            min_value = np.min(np.abs(sel_Lq[0:zero_idx]))
        else:
            sign_prod = np.prod(np.sign(sel_Lq[0:zero_idx])) * np.prod(
                np.sign(sel_Lq[zero_idx + 1 :])
            )
            min_value = min(
                np.min(np.abs(sel_Lq[0:zero_idx])),
                np.min(np.abs(sel_Lq[zero_idx + 1 :])),
            )

        min_value = max(min_value - beta, 0)
        sel_Lr[zero_idx] = alpha * sign_prod * min_value

    return sel_Lr

if __name__ == "__main__":
    print("LDPC decoder helper examples")

    sel_Lq = np.array([1.5, -0.7, 2.1, -3.0], dtype=np.float32)
    print("  BP messages      :", np.round(_BP_process(sel_Lq, range(sel_Lq.size)), 3))
    print(
        "  min-sum messages :",
        np.round(_min_sum_process(sel_Lq, range(sel_Lq.size), 0.8, 0), 3),
    )

    try:
        _load_parity_check_matrix(2, 1)
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        print("  full decoder     : skipped, scipy is required to load LDPC tables")
    else:
        print("  full decoder     : LDPC table loading is available")

    # Minimal decoder smoke test with a tiny parity-check matrix. The all-zero
    # payload plus CRC6 is a valid CRC word, so this exercises the decoder loops
    # without requiring a full generated 5G LDPC matrix.
    H = np.array(
        [
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        ],
        dtype="i1",
    )
    LLRin = 4.0 * np.ones(10, dtype=np.float32)

    for algo in ["min-sum", "layered-min-sum"]:
        if algo == "min-sum":
            decoded_bits, status = decode_ldpc(
                LLRin, H, L=3, crcpoly="6", B=10, algo=algo, alpha=1, beta=0
            )
        else:
            decoded_bits, status = layered_decode_ldpc(
                LLRin, H, crcpoly="6", B=10, L=3, algo=algo, alpha=1, beta=0
            )

        print(f"  {algo:15s}: status={status}, bits={decoded_bits.tolist()}")
