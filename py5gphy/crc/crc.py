# -*- coding:utf-8 -*-
"""CRC helpers used by the 5G PHY Python reference code."""

import numpy as np


# Polynomials are stored without the leading x^L term. The arrays contain the
# feedback taps used after the shift register drops its current MSB.
CRC_POLYNOMIALS = {
    "6": np.array([1, 0, 0, 0, 0, 1], dtype="i1"),
    "11": np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype="i1"),
    "16": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype="i1"),
    "24A": np.array(
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
         1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        dtype="i1",
    ),
    "24B": np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        dtype="i1",
    ),
    "24C": np.array(
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
         0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        dtype="i1",
    ),
}


def _as_bit_array(bits, name):
    """Return a one-dimensional int8 array and validate that it contains bits."""
    bit_array = np.asarray(bits, dtype="i1").reshape(-1)
    if np.any((bit_array != 0) & (bit_array != 1)):
        raise AssertionError(f"{name} must contain only 0/1 values")
    return bit_array


def _get_crcpoly(poly):
    """Return the CRC feedback polynomial bits for a 3GPP CRC name."""
    key = str(poly).upper()
    if key not in CRC_POLYNOMIALS:
        raise AssertionError(f"unsupported CRC polynomial: {poly}")
    return CRC_POLYNOMIALS[key]


def _mask_bits(mask, crc_len):
    """Return the L least-significant mask bits in MSB-first order."""
    # RNTI/CRC masks are applied MSB-first to the CRC field.  If the integer is
    # wider than crc_len, the low crc_len bits are used.
    return np.array(
        [(mask >> (crc_len - 1 - i)) & 1 for i in range(crc_len)],
        dtype="i1",
    )


def _calculate_remainder(bits_with_tail, crcpoly, data_len):
    """Run the shift-register CRC division over data_len input bits."""
    crc_len = crcpoly.size

    # Seed the register with the first L bits.  During encoding, the tail is
    # initialized to zeros; during checking, it contains the received CRC bits.
    remainder = bits_with_tail[:crc_len].copy()

    for idx in range(data_len):
        first_bit = remainder[0]

        # Shift in the next data/tail bit before applying the feedback taps.
        remainder[:-1] = remainder[1:]
        remainder[-1] = bits_with_tail[idx + crc_len]
        if first_bit:
            remainder = np.bitwise_xor(remainder, crcpoly)

    return remainder.astype("i1", copy=False)


def nr_crc_encode(blk, poly, mask=0):
    """Append CRC bits to a 0/1 bit array.

    Args:
        blk: Input payload bits, one bit per element.
        poly: CRC polynomial name: "6", "11", "16", "24A", "24B", or "24C".
        mask: Optional integer mask XORed with the CRC bits, MSB-first.

    Returns:
        A new int8 array containing the payload followed by CRC bits.
    """
    payload = _as_bit_array(blk, "blk")
    crcpoly = _get_crcpoly(poly)
    crc_len = crcpoly.size
    data_len = payload.size

    blkandcrc = np.zeros(data_len + crc_len, dtype="i1")
    blkandcrc[:data_len] = payload

    # The zero tail leaves room for the computed CRC parity bits.
    remainder = _calculate_remainder(blkandcrc, crcpoly, data_len)
    if mask:
        remainder = np.bitwise_xor(remainder, _mask_bits(mask, crc_len))

    blkandcrc[data_len:] = remainder
    return blkandcrc


def nr_crc_decode(blkandcrc, poly, mask=0):
    """Check CRC bits and return the payload plus error status.

    Args:
        blkandcrc: Input bits containing payload followed by CRC bits.
        poly: CRC polynomial name: "6", "11", "16", "24A", "24B", or "24C".
        mask: Optional integer mask that was XORed with the transmitted CRC.

    Returns:
        (blk, err), where err is 0 when CRC passes and 1 when it fails.
    """
    received = _as_bit_array(blkandcrc, "blkandcrc")
    crcpoly = _get_crcpoly(poly)
    crc_len = crcpoly.size
    data_len = received.size - crc_len
    if data_len < 0:
        raise AssertionError("blkandcrc is shorter than the CRC length")

    # Re-running the same division over payload+CRC should leave an all-zero
    # remainder after undoing any transmitted CRC mask.
    remainder = _calculate_remainder(received, crcpoly, data_len)
    if mask:
        remainder = np.bitwise_xor(remainder, _mask_bits(mask, crc_len))

    blk = received[:data_len]
    err = 1 if np.count_nonzero(remainder) else 0
    return blk, err


def gen_crc_mask(K, padCRC, rnti):
    """Generate the CRC24C mask used by polar coding paths.

    Args:
        K: Information bit length, excluding CRC bits.
        padCRC: When 1, prefix the CRC calculation with 24 ones.
        rnti: RNTI value used as the CRC mask.
    """
    if padCRC != 1:
        # BCH uses an all-zero mask; DCI uses the padded CRC24C path below.
        return np.zeros(24, dtype="i1")

    # For DCI, the mask is the CRC24C parity of 24 leading ones followed by K
    # zeros, with the RNTI applied to the CRC field.
    data = np.concatenate((np.ones(24, dtype="i1"), np.zeros(K, dtype="i1")))
    blkandcrc = nr_crc_encode(data, "24C", rnti)
    return blkandcrc[-24:]


def check_distributed_CRC24C(ck, crc_bit_loc, crcidx_matrix, crc_mask):
    """Verify one distributed CRC24C bit in a polar decoder candidate."""
    K = np.asarray(ck).size

    # crcidx_matrix stores the payload positions that contribute to each CRC bit.
    column = crcidx_matrix[:, crc_bit_loc]
    locs = column[column >= 0]

    calculated_crc_bit = ((sum(ck[locs]) % 2) + crc_mask[crc_bit_loc]) % 2
    decoded_crc_bit = ck[K - 24 + crc_bit_loc]

    return calculated_crc_bit == decoded_crc_bit


def gen_CRC24C_encoding_matrix(K):
    """Generate a dependency matrix for CRC24C parity-bit calculation.

    Each column corresponds to one CRC bit. Entries contain the input-bit index
    that contributes to that CRC bit, or -1 when that input bit is not involved.
    """
    crc_len = 24
    crcidx_matrix = np.zeros((K, crc_len), dtype="i2")

    for n in range(K):
        # Encode a one-hot payload to see which CRC bits depend on input bit n.
        inbits = np.zeros(K, dtype="i1")
        inbits[n] = 1
        crc_bits = nr_crc_encode(inbits, "24C")[-crc_len:]
        crcidx_matrix[n, :] = crc_bits * (n + 1)

    return crcidx_matrix - 1


if __name__ == "__main__":
    # Example: append CRC24A to a short payload and check it again.
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype="i1")
    blkandcrc = nr_crc_encode(payload, "24A")
    decoded_payload, err = nr_crc_decode(blkandcrc, "24A")

    print("Payload bits:      ", payload.tolist())
    print("Payload + CRC24A:  ", blkandcrc.tolist())
    print("Decoded payload:   ", decoded_payload.tolist())
    print("CRC status:        ", "pass" if err == 0 else "fail")

    # Example: one flipped bit should make the CRC check fail.
    corrupted = blkandcrc.copy()
    corrupted[0] ^= 1
    _, corrupted_err = nr_crc_decode(corrupted, "24A")
    print("After one bit flip:", "pass" if corrupted_err == 0 else "fail")
    
