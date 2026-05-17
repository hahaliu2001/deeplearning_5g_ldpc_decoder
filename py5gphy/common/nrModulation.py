# -*- coding:utf-8 -*-
"""5G NR modulation mapper helpers from TS 38.211 section 5.1."""

import numpy as np


MODULATION_ORDER = {
    "pi/2-bpsk": 1,
    "bpsk": 1,
    "qpsk": 2,
    "16qam": 4,
    "64qam": 6,
    "256qam": 8,
    "1024qam": 10,
}

QAM_NORMALIZATION = {
    4: np.sqrt(10.0),
    6: np.sqrt(42.0),
    8: np.sqrt(170.0),
    # 1024QAM has 32 amplitude levels per axis: +/-1, +/-3, ..., +/-31.
    10: np.sqrt(682.0),
}


def _normalize_modtype(modtype):
    """Return a canonical lower-case modulation name."""
    modtype = str(modtype).lower()
    assert modtype in MODULATION_ORDER, "modulation type is incorrect"
    return modtype


def _as_float_bits(inbits):
    """Return input bits as a flat float array."""
    bits = np.asarray(inbits).reshape(-1)
    assert np.all((bits == 0) | (bits == 1)), "input bits must be 0 or 1"
    return bits.astype("f")


def _check_length(bits, Qm):
    """Ensure the input bit length contains complete modulation symbols."""
    assert bits.size % Qm == 0, f"length of databits must be multiple of {Qm}"


def _bit_sign(bits):
    """Map bit 0 to +1 and bit 1 to -1."""
    return 1.0 - 2.0 * bits


def _qam_axis(bits, offset, Qm):
    """Return I or Q amplitudes for the selected QAM axis."""
    axis_signs = [_bit_sign(bits[offset + 2 * idx :: Qm]) for idx in range(Qm // 2)]

    if Qm == 4:
        return axis_signs[0] * (2.0 - axis_signs[1])
    if Qm == 6:
        return axis_signs[0] * (4.0 - axis_signs[1] * (2.0 - axis_signs[2]))
    if Qm == 8:
        return axis_signs[0] * (
            8.0 - axis_signs[1] * (4.0 - axis_signs[2] * (2.0 - axis_signs[3]))
        )

    if Qm == 10:
        # 1024QAM extends the same Gray-coded recursive amplitude rule used by
        # the lower-order QAM mappings.
        return axis_signs[0] * (
            16.0 - axis_signs[1] * (
                8.0 - axis_signs[2] * (
                    4.0 - axis_signs[3] * (2.0 - axis_signs[4])
                )
            )
        )

    raise ValueError(f"unsupported QAM order: {Qm}")


def nrModulate(inbits, modtype):
    """Map input bits to complex 5G NR modulation symbols.

    Args:
        inbits: One-dimensional array of 0/1 bits.
        modtype: One of "pi/2-bpsk", "bpsk", "qpsk", "16qam", "64qam",
            "256qam", or "1024qam".

    Returns:
        Complex64 NumPy array containing modulation symbols.
    """
    modtype = _normalize_modtype(modtype)
    bits = _as_float_bits(inbits)
    Qm = MODULATION_ORDER[modtype]

    if modtype == "bpsk":
        sign = _bit_sign(bits)
        return ((sign + 1j * sign) / np.sqrt(2.0)).astype("c8")

    if modtype == "pi/2-bpsk":
        sign = _bit_sign(bits)
        mod_data = ((sign + 1j * sign) / np.sqrt(2.0)).astype("c8")
        mod_data[1::2] = (((-sign[1::2]) + 1j * sign[1::2]) / np.sqrt(2.0)).astype("c8")
        return mod_data

    _check_length(bits, Qm)

    if modtype == "qpsk":
        i_axis = _bit_sign(bits[0::2])
        q_axis = _bit_sign(bits[1::2])
        return ((i_axis + 1j * q_axis) / np.sqrt(2.0)).astype("c8")

    if modtype == "1024qam":
        i_axis = _qam_axis(bits, 0, Qm)
        q_axis = _qam_axis(bits, 1, Qm)
        return ((i_axis + 1j * q_axis) / QAM_NORMALIZATION[Qm]).astype("c8")

    i_axis = _qam_axis(bits, 0, Qm)
    q_axis = _qam_axis(bits, 1, Qm)
    return ((i_axis + 1j * q_axis) / QAM_NORMALIZATION[Qm]).astype("c8")


def get_mod_list(modtype):
    """Return every constellation symbol and its associated bit pattern."""
    modtype = _normalize_modtype(modtype)
    Qm = MODULATION_ORDER[modtype]
    num_symbols = 2 ** Qm

    mod_array = np.zeros(num_symbols, dtype="c8")
    inbits_array = np.zeros((num_symbols, Qm), dtype="i8")

    for symbol_idx in range(num_symbols):
        inbits = np.array(
            [int(bit) for bit in f"{symbol_idx:0{Qm}b}"],
            dtype="i8",
        )
        inbits_array[symbol_idx, :] = inbits
        mod_array[symbol_idx] = nrModulate(inbits, modtype)[0]

    return mod_array, inbits_array


def get_opposite_syms(inbits, modtype):
    """Return nearest symbols with each bit flipped in turn.

    The returned array has one symbol per input bit. For QAM, lower-significance
    bits are moved to the nearest point with the opposite bit value, matching the
    original helper used by the demodulator derivations.
    """
    modtype = _normalize_modtype(modtype)
    inbits = np.asarray(inbits, dtype="i8").reshape(-1)

    if modtype in ["pi/2-bpsk", "bpsk"]:
        return nrModulate(1 - inbits, modtype)

    if modtype == "qpsk":
        replacement_bits = np.array([
            1 - inbits[0], inbits[1],
            inbits[0], 1 - inbits[1],
        ])
    elif modtype == "16qam":
        replacement_bits = np.array([
            1 - inbits[0], inbits[1], 0, inbits[3],
            inbits[0], 1 - inbits[1], inbits[2], 0,
            inbits[0], inbits[1], 1 - inbits[2], inbits[3],
            inbits[0], inbits[1], inbits[2], 1 - inbits[3],
        ])
    elif modtype == "64qam":
        replacement_bits = np.array([
            1 - inbits[0], inbits[1], 0, inbits[3], 1, inbits[5],
            inbits[0], 1 - inbits[1], inbits[2], 0, inbits[4], 1,
            inbits[0], inbits[1], 1 - inbits[2], inbits[3], 0, inbits[5],
            inbits[0], inbits[1], inbits[2], 1 - inbits[3], inbits[4], 0,
            inbits[0], inbits[1], inbits[2], inbits[3], 1 - inbits[4], inbits[5],
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], 1 - inbits[5],
        ])
    elif modtype == "256qam":
        replacement_bits = np.array([
            1 - inbits[0], inbits[1], 0, inbits[3], 1, inbits[5], 1, inbits[7],
            inbits[0], 1 - inbits[1], inbits[2], 0, inbits[4], 1, inbits[6], 1,
            inbits[0], inbits[1], 1 - inbits[2], inbits[3], 0, inbits[5], 1, inbits[7],
            inbits[0], inbits[1], inbits[2], 1 - inbits[3], inbits[4], 0, inbits[6], 1,
            inbits[0], inbits[1], inbits[2], inbits[3], 1 - inbits[4], inbits[5], 0, inbits[7],
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], 1 - inbits[5], inbits[6], 0,
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], inbits[5], 1 - inbits[6], inbits[7],
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], inbits[5], inbits[6], 1 - inbits[7],
        ])
    elif modtype == "1024qam":
        replacement_bits = np.array([
            1 - inbits[0], inbits[1], 0, inbits[3], 1, inbits[5], 1, inbits[7], 1, inbits[9],
            inbits[0], 1 - inbits[1], inbits[2], 0, inbits[4], 1, inbits[6], 1, inbits[8], 1,
            inbits[0], inbits[1], 1 - inbits[2], inbits[3], 0, inbits[5], 1, inbits[7], 1, inbits[9],
            inbits[0], inbits[1], inbits[2], 1 - inbits[3], inbits[4], 0, inbits[6], 1, inbits[8], 1,
            inbits[0], inbits[1], inbits[2], inbits[3], 1 - inbits[4], inbits[5], 0, inbits[7], 1, inbits[9],
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], 1 - inbits[5], inbits[6], 0, inbits[8], 1,
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], inbits[5], 1 - inbits[6], inbits[7], 0, inbits[9],
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], inbits[5], inbits[6], 1 - inbits[7], inbits[8], 0,
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], inbits[5], inbits[6], inbits[7], 1 - inbits[8], inbits[9],
            inbits[0], inbits[1], inbits[2], inbits[3], inbits[4], inbits[5], inbits[6], inbits[7], inbits[8], 1 - inbits[9],
        ])
    else:
        raise ValueError(f"unsupported modulation type: {modtype}")

    return nrModulate(replacement_bits, modtype)


def get_oppisite_syms(inbits, modtype):
    """Backward-compatible wrapper for the misspelled original function name."""
    return get_opposite_syms(inbits, modtype)


if __name__ == "__main__":
    examples = {
        "qpsk": np.array([0, 1, 1, 0], dtype=np.uint8),
        "1024qam": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
    }

    for modtype, bits in examples.items():
        symbols = nrModulate(bits, modtype)
        print(f"{modtype} example: {np.round(symbols, 4).tolist()}")
