#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/*
 * Simple complex float used by the local modulation/channel helpers.
 *
 * This file mirrors py5gphy/ldpc_utils.py:tx_rx_process() for test use.  It
 * unpacks input bytes into bits, modulates them, adds complex AWGN, demodulates
 * to soft LLRs, and quantizes the LLRs to int16_t values for the C decoder.
 * The implementation follows the same 5G NR constellation mapping and
 * piecewise max-log LLR formulas used by py5gphy.
 */
typedef struct {
    float re;
    float im;
} complex_float_t;

static float bit_sign(uint8_t bit) {
    return bit ? -1.0f : 1.0f;
}

// Scale float LLRs by 128 and clip to a practical channel-LLR range. Very high
// SNR can otherwise saturate most samples to INT16_MAX/INT16_MIN, which makes
// the fixed-point LDPC decoder overconfident around punctured bits.
static int16_t quantize_llr_to_int16(float llr) {
    const int32_t max_channel_llr = 10000;
    float scaled = llr * 128.0f;
    int32_t quantized = (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);

    if (quantized > max_channel_llr) {
        quantized = max_channel_llr;
    } else if (quantized < -max_channel_llr) {
        quantized = -max_channel_llr;
    }

    return (int16_t)quantized;
}

// Convert a quantized decoder LLR back to the approximate float LLR scale.
float dequantize_int16_to_llr(int16_t quantized_llr) {
    return (float)quantized_llr / 128.0f;
}

// Extract one packed bit using the same MSB-first order as c_ldpc_5g_encode().
static uint8_t get_packed_bit(const uint8_t *inbytes, size_t bit_idx) {
    size_t byte_idx = bit_idx / 8;
    uint8_t bit_pos = (uint8_t)(7 - (bit_idx % 8));

    return (uint8_t)((inbytes[byte_idx] >> bit_pos) & 0x01);
}

/*
 * Generate one standard normal random variable with the Box-Muller transform.
 *
 * Call srand() before tx_rx_process() if a repeatable noise sequence is needed.
 */
static float randn(void) {
    const float two_pi = 6.28318530717958647692f;
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);

    return sqrtf(-2.0f * logf(u1)) * cosf(two_pi * u2);
}

// Map one group of Qm bits to the corresponding normalized NR symbol.
static complex_float_t modulate_symbol(const uint8_t *bits, uint16_t Qm) {
    complex_float_t sym = {0.0f, 0.0f};

    if (Qm == 1) {
        float s0 = bit_sign(bits[0]);
        float A = 1.0f / sqrtf(2.0f);
        sym.re = s0 * A;
        sym.im = s0 * A;
    } else if (Qm == 2) {
        float A = 1.0f / sqrtf(2.0f);
        sym.re = bit_sign(bits[0]) * A;
        sym.im = bit_sign(bits[1]) * A;
    } else if (Qm == 4) {
        float A = 1.0f / sqrtf(10.0f);
        sym.re = bit_sign(bits[0]) * (2.0f - bit_sign(bits[2])) * A;
        sym.im = bit_sign(bits[1]) * (2.0f - bit_sign(bits[3])) * A;
    } else if (Qm == 6) {
        float A = 1.0f / sqrtf(42.0f);
        sym.re = bit_sign(bits[0])
            * (4.0f - bit_sign(bits[2]) * (2.0f - bit_sign(bits[4]))) * A;
        sym.im = bit_sign(bits[1])
            * (4.0f - bit_sign(bits[3]) * (2.0f - bit_sign(bits[5]))) * A;
    } else if (Qm == 8) {
        float A = 1.0f / sqrtf(170.0f);
        sym.re = bit_sign(bits[0]) * (8.0f - bit_sign(bits[2])
            * (4.0f - bit_sign(bits[4]) * (2.0f - bit_sign(bits[6])))) * A;
        sym.im = bit_sign(bits[1]) * (8.0f - bit_sign(bits[3])
            * (4.0f - bit_sign(bits[5]) * (2.0f - bit_sign(bits[7])))) * A;
    }

    return sym;
}

static float llr_16qam_bit_0_1(float r, float noise_var, float A) {
    if (r < -2.0f * A) {
        return 8.0f * A * (r + A) / noise_var;
    }
    if (r < 2.0f * A) {
        return 4.0f * A * r / noise_var;
    }
    return 8.0f * A * (r - A) / noise_var;
}

static float llr_16qam_bit_2_3(float r, float noise_var, float A) {
    if (r < 0.0f) {
        return 4.0f * A * (r + 2.0f * A) / noise_var;
    }
    return 4.0f * A * (-r + 2.0f * A) / noise_var;
}

static float llr_64qam_bit_0_1(float r, float noise_var, float A) {
    if (r < -6.0f * A) {
        return 16.0f * A * (r + 3.0f * A) / noise_var;
    }
    if (r < -4.0f * A) {
        return 12.0f * A * (r + 2.0f * A) / noise_var;
    }
    if (r < -2.0f * A) {
        return 8.0f * A * (r + A) / noise_var;
    }
    if (r < 2.0f * A) {
        return 4.0f * A * r / noise_var;
    }
    if (r < 4.0f * A) {
        return 8.0f * A * (r - A) / noise_var;
    }
    if (r < 6.0f * A) {
        return 12.0f * A * (r - 2.0f * A) / noise_var;
    }
    return 16.0f * A * (r - 3.0f * A) / noise_var;
}

static float llr_64qam_bit_2_3(float r, float noise_var, float A) {
    if (r < -6.0f * A) {
        return 8.0f * A * (r + 5.0f * A) / noise_var;
    }
    if (r < -2.0f * A) {
        return 4.0f * A * (r + 4.0f * A) / noise_var;
    }
    if (r < 0.0f) {
        return 8.0f * A * (r + 3.0f * A) / noise_var;
    }
    if (r < 2.0f * A) {
        return 8.0f * A * (-r + 3.0f * A) / noise_var;
    }
    if (r < 6.0f * A) {
        return 4.0f * A * (-r + 4.0f * A) / noise_var;
    }
    return 8.0f * A * (-r + 5.0f * A) / noise_var;
}

static float llr_64qam_bit_4_5(float r, float noise_var, float A) {
    if (r < -4.0f * A) {
        return 4.0f * A * (r + 6.0f * A) / noise_var;
    }
    if (r < 0.0f) {
        return 4.0f * A * (-r - 2.0f * A) / noise_var;
    }
    if (r < 4.0f * A) {
        return 4.0f * A * (r - 2.0f * A) / noise_var;
    }
    return 4.0f * A * (-r + 6.0f * A) / noise_var;
}

static float llr_256qam_bit_0_1(float r, float noise_var, float A) {
    if (r < -14.0f * A) return 32.0f * A * (r + 7.0f * A) / noise_var;
    if (r < -12.0f * A) return 28.0f * A * (r + 6.0f * A) / noise_var;
    if (r < -10.0f * A) return 24.0f * A * (r + 5.0f * A) / noise_var;
    if (r < -8.0f * A) return 20.0f * A * (r + 4.0f * A) / noise_var;
    if (r < -6.0f * A) return 16.0f * A * (r + 3.0f * A) / noise_var;
    if (r < -4.0f * A) return 12.0f * A * (r + 2.0f * A) / noise_var;
    if (r < -2.0f * A) return 8.0f * A * (r + A) / noise_var;
    if (r < 2.0f * A) return 4.0f * A * r / noise_var;
    if (r < 4.0f * A) return 8.0f * A * (r - A) / noise_var;
    if (r < 6.0f * A) return 12.0f * A * (r - 2.0f * A) / noise_var;
    if (r < 8.0f * A) return 16.0f * A * (r - 3.0f * A) / noise_var;
    if (r < 10.0f * A) return 20.0f * A * (r - 4.0f * A) / noise_var;
    if (r < 12.0f * A) return 24.0f * A * (r - 5.0f * A) / noise_var;
    if (r < 14.0f * A) return 28.0f * A * (r - 6.0f * A) / noise_var;
    return 32.0f * A * (r - 7.0f * A) / noise_var;
}

static float llr_256qam_bit_2_3(float r, float noise_var, float A) {
    if (r < -14.0f * A) return 16.0f * A * (r + 11.0f * A) / noise_var;
    if (r < -12.0f * A) return 12.0f * A * (r + 10.0f * A) / noise_var;
    if (r < -10.0f * A) return 8.0f * A * (r + 9.0f * A) / noise_var;
    if (r < -6.0f * A) return 4.0f * A * (r + 8.0f * A) / noise_var;
    if (r < -4.0f * A) return 8.0f * A * (r + 7.0f * A) / noise_var;
    if (r < -2.0f * A) return 12.0f * A * (r + 6.0f * A) / noise_var;
    if (r < 0.0f) return 16.0f * A * (r + 5.0f * A) / noise_var;
    if (r < 2.0f * A) return 16.0f * A * (-r + 5.0f * A) / noise_var;
    if (r < 4.0f * A) return 12.0f * A * (-r + 6.0f * A) / noise_var;
    if (r < 6.0f * A) return 8.0f * A * (-r + 7.0f * A) / noise_var;
    if (r < 10.0f * A) return 4.0f * A * (-r + 8.0f * A) / noise_var;
    if (r < 12.0f * A) return 8.0f * A * (-r + 9.0f * A) / noise_var;
    if (r < 14.0f * A) return 12.0f * A * (-r + 10.0f * A) / noise_var;
    return 16.0f * A * (-r + 11.0f * A) / noise_var;
}

static float llr_256qam_bit_4_5(float r, float noise_var, float A) {
    if (r < -14.0f * A) return 8.0f * A * (r + 13.0f * A) / noise_var;
    if (r < -10.0f * A) return 4.0f * A * (r + 12.0f * A) / noise_var;
    if (r < -8.0f * A) return 8.0f * A * (r + 11.0f * A) / noise_var;
    if (r < -6.0f * A) return 8.0f * A * (-r - 5.0f * A) / noise_var;
    if (r < -2.0f * A) return 4.0f * A * (-r - 4.0f * A) / noise_var;
    if (r < 0.0f) return 8.0f * A * (-r - 3.0f * A) / noise_var;
    if (r < 2.0f * A) return 8.0f * A * (r - 3.0f * A) / noise_var;
    if (r < 6.0f * A) return 4.0f * A * (r - 4.0f * A) / noise_var;
    if (r < 8.0f * A) return 8.0f * A * (r - 5.0f * A) / noise_var;
    if (r < 10.0f * A) return 8.0f * A * (-r + 11.0f * A) / noise_var;
    if (r < 14.0f * A) return 4.0f * A * (-r + 12.0f * A) / noise_var;
    return 8.0f * A * (-r + 13.0f * A) / noise_var;
}

static float llr_256qam_bit_6_7(float r, float noise_var, float A) {
    if (r < -12.0f * A) return 4.0f * A * (r + 14.0f * A) / noise_var;
    if (r < -8.0f * A) return 4.0f * A * (-r - 10.0f * A) / noise_var;
    if (r < -4.0f * A) return 4.0f * A * (r + 6.0f * A) / noise_var;
    if (r < 0.0f) return 4.0f * A * (-r - 2.0f * A) / noise_var;
    if (r < 4.0f * A) return 4.0f * A * (r - 2.0f * A) / noise_var;
    if (r < 8.0f * A) return 4.0f * A * (-r + 6.0f * A) / noise_var;
    if (r < 12.0f * A) return 4.0f * A * (r - 10.0f * A) / noise_var;
    return 4.0f * A * (-r + 14.0f * A) / noise_var;
}

// Demodulate one noisy symbol and write Qm LLRs in the original bit order.
static void demodulate_symbol(complex_float_t sym, uint16_t Qm,
    float noise_var, float *llr_out) {
    if (Qm == 1) {
        float A = 1.0f / sqrtf(2.0f);
        llr_out[0] = 4.0f * (sym.re + sym.im) * A / noise_var;
    } else if (Qm == 2) {
        float A = 1.0f / sqrtf(2.0f);
        llr_out[0] = 4.0f * sym.re * A / noise_var;
        llr_out[1] = 4.0f * sym.im * A / noise_var;
    } else if (Qm == 4) {
        float A = 1.0f / sqrtf(10.0f);
        llr_out[0] = llr_16qam_bit_0_1(sym.re, noise_var, A);
        llr_out[1] = llr_16qam_bit_0_1(sym.im, noise_var, A);
        llr_out[2] = llr_16qam_bit_2_3(sym.re, noise_var, A);
        llr_out[3] = llr_16qam_bit_2_3(sym.im, noise_var, A);
    } else if (Qm == 6) {
        float A = 1.0f / sqrtf(42.0f);
        llr_out[0] = llr_64qam_bit_0_1(sym.re, noise_var, A);
        llr_out[1] = llr_64qam_bit_0_1(sym.im, noise_var, A);
        llr_out[2] = llr_64qam_bit_2_3(sym.re, noise_var, A);
        llr_out[3] = llr_64qam_bit_2_3(sym.im, noise_var, A);
        llr_out[4] = llr_64qam_bit_4_5(sym.re, noise_var, A);
        llr_out[5] = llr_64qam_bit_4_5(sym.im, noise_var, A);
    } else if (Qm == 8) {
        float A = 1.0f / sqrtf(170.0f);
        llr_out[0] = llr_256qam_bit_0_1(sym.re, noise_var, A);
        llr_out[1] = llr_256qam_bit_0_1(sym.im, noise_var, A);
        llr_out[2] = llr_256qam_bit_2_3(sym.re, noise_var, A);
        llr_out[3] = llr_256qam_bit_2_3(sym.im, noise_var, A);
        llr_out[4] = llr_256qam_bit_4_5(sym.re, noise_var, A);
        llr_out[5] = llr_256qam_bit_4_5(sym.im, noise_var, A);
        llr_out[6] = llr_256qam_bit_6_7(sym.re, noise_var, A);
        llr_out[7] = llr_256qam_bit_6_7(sym.im, noise_var, A);
    }
}

/*
 * Unpack bytes, modulate bits, add AWGN, demodulate, and quantize LLRs.
 *
 * Inputs:
 *   inbytes: Packed input bits, MSB-first in each byte.  This matches the
 *       packed byte output from c_ldpc_5g_encode().
 *   num_bits: Number of valid bits in inbytes. Must be a multiple of Qm.
 *   Qm: Modulation order, one of 1, 2, 4, 6, or 8.
 *   noise_power_db: AWGN power in dB, matching Python's noise_power_db.
 *
 * Output:
 *   llr_out: Quantized demodulated LLRs, one int16_t value per input bit.
 *
 * Returns:
 *   0 on success, -1 for invalid arguments.
 */
int tx_rx_process(const uint8_t *inbytes, size_t num_bits, uint16_t Qm,
    float noise_power_db, int16_t *llr_out) {
    if (inbytes == NULL || llr_out == NULL || num_bits == 0) {
        return -1;
    }
    if (!(Qm == 1 || Qm == 2 || Qm == 4 || Qm == 6 || Qm == 8)) {
        return -1;
    }
    if ((num_bits % Qm) != 0) {
        return -1;
    }

    float noise_var = powf(10.0f, noise_power_db / 10.0f);
    float noise_sigma = sqrtf(noise_var / 2.0f);
    size_t num_symbols = num_bits / Qm;

    for (size_t i = 0; i < num_symbols; i++) {
        uint8_t symbol_bits[8];
        float symbol_llr[8];

        for (uint16_t j = 0; j < Qm; j++) {
            symbol_bits[j] = get_packed_bit(inbytes, i * Qm + j);
        }

        complex_float_t sym = modulate_symbol(symbol_bits, Qm);

        sym.re += noise_sigma * randn();
        sym.im += noise_sigma * randn();

        demodulate_symbol(sym, Qm, noise_var, symbol_llr);
        for (uint16_t j = 0; j < Qm; j++) {
            //for fixed point, LLR quantization is important for LDPC decoder performance,for both min-sum and BP algorithm
            //BP algorithm need remove this quantization scaling, or the performance is poor
            llr_out[i * Qm + j] = quantize_llr_to_int16(symbol_llr[j]);
        }
    }

    return 0;
}
