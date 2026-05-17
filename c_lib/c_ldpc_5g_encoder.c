#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

// Maximum input code-block size is BG1 K = 22 * Zc.
static uint8_t ldpc_in_bit[MAX_ZC * 22];

// Encoder output excludes the first two punctured systematic groups.
static uint8_t ldpc_out_bits[MAX_ZC * (BG1_NUM_COLS - 2)];

// Print the user-facing input configuration that failed validation.
static void print_invalid_ldpc_config(const ldpc_in_config_t *inconfig) {
    fprintf(stderr, "Invalid LDPC configuration:\n");
    fprintf(stderr, "bgn=%d, A=%d, crctype=%d, E=%d, rv=%d, Qm=%d, HARQ_on=%d\n",
            inconfig->bgn, inconfig->A, inconfig->crctype, inconfig->E,
            inconfig->rv, inconfig->Qm, inconfig->HARQ_on);
}

/*
 * Unpack input bytes into one bit per byte, preserving NR's MSB-first bit order.
 *
 * Example: bit 0 of the output is bit 7 of infobytes[0].
 */
static void unpack_input_bytes_msb_first(const uint8_t *infobytes,
    uint8_t *out_bits, uint16_t num_bits) {
    for (uint16_t i = 0; i < num_bits; i++) {
        uint16_t byte_idx = i / 8;
        uint16_t bit_pos = 7 - (i % 8);
        out_bits[i] = (uint8_t)((infobytes[byte_idx] >> bit_pos) & 0x01);
    }
}

/*
 * Append the CRC remainder after the payload bits, MSB-first.
 *
 * The CRC length is ldpc_config.B - ldpc_config.A and depends on crctype.
 */
static void append_crc_bits_msb_first(uint8_t *out_bits, uint16_t offset,
    uint16_t crc_len, uint32_t crc) {
    uint16_t first_bit_pos = crc_len - 1;

    for (uint16_t i = 0; i < crc_len; i++) {
        uint16_t bit_pos = first_bit_pos - i;
        out_bits[offset + i] = (uint8_t)((crc >> bit_pos) & 0x01);
    }
}

/*
 * Encode one 5G LDPC code block and perform rate matching.
 *
 * Inputs:
 *   infobytes: Input transport/code-block payload bytes before CRC attachment.
 *   rm_bytes: Output buffer for packed rate-matched bits.
 *   inconfig: External LDPC configuration.
 *
 * Outputs:
 *   rm_bytes contains the rate-matched encoded bit stream.
 *
 * Returns:
 *   0 on success, -2 for an invalid LDPC configuration.
 */
int32_t c_ldpc_5g_encode(uint8_t *infobytes, uint8_t *rm_bytes, ldpc_in_config_t *inconfig) {
    ldpc_config_t ldpc_config;

    // Build derived LDPC parameters and select the base-graph tables.
    if(gen_ldpc_parameters(&ldpc_config, inconfig) != 0) {
        print_invalid_ldpc_config(inconfig);
        return -2;
    }

    // Attach CRC to the input bit stream before LDPC encoding.
    uint32_t crc = calculate_crc(infobytes, inconfig->A/8, inconfig->crctype);
    uint16_t crc_len = ldpc_config.B - ldpc_config.A;

    // Convert payload bytes to bits, then append the CRC bits to form B bits.
    unpack_input_bytes_msb_first(infobytes, ldpc_in_bit, inconfig->A);
    append_crc_bits_msb_first(ldpc_in_bit, ldpc_config.A, crc_len, crc);

    // Produce the LDPC code block and rate-match it to E transmitted bits.
    nr_ldpc_encode(ldpc_in_bit, ldpc_out_bits, &ldpc_config);
    nr_ldpc_rate_match(ldpc_out_bits, rm_bytes, &ldpc_config);

    return 0;
}
