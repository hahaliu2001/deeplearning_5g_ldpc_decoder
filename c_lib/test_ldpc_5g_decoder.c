#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

// Buffers for the generated end-to-end decoder smoke test.
uint8_t info_bits[8448];
int16_t LLr[BG1_NUM_COLS*MAX_ZC];
uint8_t info_bytes[8448/8];
uint8_t hardbits[8448/8];
uint8_t encoded_bytes[8448*256];
ldpc_in_config_t inconfig;
ldpc_decoder_cfg_t decoder_config;

/*
 * Fill the LDPC and decoder configuration used by this smoke test.
 */
static void init_test_config(void) {
    inconfig.A = 512-24;
    inconfig.bgn = 1;
    inconfig.crctype = 1; // CRC24A.
    inconfig.E = 512*3;
    inconfig.rv = 0;
    inconfig.Qm = 1; // BPSK.
    inconfig.HARQ_on = 0;

    decoder_config.algo = 0; // 0: BP, 1: min-sum, 2: trained min-sum, 3: degree min-sum.
    decoder_config.max_iterations = 32;
    decoder_config.alpha = 32767; // Q15 min-sum scale; ignored by BP.
    decoder_config.beta = 0 * 32767; // Offset min-sum value; ignored by BP.
    decoder_config.filling_bit_value = 128*20; // Strong positive LLR for filler bits.
    decoder_config.max_llr = 32767; // Upper LLR clipping bound / BP normalization scale.
    decoder_config.min_llr = -32768; // Lower LLR clipping bound.
}

/*
 * Generate random input bits and pack them MSB-first into bytes.
 *
 * c_ldpc_5g_encode() unpacks bit 0 from bit 7 of info_bytes[0], matching
 * numpy.packbits(..., bitorder="big") in the Python tests.
 */
static void generate_random_info_bytes(void) {
    size_t num_bytes = (inconfig.A + 7) / 8;

    memset(info_bytes, 0, num_bytes);
    for (uint16_t i = 0; i < inconfig.A; i++) {
        uint8_t bit = (uint8_t)(rand() & 0x01);
        uint16_t byte_idx = i / 8;
        uint16_t bit_pos = 7 - (i % 8);

        info_bits[i] = bit;
        info_bytes[byte_idx] |= (uint8_t)(bit << bit_pos);
    }
}

// Extract one decoded hard bit from the MSB-first packed decoder output.
static uint8_t get_packed_bit(const uint8_t *bytes, uint16_t bit_idx) {
    uint16_t byte_idx = bit_idx / 8;
    uint8_t bit_pos = (uint8_t)(7 - (bit_idx % 8));

    return (uint8_t)((bytes[byte_idx] >> bit_pos) & 0x01);
}

// Compare the decoded payload bits against the random input payload.
static int verify_decoded_payload(void) {
    for (uint16_t i = 0; i < inconfig.A; i++) {
        if (get_packed_bit(hardbits, i) != info_bits[i]) {
            fprintf(stderr, "Decoded bit mismatch at bit %u\n", i);
            return 1;
        }
    }

    return 0;
}

int main() {
    // Configure one BG1 code block and generate a fresh random payload.
    srand(0);
    init_test_config();
    generate_random_info_bytes();

    // Encode the payload and rate-match it to E packed transmit bits.
    int32_t result = c_ldpc_5g_encode(info_bytes, encoded_bytes, &inconfig);
    if (result != 0) {
        printf("Encoding failed with status %d.\n", result);
        return 1;
    }

    float noise_power_db = -20;
    // Pass the encoded bytes through modulation, AWGN, and LLR quantization.
    result = tx_rx_process(encoded_bytes, inconfig.E, inconfig.Qm, noise_power_db, LLr);
    if (result != 0) {
        printf("tx_rx_process failed with status %d.\n", result);
        return 1;
    }

    // Decode the generated LLRs and compare the decoded payload bits.
    result = c_ldpc_5g_decode(LLr, hardbits, &inconfig, &decoder_config);

    if (result == 0) {
        if (verify_decoded_payload() != 0) {
            printf("Decoding finished, but payload verification failed.\n");
            return 1;
        }
        printf("Decoding successful!\n");
    } else {
        printf("Decoding failed with status %d.\n", result);
        return 1;
    }
    return 0;
}
