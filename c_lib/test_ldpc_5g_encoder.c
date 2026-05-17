#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

// Buffers for the generated encoder smoke-test input and output.
uint8_t info_bits[8448];
uint8_t info_bytes[8448/8];
ldpc_in_config_t inconfig;
uint8_t c_rmbytes[8448*256];

/*
 * Fill the LDPC configuration used by this smoke test.
 */
static void init_test_config(void) {
    inconfig.A = 512-24;
    inconfig.bgn = 1;
    inconfig.crctype = 1; // CRC24A.
    inconfig.E = 512*3;
    inconfig.rv = 0;
    inconfig.Qm = 1; // BPSK.
    inconfig.HARQ_on = 0;
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

int main() {
    // Configure one BG1 code block and generate a fresh random payload.
    srand(0);
    init_test_config();
    generate_random_info_bytes();
    
    // Run the public C encoder wrapper.
    int32_t result = c_ldpc_5g_encode(info_bytes, c_rmbytes, &inconfig);

    if (result == 0) {
        printf("Encoding successful!\n");
    } else {
        printf("Encoding failed with status %d.\n", result);
    }
    return 0;
}
