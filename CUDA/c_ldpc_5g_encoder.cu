#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

static uint8_t ldpc_out_bits[MAX_ZC * BG1_NUM_COLS]; // includes the 2*Zc punctured prefix

// Unified CUDA CRC entry point. This encoder requests only the device output
// pointer so LDPC encoding can consume payload+CRC bytes without a host copy.
extern int32_t cuda_crc_encode_bytes(const uint8_t *input_bytes,
    uint8_t *out_bytes, uint8_t **device_out_bytes, uint32_t *crc,
    uint16_t A, uint16_t B, int crctype);
extern int32_t cuda_nr_ldpc_encode_packed_from_device(
    const uint8_t *device_in_bytes,
    uint8_t *out_bits, ldpc_config_t *ldpc_config);

/********* LDPC Decoder Main Function *********/
int32_t c_ldpc_5g_encode(uint8_t *infobytes, uint8_t *rm_bytes, ldpc_in_config_t *inconfig) {
    ldpc_config_t ldpc_config;

    //first generate the ldpc parameters based on the input configuration, which will be used for rate recovery and LDPC decoding
    if(gen_ldpc_parameters(&ldpc_config, inconfig) != 0) {
        fprintf(stderr, "Invalid ldpc configuration: \n");
        fprintf(stderr, "bgn= %d, A=%d,crctype=%d,E=%d, rv=%d, Qm=%d, HARQ_on=%d \n",
            inconfig->bgn, inconfig->A, inconfig->crctype, inconfig->E, inconfig->rv, inconfig->Qm, inconfig->HARQ_on);
        return -2; // Invalid ldpc configuration
    }

    uint8_t *device_ldpc_in_bytes = NULL;

    // Prepare payload bytes plus CRC bytes on CUDA. A and B are byte-aligned.
    if (cuda_crc_encode_bytes(infobytes, NULL, &device_ldpc_in_bytes, NULL,
            ldpc_config.A, ldpc_config.B, ldpc_config.crctype) != 0) {
        return -1;
    }

    //ldpc encoding
    if (cuda_nr_ldpc_encode_packed_from_device(device_ldpc_in_bytes,
            ldpc_out_bits, &ldpc_config) != 0) {
        return -1;
    }

    //perform rate match
    nr_ldpc_rate_match(ldpc_out_bits, rm_bytes, &ldpc_config);

    return 0;
}
