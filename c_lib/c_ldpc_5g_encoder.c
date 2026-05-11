#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

static uint8_t ldpc_in_bit[MAX_ZC*22]; //max bit size for bgn=1
static uint8_t ldpc_out_bits[MAX_ZC * (BG1_NUM_COLS-2)]; //ldpc out bits, size = N(ZC*ROWS) 

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

    //crc encoder
    uint32_t crc = calculate_crc(infobytes, inconfig->A/8, inconfig->crctype);

    //unpack bytes to bits with MSB first
    for(int i=0; i<inconfig->A; i++) {
        int byte_idx = i / 8;
        int bit_pos = 7 - (i % 8);
        ldpc_in_bit[i] = (infobytes[byte_idx] >> bit_pos) & 0x01;
    }

    //unpack crc
    int first_bit_pos = ldpc_config.B-ldpc_config.A - 1;
    for(int i=0; i<ldpc_config.B-ldpc_config.A; i++) {
        int bit_pos = first_bit_pos - i;
        ldpc_in_bit[ldpc_config.A + i] = (crc >> bit_pos) & 1;
    }

    //ldpc encoding
    nr_ldpc_encode(ldpc_in_bit, ldpc_out_bits, &ldpc_config);

    //perform rate match
    nr_ldpc_rate_match(ldpc_out_bits, rm_bytes, &ldpc_config);

    return 0;
}
