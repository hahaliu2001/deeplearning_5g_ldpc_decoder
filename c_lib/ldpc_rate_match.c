#include <stdio.h>
#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nr_ldpc.h"

// Maximum supported rate-matched bit count:
// 100 PRB * 12 subcarriers * 14 OFDM symbols * 8 bits/symbol (256QAM) * 4 layers.
#define MAX_E (100*12*14*8*4)
static uint8_t ek[MAX_E];

/*
 * Perform 5G LDPC rate matching.
 *
 * Inputs:
 *   inbits: Encoded LDPC code block bits. The first 2*Zc systematic bits are
 *       punctured by rate matching.
 *   outbytes: Packed output buffer for E selected and interleaved bits.
 *   config: Derived LDPC configuration, including E, N, K, k0, Zc, B, and Qm.
 *
 * Outputs:
 *   outbytes contains the rate-matched bit stream, packed MSB-first.
 */
void nr_ldpc_rate_match(uint8_t *inbits, uint8_t *outbytes,ldpc_config_t *config) {
    // Read the parameters needed by 38.212 section 5.4.2.
    uint16_t E = config->E;
    uint16_t N = config->N;
    uint16_t k0 = config->k0;
    uint16_t Zc = config->Zc;
    uint16_t B = config->B;
    uint16_t Qm = config->Qm;
    uint16_t K = config->K;

    uint8_t *outptr = outbytes;
    // Rate matching starts after the first 2*Zc punctured systematic bits.
    uint8_t *rm_in = inbits + 2*Zc;

    // Bit selection, 3GPP TS 38.212 section 5.4.2.1.
    int k=0;
    int j = 0;
    while(k < E) {
        int pos = (k0+j) % N;
        if(pos < B-2*Zc || pos >= K-2*Zc) {
            // Skip filler-bit positions; keep all other circular-buffer bits.
            ek[k++] = rm_in[pos];
        }
        j++;
    }

    // Bit interleaving, 3GPP TS 38.212 section 5.4.2.2, then pack MSB-first.
    int bit_pos = 8;
    uint8_t dbyte = 0;
    for(j=0; j<E/Qm; j++) {
        for(int i=0; i< Qm; i++) {
            dbyte += ek[i*E/Qm+j] << (bit_pos-1);
            bit_pos--;
            if(bit_pos == 0) {
                // Flush one complete output byte.
                *outptr++ = dbyte;
                dbyte = 0;
                bit_pos = 8;
            }
        }
    }

    // Flush the final partial byte when E is not byte-aligned.
    if(E % 8) {
        *outptr = dbyte;
    }

}
