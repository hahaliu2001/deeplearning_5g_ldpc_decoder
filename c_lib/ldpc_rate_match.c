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
    uint16_t F = config->F;

    uint8_t *outptr = outbytes;

    // Bit selection, 3GPP TS 38.212 section 5.4.2.1.
    //
    // Compact the circular buffer by moving the non-filler systematic bits next
    // to the parity bits.  The compacted view starts at inbits[F + 2*Zc]:
    //   [inbits[2*Zc:B]][inbits[K:N + 2*Zc]]
    //
    // This removes the filler span [B - 2*Zc, K - 2*Zc), so selection can copy
    // from a shorter circular buffer without checking every candidate position.
    memmove(inbits + F + 2*Zc, inbits + 2*Zc, B - 2*Zc);

    uint8_t *rm_in = inbits + F + 2*Zc;
    uint16_t compact_N = N - F;
    uint16_t filler_start = B - 2*Zc;
    uint16_t filler_end = K - 2*Zc;
    uint16_t compact_k0;

    if (k0 < filler_start) {
        compact_k0 = k0;
    } else if (k0 < filler_end) {
        compact_k0 = filler_start;
    } else {
        compact_k0 = k0 - F;
    }

    uint16_t k = 0;
    uint16_t pos = compact_k0;
    while(k < E) {
        uint16_t copy_len = E - k;
        uint16_t contiguous_len = compact_N - pos;

        if(copy_len > contiguous_len) {
            copy_len = contiguous_len;
        }

        memcpy(ek + k, rm_in + pos, copy_len);
        k += copy_len;
        pos += copy_len;

        // Wrap only after copying the contiguous tail of the compact buffer.
        // The next iteration will copy from rm_in[0].
        if(pos == compact_N) {
            pos = 0;
        }
    }

    // Bit interleaving, 3GPP TS 38.212 section 5.4.2.2, then pack MSB-first.
    int bit_pos = 8;
    uint8_t dbyte = 0;
    for(int j=0; j<E/Qm; j++) {
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
