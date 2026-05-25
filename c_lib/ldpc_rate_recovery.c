#include <stdio.h>
#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nr_ldpc.h"

/*
 * Perform LDPC rate recovery and optional HARQ combining.
 *
 * Inputs:
 *   config: Derived LDPC configuration, including E, N, K, B, F, k0, and Zc.
 *   filling_bit_value: LLR assigned to filler-bit positions.
 *   input_LLr: E received LLRs after demodulation, still in rate-matched
 *       modulation-interleaved order.
 *
 * Outputs:
 *   HARQ_memory is populated with the recovered decoder input layout:
 *     [punctured 2*Zc bits][information/filler bits][parity bits].
 *   If HARQ_on is enabled, new LLRs are added to existing HARQ_memory values.
 */
void nr_ldpc_rate_recover_harq_combine(ldpc_config_t *config, int16_t filling_bit_value, int16_t *input_LLr) {
    // Read the circular-buffer parameters used by 38.212 rate recovery.
    uint16_t E = config->E;
    uint16_t N = config->N;
    uint16_t k0 = config->k0;
    uint16_t Zc = config->Zc;
    uint16_t Qm = config->Qm;
    uint16_t symbols = E / Qm;

    // Start from a clean buffer unless this call is combining with prior HARQ data.
    if(config->HARQ_on == 0){
        memset(HARQ_memory, 0, sizeof(int16_t)*(N+2*Zc));
    }

    // Restore filler-bit positions with a strong known-bit LLR.
    for (uint16_t i = 0; i < config->F; i++) {
        HARQ_memory[config->B+i] = filling_bit_value;
    }

    // Rate matching selected bits from the circular buffer excluding filler
    // positions. Put received LLRs back into their decoder-buffer positions.
    uint16_t k=0; //index for input LLR
    uint16_t pos = k0; 
    while(k < E){
        uint16_t bit_position = k / symbols;
        uint16_t symbol_idx = k % symbols;
        uint16_t llr_idx = symbol_idx * Qm + bit_position;

        // Information/systematic region after the punctured first 2*Zc bits.
        if(pos < config->B - 2*Zc){ 
            HARQ_memory[pos+2*Zc] += input_LLr[llr_idx]; // HARQ combining: sum LLRs.
            k++;
        }

        // Parity region. Positions between B-2*Zc and K-2*Zc are filler bits.
        else if(pos >= config->K - 2*Zc){
            HARQ_memory[pos+2*Zc] += input_LLr[llr_idx]; // HARQ combining: sum LLRs.
            k++;
        }
        pos = (pos + 1) % N; // Next position in the circular buffer.
    }
}
