#include <stdio.h>
#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ldpc_decoder.h"

void nr_ldpc_rate_recover_harq_combine(ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg, int16_t *input_LLr) {
    // This function performs LDPC rate recovery based on the input configuration and input LLR.
    // The output bits are stored in HARQ_memory.

    //read parameters from config
    uint16_t E = config->E;
    uint16_t N = config->N;
    uint16_t k0 = config->k0;
    uint16_t Zc = config->Zc;

    //clean memory if not HARQ combining
    if(config->HARQ_on == 0){
        memset(HARQ_memory, 0, sizeof(int16_t)*(N+2*Zc));
    }

    //fill filling bits with filling_bit_LLRvalue
    for (uint16_t i = 0; i < config->F; i++) {
        HARQ_memory[config->B+i] = decoder_cfg->filling_bit_value;
    }

    //rate matching data is [2*Zc:B,K:E]
    // rate recover data shall put back to[2*Zc:B, K:N+2*Zc]
    uint16_t k=0; //index for input LLR
    uint16_t pos = k0; 
    while(k < E){
        //if it is information bit position
        if(pos < config->B - 2*Zc){ 
            HARQ_memory[pos+2*Zc] += input_LLr[k++]; //HARQ combining by summing LLRs
        }

        //if it is parity bit position
        else if(pos >= config->K - 2*Zc){
            HARQ_memory[pos+2*Zc] += input_LLr[k++]; //HARQ combining by summing LLRs
        }
        pos = (pos + 1) % N; //index in the circular bufferidx++;
    }
}