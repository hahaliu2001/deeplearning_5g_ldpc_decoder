#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

/********* LDPC Decoder Main Function *********/
int32_t c_ldpc_5g_decode(int16_t *LLr, uint8_t *hardbits, ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config) {
    ldpc_config_t ldpc_config;

    //first generate the ldpc parameters based on the input configuration, which will be used for rate recovery and LDPC decoding
    if(gen_ldpc_parameters(&ldpc_config, inconfig) != 0) {
        fprintf(stderr, "Invalid ldpc configuration: \n");
        fprintf(stderr, "bgn= %d, A=%d,crctype=%d,E=%d, rv=%d, Qm=%d, HARQ_on=%d \n",
            inconfig->bgn, inconfig->A, inconfig->crctype, inconfig->E, inconfig->rv, inconfig->Qm, inconfig->HARQ_on);
        return -2; // Invalid ldpc configuration
    }


    //perform rate recovery and HARQ combining if needed, the output is stored in HARQ_memory
    nr_ldpc_rate_recover_harq_combine(&ldpc_config, decoder_config->filling_bit_value, LLr);

    //ldpc decoding 
    if (decoder_config->algo == 0) {
        //sum-product algorithm, not implemented yet
        return layered_BP_ldpc_decode(HARQ_memory, hardbits, &ldpc_config, decoder_config);
    } else if (decoder_config->algo == 1) {
        //min-sum algorithm
         return layered_minsum_ldpc_decode(HARQ_memory, hardbits, &ldpc_config, decoder_config);
    } else if (decoder_config->algo == 2) {
        //neural min-sum algorithm with trained BG1/BG2 alpha and beta tables
        return layered_neural_minsum_ldpc_decode(HARQ_memory, hardbits, &ldpc_config, decoder_config);
    } else if (decoder_config->algo == 3) {
        //min-sum algorithm with degree-based alpha
        return layered_degree_minsum_ldpc_decode(HARQ_memory, hardbits, &ldpc_config, decoder_config);
    } else {
        printf("Invalid decoding algorithm selection!\n");
        return -3;    }
    
}
