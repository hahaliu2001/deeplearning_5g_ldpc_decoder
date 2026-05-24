#include <stdio.h>
#include <stdint.h>

#include "nr_ldpc.h"

extern int32_t cuda_nr_ldpc_rate_recover_harq_combine(
    ldpc_config_t *config, int16_t filling_bit_value, int16_t *input_LLr);
extern int32_t cuda_layered_minsum_ldpc_decode(int16_t *LLr,
    uint8_t *hardbits, ldpc_config_t *config,
    ldpc_decoder_cfg_t *decoder_cfg);
extern int32_t cuda_layered_float_bp_ldpc_decode(int16_t *LLr,
    uint8_t *hardbits, ldpc_config_t *config,
    ldpc_decoder_cfg_t *decoder_cfg);

// Print the user-facing input configuration that failed validation.
static void print_invalid_ldpc_config(const ldpc_in_config_t *inconfig) {
    fprintf(stderr, "Invalid LDPC configuration:\n");
    fprintf(stderr, "bgn=%d, A=%d, crctype=%d, E=%d, rv=%d, Qm=%d, HARQ_on=%d\n",
            inconfig->bgn, inconfig->A, inconfig->crctype, inconfig->E,
            inconfig->rv, inconfig->Qm, inconfig->HARQ_on);
}

/*
 * CUDA-library 5G LDPC decoder wrapper.
 *
 * This mirrors c_lib/c_ldpc_5g_decoder.c, but intentionally exposes only the
 * two decoder algorithms currently implemented in the CUDA library:
 *   1: layered fixed-point min-sum
 *   4: layered floating-point min-sum
 */
extern "C" int32_t cuda_ldpc_5g_decode(int16_t *LLr, uint8_t *hardbits,
    ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config) {
    ldpc_config_t ldpc_config;

    if(gen_ldpc_parameters(&ldpc_config, inconfig) != 0) {
        print_invalid_ldpc_config(inconfig);
        return -2;
    }

    if (cuda_nr_ldpc_rate_recover_harq_combine(&ldpc_config,
            decoder_config->filling_bit_value, LLr) != 0) {
        return -1;
    }

    switch (decoder_config->algo) {
        case 1:
            return cuda_layered_minsum_ldpc_decode(HARQ_memory, hardbits,
                &ldpc_config, decoder_config);
        case 4:
            return cuda_layered_float_bp_ldpc_decode(HARQ_memory, hardbits,
                &ldpc_config, decoder_config);
        default:
            fprintf(stderr,
                "Unsupported CUDA LDPC decoding algorithm selection: %u\n",
                decoder_config->algo);
            return -3;
    }
}
