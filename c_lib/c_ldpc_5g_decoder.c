#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

// Print the user-facing input configuration that failed validation.
static void print_invalid_ldpc_config(const ldpc_in_config_t *inconfig) {
    fprintf(stderr, "Invalid LDPC configuration:\n");
    fprintf(stderr, "bgn=%d, A=%d, crctype=%d, E=%d, rv=%d, Qm=%d, HARQ_on=%d\n",
            inconfig->bgn, inconfig->A, inconfig->crctype, inconfig->E,
            inconfig->rv, inconfig->Qm, inconfig->HARQ_on);
}

/*
 * Run 5G LDPC rate recovery/HARQ combining and dispatch to the selected decoder.
 *
 * Inputs:
 *   LLr: Input LLRs after demodulation and rate matching order.
 *   hardbits: Output buffer for packed decoded bits.
 *   inconfig: External transport-block/code-block configuration.
 *   decoder_config: Decoder algorithm selection and numeric parameters.
 *
 * Outputs:
 *   hardbits is filled by the selected decoder when decoding is attempted.
 *
 * Returns:
 *   0 if decoding succeeds, 1 if decoding does not converge, -2 for invalid
 *   LDPC configuration, and -3 for invalid algorithm selection. Other negative
 *   values may be returned by a selected decoder implementation.
 */
int32_t c_ldpc_5g_decode(int16_t *LLr, uint8_t *hardbits, ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config) {
    ldpc_config_t ldpc_config;

    // Build derived LDPC parameters and select the base-graph tables.
    if(gen_ldpc_parameters(&ldpc_config, inconfig) != 0) {
        print_invalid_ldpc_config(inconfig);
        return -2;
    }

    // Perform rate recovery and optional HARQ combining. The decoder kernels all
    // consume HARQ_memory, whose layout includes the punctured and filler regions.
    nr_ldpc_rate_recover_harq_combine(&ldpc_config, decoder_config->filling_bit_value, LLr);

    // Algorithm selector:
    //   0: floating-point BP
    //   1: fixed-point min-sum
    //   2: trained-table neural min-sum
    //   3: degree-dependent min-sum
    //   4: floating-point min-sum
    switch (decoder_config->algo) {
        case 0:
            // Floating-point layered belief propagation.
            return layered_BP_ldpc_decode(HARQ_memory, hardbits, &ldpc_config,
                    decoder_config);
        case 1:
            // Fixed-point layered min-sum using one configured alpha/beta pair.
            return layered_minsum_ldpc_decode(HARQ_memory, hardbits,
                    &ldpc_config, decoder_config);
        case 2:
            // Fixed-point layered min-sum using trained per-row alpha/beta tables.
            return layered_neural_minsum_ldpc_decode(HARQ_memory, hardbits,
                    &ldpc_config, decoder_config);
        case 3:
            // Fixed-point layered min-sum using degree-dependent alpha values.
            return layered_degree_minsum_ldpc_decode(HARQ_memory, hardbits,
                    &ldpc_config, decoder_config);
        case 4:
            // Floating-point layered min-sum using one configured alpha/beta pair.
            return layered_float_minsum_ldpc_decode(HARQ_memory, hardbits,
                    &ldpc_config, decoder_config);
        default:
            fprintf(stderr, "Invalid decoding algorithm selection: %u\n",
                    decoder_config->algo);
            return -3;
    }
}
