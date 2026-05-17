#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#include "nr_ldpc.h"

/*
 * Floating-point layered belief-propagation LDPC decoder.
 *
 * This path keeps the same table-driven layered schedule as the fixed-point
 * decoders, but uses the sum-product check-node rule with tanh/atanh. It is
 * useful as a reference implementation because it is closer to the mathematical
 * BP update than the min-sum approximations.
 */

// LQ stores current variable-node beliefs as normalized floating-point LLRs.
static float LQ[BG1_NUM_COLS*MAX_ZC];

// Lr stores check-node to variable-node messages for all lifted non-zero entries.
static float Lr[BG1_MAX_NON_NULL_ELEMENTS*MAX_ZC];

// Scratch arrays reused for one lifted check node at a time.
static float rel_Lq[MAX_DEGREE];
static uint16_t column_list[MAX_DEGREE];

// tanh/atanh saturates numerically near +/-1. Use a large finite LLR instead.
#define BP_ATANH_CLIP 19.07f

// Return the table sizes needed for the selected base graph.
static void get_base_graph_dimensions(uint16_t bgn, uint16_t *non_zero_elements,
    uint16_t *num_rows) {
    if (bgn == 1) {
        *non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        *num_rows = BG1_NUM_ROWS;
    } else {
        *non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        *num_rows = BG2_NUM_ROWS;
    }
}

// Pack the first num_bits variable beliefs into hard bits, MSB-first per byte.
static void pack_hardbits_from_float(const float *beliefs, uint8_t *hardbits,
    uint16_t num_bits) {
    for (int i = 0; i < num_bits; i++) {
        hardbits[i / 8] = (hardbits[i / 8] << 1) | (beliefs[i] < 0.0f ? 1 : 0);
    }
}

// Convert an extrinsic tanh-product into the outgoing check-to-variable message.
static float bp_message_from_product(float product) {
    if (product >= 1.0f) {
        return 2.0f * BP_ATANH_CLIP;
    }
    if (product <= -1.0f) {
        return -2.0f * BP_ATANH_CLIP;
    }
    return 2.0f * atanhf(product);
}

// Compute tanh(rel_Lq/2), count exact zero inputs, and multiply non-zero terms.
static void compute_tanh_values(uint16_t degree, const float *rel_Lq,
    float *tanh_values, int16_t *num_zeros, int16_t *zero_idx,
    float *nonzero_product) {
    *num_zeros = 0;
    *zero_idx = -1;
    *nonzero_product = 1.0f;

    for (uint16_t j = 0; j < degree; j++) {
        if (rel_Lq[j] == 0.0f) {
            (*num_zeros)++;
            *zero_idx = (int16_t)j;
            tanh_values[j] = 0.0f;
        } else {
            tanh_values[j] = tanhf(rel_Lq[j] / 2.0f);
            *nonzero_product *= tanh_values[j];
        }
    }
}

// Build products before and after each edge, avoiding division by tanh(rel_Lq/2).
static void build_prefix_suffix_products(uint16_t degree,
    const float *values, float *prefix_products, float *suffix_products) {
    prefix_products[0] = 1.0f;
    for (uint16_t j = 0; j < degree; j++) {
        prefix_products[j + 1] = prefix_products[j] * values[j];
    }

    suffix_products[degree] = 1.0f;
    for (int j = (int)degree - 1; j >= 0; j--) {
        suffix_products[j] = suffix_products[j + 1] * values[j];
    }
}

/*
 * Update one lifted check node using the floating-point sum-product rule.
 *
 * Inputs:
 *   degree: Number of connected variable nodes.
 *   rel_Lq: Extrinsic variable-node beliefs, computed as LQ - old Lr.
 *   sel_lr: Check-to-variable message slice for this lifted check node.
 *   column_list: Variable-node indices corresponding to rel_Lq entries.
 *   LQ: Mutable variable-node belief array.
 *
 * Outputs:
 *   Updates sel_lr with new check-to-variable messages.
 *   Updates LQ[column_list[j]] in-place for every connected variable node.
 */
static void bp_update(uint16_t degree, float *rel_Lq,
    float *sel_lr, uint16_t *column_list, float *LQ) {
    // Zeros need special handling because tanh(0/2) is zero, and the usual
    // "total product divided by own input" form would be unstable.
    int16_t zero_idx = -1;
    int16_t num_zeros = 0;
    float nonzero_product = 1.0f;
    float tanh_values[MAX_DEGREE];

    compute_tanh_values(degree, rel_Lq, tanh_values, &num_zeros, &zero_idx,
            &nonzero_product);

    // Recompute each outgoing message from all other incoming messages.
    if (num_zeros == 1) {
        // With one zero input, only the zero edge can receive a non-zero message.
        memset(sel_lr, 0, degree * sizeof(float));
        sel_lr[zero_idx] = bp_message_from_product(nonzero_product);
    } else if (num_zeros >= 2) {
        // With two or more zero inputs, every extrinsic product includes a zero.
        memset(sel_lr, 0, degree * sizeof(float));
    } else {
        // No zero inputs: prefix/suffix products give each extrinsic product
        // without a division by the current edge value.
        float prefix_products[MAX_DEGREE + 1];
        float suffix_products[MAX_DEGREE + 1];

        build_prefix_suffix_products(degree, tanh_values, prefix_products,
                suffix_products);

        for (uint16_t j = 0; j < degree; j++) {
            float extrinsic_product = prefix_products[j] * suffix_products[j + 1];
            sel_lr[j] = bp_message_from_product(extrinsic_product);
        }
    }

    // Layered decoding makes the new beliefs visible to following rows.
    for (uint16_t j = 0; j < degree; j++) {
        LQ[column_list[j]] = rel_Lq[j] + sel_lr[j];
    }
}

static float clip_float_llr(float value, float max_llr, float min_llr) {
    if (value > max_llr) {
        return max_llr;
    }
    if (value < min_llr) {
        return min_llr;
    }
    return value;
}

static float float_min_sum_scale(float magnitude, float alpha, float beta) {
    float offset_magnitude = magnitude - beta;
    if (offset_magnitude < 0.0f) {
        offset_magnitude = 0.0f;
    }
    return alpha * offset_magnitude;
}

static void float_update_two_minima(float value, uint16_t index,
    float *min1, float *min2, uint16_t *min1_idx) {
    if (value < *min1) {
        *min2 = *min1;
        *min1 = value;
        *min1_idx = index;
    } else if (value < *min2) {
        *min2 = value;
    }
}

/*
 * Update one lifted check node using floating-point normalized/offset min-sum.
 *
 * alpha is converted from the same Q15 value used by the fixed-point decoder.
 * beta, max_llr, and min_llr are in the floating LLR scale after dequantizing
 * the int16_t channel/input representation by 128.
 */
static void float_min_sum_update(uint16_t degree, float *rel_Lq,
    float *sel_lr, float alpha, float beta, float max_llr, float min_llr,
    uint16_t *column_list, float *LQ) {
    float min1 = FLT_MAX;
    float min2 = FLT_MAX;
    uint16_t min1_idx = 0;
    int sign = 1;

    for (uint16_t j = 0; j < degree; j++) {
        float value = rel_Lq[j];
        if (value < 0.0f) {
            sign = -sign;
        }
        float abs_val = fabsf(value);
        float_update_two_minima(abs_val, j, &min1, &min2, &min1_idx);
    }

    min1 = clip_float_llr(min1, max_llr, 0.0f);
    min2 = clip_float_llr(min2, max_llr, 0.0f);

    float new_min1 = float_min_sum_scale(min1, alpha, beta);
    float new_min2 = float_min_sum_scale(min2, alpha, beta);

    for (uint16_t j = 0; j < degree; j++) {
        float new_val = (j == min1_idx) ? new_min2 : new_min1;
        float edge_sign = (rel_Lq[j] < 0.0f) ? -(float)sign : (float)sign;

        sel_lr[j] = edge_sign * new_val;
        LQ[column_list[j]] = clip_float_llr(rel_Lq[j] + sel_lr[j],
                max_llr, min_llr);
    }
}

/*
 * Decode one LDPC code block with layered floating-point belief propagation.
 *
 * Inputs:
 *   LLr: Rate-recovered input LLRs, including punctured/filler layout.
 *   hardbits: Output buffer for packed decoded bits.
 *   config: Derived LDPC configuration and base-graph table pointers.
 *   decoder_cfg: Decoder parameters for iteration count and clipping metadata.
 *
 * Outputs:
 *   hardbits is filled MSB-first with config->B decoded bits.
 *
 * Returns:
 *   0 if CRC passes before max_iterations, 1 if decoding does not converge.
 */
int32_t layered_BP_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    uint16_t non_zero_elements;
    uint16_t num_rows;

    uint16_t bgn = config->bgn;
    uint16_t Zc = config->Zc;
    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_array = config->HBG_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    // Select the active base graph dimensions.
    get_base_graph_dimensions(bgn, &non_zero_elements, &num_rows);

    // Convert quantized int16_t channel LLRs back to the float scale used by BP.
    for (int i = 0; i < config->N + 2 * Zc; i++) {
        LQ[i] = dequantize_int16_to_llr(LLr[i]);
    }

    // Check-to-variable messages start at zero for every decode call.
    for (int i = 0; i < non_zero_elements * Zc; i++) {
        Lr[i] = 0.0f;
    }
    
    // Main layered decoding loop.
    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            // HBG_offset_array gives the contiguous table slice for this row.
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            
            // HBGcolumns and Vij describe connected columns and cyclic shifts.
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            // Each base-graph row expands into Zc lifted check nodes.
            for (uint16_t zx = 0; zx < Zc; zx++) {
                float *sel_lr = Lr + offset * Zc + zx * degree;
                
                // Remove the previous check-to-variable message to form the
                // extrinsic variable-to-check input.
                for (uint16_t j = 0; j < degree; j++) {
                    int16_t HBG_col_idx = HBGcolumns[j];
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    column_list[j] = column;
                    rel_Lq[j] = LQ[column] - sel_lr[j];
                }

                // Update all edges for this lifted check node.
                bp_update((uint16_t)degree, rel_Lq, sel_lr, column_list, LQ);
            }
        }
        
        // Pack hard bits MSB-first and use CRC for early termination.
        pack_hardbits_from_float(LQ, hardbits, config->B);
        
        if (calculate_crc(hardbits, config->B / 8, config->crctype) == 0) {
            return 0;
        }
    }
    return 1;
}

/*
 * Decode one LDPC code block with layered floating-point min-sum.
 *
 * This mirrors layered_minsum_ldpc_decode(), but all variable beliefs and
 * check-to-variable messages are kept as float values. The input LLr buffer is
 * still the rate-recovered int16_t representation used by the C pipeline.
 */
int32_t layered_float_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits,
    ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    uint16_t non_zero_elements;
    uint16_t num_rows;

    uint16_t bgn = config->bgn;
    uint16_t Zc = config->Zc;
    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_array = config->HBG_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    float alpha = (float)decoder_cfg->alpha / 32768.0f;
    float beta = (float)decoder_cfg->beta / 128.0f;
    float max_llr = dequantize_int16_to_llr(decoder_cfg->max_llr);
    float min_llr = dequantize_int16_to_llr(decoder_cfg->min_llr);

    get_base_graph_dimensions(bgn, &non_zero_elements, &num_rows);

    for (int i = 0; i < config->N + 2 * Zc; i++) {
        LQ[i] = dequantize_int16_to_llr(LLr[i]);
    }

    for (int i = 0; i < non_zero_elements * Zc; i++) {
        Lr[i] = 0.0f;
    }

    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;

            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            for (uint16_t zx = 0; zx < Zc; zx++) {
                float *sel_lr = Lr + offset * Zc + zx * degree;

                for (uint16_t j = 0; j < degree; j++) {
                    int16_t HBG_col_idx = HBGcolumns[j];
                    int16_t shift = Vij[j];
                    int16_t column = HBG_col_idx * Zc + ((shift + zx) % Zc);

                    column_list[j] = column;
                    rel_Lq[j] = LQ[column] - sel_lr[j];
                }

                float_min_sum_update((uint16_t)degree, rel_Lq, sel_lr,
                        alpha, beta, max_llr, min_llr, column_list, LQ);
            }
        }

        pack_hardbits_from_float(LQ, hardbits, config->B);

        if (calculate_crc(hardbits, config->B / 8, config->crctype) == 0) {
            return 0;
        }
    }

    return 1;
}
