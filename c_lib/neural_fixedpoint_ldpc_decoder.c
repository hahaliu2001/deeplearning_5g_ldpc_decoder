#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

/*
 * Neural fixed-point layered min-sum LDPC decoder.
 *
 * This decoder uses the same table-driven layered schedule as
 * fixedpoint_ldpc_decoder.c. The difference is that alpha and beta are selected
 * from trained tables for each iteration and base-graph row:
 *
 *     alpha = alpha_list[iter * num_rows + HBG_row]
 *     beta  = beta_list [iter * num_rows + HBG_row]
 *
 * The alpha/beta tables are generated from trained PyTorch values with:
 *
 *     fixed_value = round(float_value * 32767)
 *
 * The fixed values are clipped to the int16_t range. For BG1, num_rows is
 * BG1_NUM_ROWS. For BG2, num_rows is BG2_NUM_ROWS.
 */

#include "BG1_trained_ldpc_weights.h"
#include "BG2_trained_ldpc_weights.h"

// Current variable-node beliefs and check-to-variable messages.
static int16_t neural_LQ[BG1_NUM_COLS * MAX_ZC];
static int16_t neural_Lr[BG1_MAX_NON_NULL_ELEMENTS * MAX_ZC];

// Scratch arrays reused for one lifted check node at a time.
static int16_t neural_rel_Lq[MAX_DEGREE];
static uint16_t neural_column_list[MAX_DEGREE];

// Base-graph-specific dimensions and trained parameter tables.
typedef struct {
    uint16_t non_zero_elements;
    uint16_t num_rows;
    const int16_t *alpha_list;
    const int16_t *beta_list;
} neural_decoder_context_t;

// Compressed row view of one base-graph row.
typedef struct {
    uint32_t offset;
    uint32_t degree;
    uint16_t *columns;
    uint16_t *shifts;
} hbg_row_t;

// Select the trained table set and dimensions for BG1 or BG2.
static int get_neural_decoder_context(uint16_t bgn,
    neural_decoder_context_t *context) {
    if (bgn == 1) {
        context->non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        context->num_rows = BG1_NUM_ROWS;
        context->alpha_list = BG1_trained_alpha_list;
        context->beta_list = BG1_trained_beta_list;
        return 0;
    }

    if (bgn == 2) {
        context->non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        context->num_rows = BG2_NUM_ROWS;
        context->alpha_list = BG2_trained_alpha_list;
        context->beta_list = BG2_trained_beta_list;
        return 0;
    }

    return -1;
}

// Return the compressed table slice for one base-graph row.
static hbg_row_t get_hbg_row(ldpc_config_t *config, uint16_t row) {
    hbg_row_t hbg_row;

    hbg_row.offset = config->HBG_offset_array[row];
    hbg_row.degree = config->HBG_offset_array[row + 1] - hbg_row.offset;
    hbg_row.columns = config->HBG_array + hbg_row.offset;
    hbg_row.shifts = config->Vij_array + hbg_row.offset;
    return hbg_row;
}

// Map a base-graph column and lifted-row index to a variable-node index.
static uint16_t lifted_column(uint16_t hbg_col, uint16_t shift, uint16_t zx,
    uint16_t Zc) {
    return hbg_col * Zc + ((shift + zx) % Zc);
}

// Convert LQ signs to packed hard bits, MSB-first in each byte.
static void pack_neural_hardbits(uint8_t *hardbits, uint16_t num_bits) {
    memset(hardbits, 0, (num_bits + 7) / 8);
    for (uint16_t i = 0; i < num_bits; i++) {
        hardbits[i / 8] = (uint8_t)((hardbits[i / 8] << 1) |
                (neural_LQ[i] < 0 ? 1 : 0));
    }
}

// Saturate a belief value to the configured LLR range.
static int16_t neural_clip_llr(int32_t value, int16_t max_llr,
    int16_t min_llr) {
    if (value > (int32_t)max_llr) {
        return max_llr;
    }
    if (value < (int32_t)min_llr) {
        return min_llr;
    }
    return (int16_t)value;
}

// Apply offset min-sum beta, then Q15 normalized min-sum alpha.
static int16_t neural_min_sum_scale(int16_t magnitude, int16_t alpha,
    int16_t beta) {
    int32_t offset_magnitude = (int32_t)magnitude - (int32_t)beta;
    if (offset_magnitude < 0) {
        offset_magnitude = 0;
    }

    int32_t scaled = (int32_t)alpha * offset_magnitude;
    return (int16_t)((scaled + (1 << 14)) >> 15);
}

// Return |value| and update the accumulated check-node sign.
static int16_t neural_abs_llr_and_update_sign(int16_t value, int *sign) {
    if (value >= 0) {
        return value;
    }

    *sign = -*sign;
    return (value == MIN_LLR_I16) ? MAX_LLR_I16 : (int16_t)(-value);
}

// Maintain the smallest and second-smallest magnitudes for min-sum.
static void neural_update_two_minima(int16_t value, int16_t index,
    int16_t *min1, int16_t *min2, int16_t *min1_idx) {
    if (value < *min1) {
        *min2 = *min1;
        *min1 = value;
        *min1_idx = index;
    } else if (value < *min2) {
        *min2 = value;
    }
}

/*
 * Update one lifted check node with fixed-point min-sum and trained alpha/beta.
 *
 * Inputs:
 *   degree: Number of connected variable nodes.
 *   rel_Lq: Extrinsic variable-node beliefs, computed as LQ - old Lr.
 *   sel_lr: Check-to-variable message slice for this lifted check node.
 *   alpha: Trained Q15 normalized min-sum scale for this iteration and row.
 *   beta: Trained offset value for this iteration and row.
 *   max_llr/min_llr: Saturation bounds for updated LQ values.
 *   column_list: Variable-node indices corresponding to rel_Lq entries.
 *   LQ: Mutable variable-node belief array.
 *
 * Outputs:
 *   Updates sel_lr with new check-to-variable messages.
 *   Updates LQ[column_list[j]] in-place for every connected variable node.
 */
static void neural_min_sum_update(uint16_t degree, int16_t *rel_Lq,
    int16_t *sel_lr, int16_t alpha, int16_t beta, int16_t max_llr,
    int16_t min_llr, uint16_t *column_list, int16_t *LQ) {
    // Find the two smallest magnitudes and the product of all input signs.
    int16_t min1 = MAX_LLR_I16;
    int16_t min2 = MAX_LLR_I16;
    int16_t min1_idx = -1;
    int sign = 1;

    for (uint16_t j = 0; j < degree; j++) {
        int16_t abs_val = neural_abs_llr_and_update_sign(rel_Lq[j], &sign);
        neural_update_two_minima(abs_val, (int16_t)j, &min1, &min2,
                &min1_idx);
    }

    // Clip magnitudes before applying offset and scaling.
    if (min1 > max_llr) {
        min1 = max_llr;
    }
    if (min2 > max_llr) {
        min2 = max_llr;
    }

    // Apply trained beta, then trained alpha in Q15 format.
    int16_t new_min1 = neural_min_sum_scale(min1, alpha, beta);
    int16_t new_min2 = neural_min_sum_scale(min2, alpha, beta);

    // Write new check-to-variable messages and update LQ in-place.
    for (uint16_t j = 0; j < degree; j++) {
        int16_t new_val = (j == (uint16_t)min1_idx) ? new_min2 : new_min1;
        if (rel_Lq[j] >= 0) {
            sel_lr[j] = (int16_t)(sign * new_val);
        } else {
            sel_lr[j] = (int16_t)(-sign * new_val);
        }

        int32_t new_LQ = (int32_t)rel_Lq[j] + (int32_t)sel_lr[j];
        LQ[column_list[j]] = neural_clip_llr(new_LQ, max_llr, min_llr);
    }
}

/*
 * Decode one LDPC code block with layered fixed-point min-sum using trained
 * alpha/beta tables indexed by iteration and base-graph row.
 *
 * Inputs:
 *   LLr: Rate-recovered input LLRs, including punctured/filler layout.
 *   hardbits: Output buffer for packed decoded bits.
 *   config: Derived LDPC configuration and base-graph table pointers.
 *   decoder_cfg: Decoder parameters such as max_iterations and LLR clips.
 *
 * Outputs:
 *   hardbits is filled MSB-first with config->B decoded bits.
 *
 * Returns:
 *   0 if CRC passes before max_iterations, 1 if decoding does not converge,
 *   -1 if config->bgn is invalid.
 */
int32_t layered_neural_minsum_ldpc_decode(
    int16_t *LLr,
    uint8_t *hardbits,
    ldpc_config_t *config,
    ldpc_decoder_cfg_t *decoder_cfg
    
) {
    neural_decoder_context_t context;
    uint16_t Zc = config->Zc;

    // Select the active base graph and trained parameter tables.
    if (get_neural_decoder_context(config->bgn, &context) != 0) {
        return -1;
    }

    // Initialize variable beliefs and reset check-to-variable messages.
    memcpy(neural_LQ, LLr, sizeof(int16_t) * (config->N + 2 * Zc));
    memset(neural_Lr, 0, sizeof(int16_t) * context.non_zero_elements * Zc);

    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < context.num_rows; HBG_row++) {
            // Get the connected columns and cyclic shifts for this HBG row.
            hbg_row_t hbg_row = get_hbg_row(config, HBG_row);

            // Trained tables are laid out as [iteration][base-graph row].
            uint32_t weight_idx = iter * context.num_rows + HBG_row;
            int16_t alpha = context.alpha_list[weight_idx];
            int16_t beta = context.beta_list[weight_idx];

            // Each base-graph row expands into Zc lifted check nodes.
            for (uint16_t zx = 0; zx < Zc; zx++) {
                int16_t *sel_lr = neural_Lr +
                        hbg_row.offset * Zc + zx * hbg_row.degree;

                // Remove the previous check-to-variable message to form the
                // extrinsic variable-to-check input.
                for (uint16_t j = 0; j < hbg_row.degree; j++) {
                    uint16_t column = lifted_column(hbg_row.columns[j],
                            hbg_row.shifts[j], zx, Zc);
                    neural_column_list[j] = column;
                    neural_rel_Lq[j] = neural_LQ[column] - sel_lr[j];
                }

                neural_min_sum_update((uint16_t)hbg_row.degree, neural_rel_Lq,
                        sel_lr, alpha, beta, decoder_cfg->max_llr,
                        decoder_cfg->min_llr, neural_column_list, neural_LQ);
            }
        }

        // Pack hard bits MSB-first and use CRC for early termination.
        pack_neural_hardbits(hardbits, config->B);

        if (calculate_crc(hardbits, config->B / 8, config->crctype) == 0) {
            return 0;
        }
    }

    return 1;
}
