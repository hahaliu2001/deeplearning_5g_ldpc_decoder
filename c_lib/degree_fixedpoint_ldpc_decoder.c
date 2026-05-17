#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

/* Layered min-sum decoder with check-degree-dependent alpha.
 * alpha_fixed = round(alpha_float * 32767)
 *   degree <= 4:  0.8125 -> 26623
 *   degree 5..8:  0.7500 -> 24575
 *   degree > 8:   0.6875 -> 22527
 *
 * Custom degree_alpha_values may be passed as a MAX_DEGREE-entry table:
 *   [0] degree 1, [1] degree 2, ..., [MAX_DEGREE - 1] degree MAX_DEGREE.
 * Pass NULL to use the defaults above.
 */
#define DEGREE_MINSUM_ALPHA_LE4 26623
#define DEGREE_MINSUM_ALPHA_5_TO_8 24575
#define DEGREE_MINSUM_ALPHA_GT8 22527

static const int16_t default_degree_minsum_alpha_values[MAX_DEGREE] = {
    DEGREE_MINSUM_ALPHA_LE4,
    DEGREE_MINSUM_ALPHA_LE4,
    DEGREE_MINSUM_ALPHA_LE4,
    DEGREE_MINSUM_ALPHA_LE4,
    DEGREE_MINSUM_ALPHA_5_TO_8,
    DEGREE_MINSUM_ALPHA_5_TO_8,
    DEGREE_MINSUM_ALPHA_5_TO_8,
    DEGREE_MINSUM_ALPHA_5_TO_8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8,
    DEGREE_MINSUM_ALPHA_GT8
};

// Current variable-node beliefs and check-to-variable messages.
static int16_t degree_LQ[BG1_NUM_COLS*MAX_ZC];
static int16_t degree_Lr[BG1_MAX_NON_NULL_ELEMENTS*MAX_ZC];

// Scratch arrays for one lifted check node.
static int16_t degree_rel_Lq[MAX_DEGREE];
static uint16_t degree_column_list[MAX_DEGREE];

/*
 * Select the normalized min-sum alpha value for a check-node degree.
 *
 * Inputs:
 *   degree: Number of variable nodes connected to this check node.
 *   degree_alpha_values: Optional MAX_DEGREE-entry alpha table in Q15 format.
 *       Entry 0 is degree 1, entry 1 is degree 2, and so on. If this pointer
 *       is NULL, the built-in default table is used.
 *
 * Output:
 *   Returns the Q15 alpha value used by the check-node update.
 */
static int16_t degree_minsum_alpha(uint16_t degree,
    const int16_t *degree_alpha_values) {
    // degree_alpha_values is indexed by exact degree: degree 1 maps to index 0.
    const int16_t *alpha_values = degree_alpha_values != NULL
        ? degree_alpha_values
        : default_degree_minsum_alpha_values;

    // Defensive bounds handling. Valid table degrees are 1..MAX_DEGREE.
    if (degree == 0) {
        return alpha_values[0];
    }
    if (degree > MAX_DEGREE) {
        return alpha_values[MAX_DEGREE - 1];
    }
    return alpha_values[degree - 1];
}

static int16_t degree_clip_llr(int32_t value, int16_t max_llr,
    int16_t min_llr) {
    if (value > (int32_t)max_llr) {
        return max_llr;
    }
    if (value < (int32_t)min_llr) {
        return min_llr;
    }
    return (int16_t)value;
}

static int16_t degree_min_sum_scale(int16_t magnitude, int16_t alpha,
    int16_t beta) {
    int32_t offset_magnitude = (int32_t)magnitude - (int32_t)beta;
    if (offset_magnitude < 0) {
        offset_magnitude = 0;
    }

    int32_t scaled = (int32_t)alpha * offset_magnitude;
    return (int16_t)((scaled + (1 << 14)) >> 15);
}

static int16_t degree_abs_llr_and_update_sign(int16_t value, int *sign) {
    if (value >= 0) {
        return value;
    }

    *sign = -*sign;
    return (value == MIN_LLR_I16) ? MAX_LLR_I16 : (int16_t)(-value);
}

static void degree_update_two_minima(int16_t value, int16_t index,
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
 * Update one lifted check node using fixed-point normalized/offset min-sum.
 *
 * Inputs:
 *   degree: Number of connected variable nodes.
 *   rel_Lq: Extrinsic variable-node beliefs, computed as LQ - old Lr.
 *   sel_lr: Check-to-variable message slice for this lifted check node.
 *   alpha: Q15 normalized min-sum scale.
 *   beta: Offset min-sum value in the same LLR scale as rel_Lq.
 *   max_llr/min_llr: Saturation bounds for updated LQ values.
 *   column_list: Variable-node indices corresponding to rel_Lq entries.
 *   LQ: Mutable variable-node belief array.
 *
 * Outputs:
 *   Updates sel_lr with new check-to-variable messages.
 *   Updates LQ[column_list[j]] in-place for every connected variable node.
 */
static void degree_min_sum_update(uint16_t degree, int16_t *rel_Lq,
    int16_t *sel_lr, int16_t alpha, int16_t beta, int16_t max_llr, int16_t min_llr,
    uint16_t *column_list, int16_t *LQ) {
    
    // Find the smallest two magnitudes and the product of all input signs.
    int16_t min1 = MAX_LLR_I16;
    int16_t min2 = MAX_LLR_I16;
    int16_t min1_idx = -1;
    int sign = 1;

    for (int j = 0; j < degree; j++) {
        int16_t abs_val = degree_abs_llr_and_update_sign(rel_Lq[j], &sign);
        degree_update_two_minima(abs_val, (int16_t)j, &min1, &min2,
                &min1_idx);
    }

    // Clip magnitudes before applying normalized/offset min-sum. In practice,
    // frequent clipping can hurt decoder performance, so input LLRs should be
    // scaled to the configured range before decoding.
    if (min1 > max_llr) min1 = max_llr;
    if (min2 > max_llr) min2 = max_llr;

    // Apply offset beta, then normalized min-sum scale alpha in Q15 format.
    int16_t new_min1 = degree_min_sum_scale(min1, alpha, beta);
    int16_t new_min2 = degree_min_sum_scale(min2, alpha, beta);

    // Write the new check-to-variable messages and update LQ in-place.
    for (uint16_t j = 0; j < degree; j++) {
        int16_t new_val = (j == min1_idx) ? new_min2 : new_min1;
        if (rel_Lq[j] >= 0) {
            sel_lr[j] = sign * new_val;
        } else {
            sel_lr[j] = -sign * new_val;
        }

        int32_t new_LQ = (int32_t)rel_Lq[j] + (int32_t)sel_lr[j];

        LQ[column_list[j]] = degree_clip_llr(new_LQ, max_llr, min_llr);
    }
}

/*
 * Decode one LDPC code block with layered fixed-point min-sum and per-degree
 * alpha values.
 *
 * Inputs:
 *   LLr: Rate-recovered input LLRs, including punctured/filler layout.
 *   hardbits: Output buffer for packed decoded bits.
 *   config: Derived LDPC configuration and base-graph table pointers.
 *   decoder_cfg: Decoder parameters such as max_iterations, beta, and LLR clips.
 *   degree_alpha_values: Optional MAX_DEGREE-entry Q15 alpha table indexed by
 *       exact degree. Pass NULL to use the default alpha values.
 *
 * Outputs:
 *   hardbits is filled MSB-first with config->B decoded bits.
 *
 * Returns:
 *   0 if CRC passes before max_iterations, 1 if decoding does not converge.
 */
int32_t layered_degree_minsum_ldpc_decode_with_alpha(int16_t *LLr,
    uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg,
    const int16_t *degree_alpha_values) {
    uint16_t non_zero_elements;
    uint16_t num_rows;
    uint16_t bgn = config->bgn;
    uint16_t Zc = config->Zc;

    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_array = config->HBG_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    // Select the active base graph tables.
    if (bgn == 1) {
        non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        num_rows = BG1_NUM_ROWS;
    } else {
        non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        num_rows = BG2_NUM_ROWS;
    }

    // Copy rate-recovered LLRs and reset messages for this decode call.
    memcpy(degree_LQ, LLr, sizeof(int16_t)*(config->N + 2*Zc));
    memset(degree_Lr, 0, sizeof(int16_t)*non_zero_elements*Zc);

    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            // HBG_offset_array gives the contiguous table slice for this row.
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            int16_t alpha = degree_minsum_alpha((uint16_t)degree,
                    degree_alpha_values);

            // HBGcolumns and Vij describe connected columns and cyclic shifts.
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            // Each base-graph row expands into Zc lifted check nodes.
            for (uint16_t zx = 0; zx < Zc; zx++) {
                int16_t *sel_lr = degree_Lr + offset*Zc + zx*degree;

                // Remove the old message before recomputing the check-node update.
                for (uint16_t j = 0; j < degree; j++) {
                    int16_t HBG_col_idx = HBGcolumns[j];
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    degree_column_list[j] = column;
                    degree_rel_Lq[j] = degree_LQ[column] - sel_lr[j];
                }

                degree_min_sum_update((uint16_t)degree, degree_rel_Lq, sel_lr,
                        alpha, decoder_cfg->beta,
                        decoder_cfg->max_llr, decoder_cfg->min_llr,
                        degree_column_list, degree_LQ);
            }
        }

        // Pack hard bits MSB-first and use CRC for early termination.
        for (int i = 0; i < config->B; i++) {
            hardbits[i/8] = (hardbits[i/8] << 1) | (degree_LQ[i] < 0 ? 1 : 0);
        }

        if (calculate_crc(hardbits, config->B/8, config->crctype) == 0) {
            return 0;
        }
    }

    return 1;
}

/*
 * Decode one LDPC code block with the built-in degree-dependent alpha table.
 *
 * Inputs:
 *   LLr: Rate-recovered input LLRs.
 *   hardbits: Output buffer for packed decoded bits.
 *   config: Derived LDPC configuration and base-graph table pointers.
 *   decoder_cfg: Decoder parameters such as max_iterations, beta, and LLR clips.
 *
 * Outputs:
 *   hardbits is filled MSB-first with config->B decoded bits.
 *
 * Returns:
 *   0 if CRC passes before max_iterations, 1 if decoding does not converge.
 */
int32_t layered_degree_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits,
    ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    // Compatibility wrapper: NULL selects the built-in degree alpha table.
    return layered_degree_minsum_ldpc_decode_with_alpha(LLr, hardbits, config,
            decoder_cfg, NULL);
}
