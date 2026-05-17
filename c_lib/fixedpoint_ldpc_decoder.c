#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

/*
 * Fixed-point layered min-sum LDPC decoder.
 *
 * LQ stores the current variable-node belief values. Lr stores check-to-variable
 * messages for each non-zero base-graph entry and each lifted row. Layered
 * decoding updates LQ immediately after each check-node update, so later rows in
 * the same iteration use the newest available information.
 */

// LQ contains the mutable variable-node beliefs initialized from rate-recovered LLRs.
static int16_t LQ[BG1_NUM_COLS*MAX_ZC];

// Lr stores check-node to variable-node messages for all lifted non-zero entries.
static int16_t Lr[BG1_MAX_NON_NULL_ELEMENTS*MAX_ZC];

// Scratch arrays for one lifted check node.
static int16_t rel_Lq[MAX_DEGREE];
static uint16_t column_list[MAX_DEGREE];

static int16_t clip_llr(int32_t value, int16_t max_llr, int16_t min_llr) {
    if (value > (int32_t)max_llr) {
        return max_llr;
    }
    if (value < (int32_t)min_llr) {
        return min_llr;
    }
    return (int16_t)value;
}

static int16_t min_sum_scale(int16_t magnitude, int16_t alpha, int16_t beta) {
    int32_t offset_magnitude = (int32_t)magnitude - (int32_t)beta;
    if (offset_magnitude < 0) {
        offset_magnitude = 0;
    }

    int32_t scaled = (int32_t)alpha * offset_magnitude;
    return (int16_t)((scaled + (1 << 14)) >> 15);
}

static int16_t abs_llr_and_update_sign(int16_t value, int *sign) {
    if (value >= 0) {
        return value;
    }

    *sign = -*sign;
    return (value == MIN_LLR_I16) ? MAX_LLR_I16 : (int16_t)(-value);
}

static void update_two_minima(int16_t value, int16_t index,
    int16_t *min1, int16_t *min2, int16_t *min1_idx) {
    if (value < *min1) {
        *min2 = *min1;
        *min1 = value;
        *min1_idx = index;
    } else if (value < *min2) {
        *min2 = value;
    }
}

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

static void pack_hardbits(const int16_t *beliefs, uint8_t *hardbits,
    uint16_t num_bits) {
    for (int i = 0; i < num_bits; i++) {
        hardbits[i / 8] = (hardbits[i / 8] << 1) | (beliefs[i] < 0 ? 1 : 0);
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
static void min_sum_update(uint16_t degree, int16_t *rel_Lq,
    int16_t *sel_lr, int16_t alpha, int16_t beta, int16_t max_llr, int16_t min_llr,
    uint16_t *column_list, int16_t *LQ) {
    // Find the smallest two magnitudes and the product of all input signs.
    // The second minimum is used for the edge that contributed the first minimum.
    int16_t min1 = MAX_LLR_I16;
    int16_t min2 = MAX_LLR_I16;
    int16_t min1_idx = -1;
    int sign = 1;

    for (int j = 0; j < degree; j++) {
        int16_t abs_val = abs_llr_and_update_sign(rel_Lq[j], &sign);
        update_two_minima(abs_val, (int16_t)j, &min1, &min2, &min1_idx);
    }

    // Clip magnitudes before applying normalized/offset min-sum.
    // frequent clipping can hurt decoder performance, so input LLRs should be
    // scaled to the configured range before decoding.
    if (min1 > max_llr)  min1 = max_llr;
    if (min2 > max_llr)  min2 = max_llr;

    // Apply offset beta, then normalized min-sum scale alpha in Q15 format.
    int16_t new_min1 = min_sum_scale(min1, alpha, beta);
    int16_t new_min2 = min_sum_scale(min2, alpha, beta);
    
    // Write the new check-to-variable messages and update LQ in-place.
    for (int j = 0; j < degree; j++) {
        int16_t new_val = (j == min1_idx) ? new_min2 : new_min1;
        if (rel_Lq[j] >= 0) {
            sel_lr[j] = sign * new_val;
        } else {
            sel_lr[j] = -sign * new_val;
        }              
        
        // Layered decoding makes this new belief visible to following rows immediately.
        int32_t new_LQ = (int32_t)rel_Lq[j] + (int32_t)sel_lr[j];
        
        // frequent clipping can hurt decoder performance, so input LLRs should be
        // scaled to the configured range before decoding.
        LQ[column_list[j]] = clip_llr(new_LQ, max_llr, min_llr);
    }
}

/*
 * Decode one LDPC code block with layered fixed-point min-sum.
 *
 * Inputs:
 *   LLr: Rate-recovered input LLRs, including punctured/filler layout.
 *   hardbits: Output buffer for packed decoded bits.
 *   config: Derived LDPC configuration and base-graph table pointers.
 *   decoder_cfg: Decoder parameters such as max_iterations, alpha, beta, and
 *       LLR clipping limits.
 *
 * Outputs:
 *   hardbits is filled MSB-first with config->B decoded bits.
 *
 * Returns:
 *   0 if CRC passes before max_iterations, 1 if decoding does not converge.
 */
int32_t layered_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    uint16_t non_zero_elements;
    uint16_t num_rows;

    uint16_t bgn = config->bgn;
    uint16_t Zc = config->Zc;
    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_array = config->HBG_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    // Select the active base graph dimensions.
    get_base_graph_dimensions(bgn, &non_zero_elements, &num_rows);

    // Copy rate-recovered LLRs into the mutable variable-node belief array.
    memcpy(LQ, LLr, sizeof(int16_t) * (config->N + 2 * Zc));

    // Check-to-variable messages start at zero for every decode call.
    memset(Lr, 0, sizeof(int16_t) * non_zero_elements * Zc);

    // Main layered decoding loop.
    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            // HBG_offset_array gives the contiguous table slice for this row.
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            
            // HBGcolumns and Vij describe the connected base-graph columns and
            // their cyclic shifts for this base-graph row.
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            // Each base-graph row expands into Zc lifted check nodes.
            for (uint16_t zx = 0; zx < Zc; zx++) {
                int16_t *sel_lr = Lr + offset * Zc + zx * degree;
                
                // Remove the old message before recomputing the check-node update.
                for (uint16_t j = 0; j < degree; j++) {
                    int16_t HBG_col_idx = HBGcolumns[j];
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    column_list[j] = column;
                    rel_Lq[j] = LQ[column] - sel_lr[j];
                }

                // Update all edges for this lifted check node.
                min_sum_update((uint16_t)degree, rel_Lq, sel_lr,
                        decoder_cfg->alpha, decoder_cfg->beta,
                        decoder_cfg->max_llr, decoder_cfg->min_llr,
                        column_list, LQ);
            }
        }
        
        // Pack hard bits MSB-first and use CRC for early termination.
        pack_hardbits(LQ, hardbits, config->B);
        
        if (calculate_crc(hardbits, config->B / 8, config->crctype) == 0) {
            return 0;
        }
    }
    return 1;
}
