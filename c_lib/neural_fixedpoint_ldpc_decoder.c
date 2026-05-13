#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

/*
 * Neural fixed-point layered min-sum LDPC decoder.
 *
 * This decoder is the same table-driven layered schedule used by
 * fixedpoint_ldpc_decoder.c, but alpha and beta are selected per iteration
 * and per HBG row:
 *
 *     alpha = alpha_list[iter * num_rows + HBG_row]
 *     beta  = beta_list [iter * num_rows + HBG_row]
 *
 * The alpha/beta lists should be generated from the trained PyTorch values by:
 *
 *     fixed_value = round(float_value * 32767)
 *
 * and clipped to the int16_t range.  For BG1, num_rows is BG1_NUM_ROWS.  For
 * BG2, num_rows is BG2_NUM_ROWS.
 */

#include "BG1_trained_ldpc_weights.h"
#include "BG2_trained_ldpc_weights.h"



static int16_t neural_LQ[BG1_NUM_COLS * MAX_ZC];
static int16_t neural_Lr[BG1_MAX_NON_NULL_ELEMENTS * MAX_ZC];
static int16_t neural_rel_Lq[MAX_DEGREE];
static uint16_t neural_column_list[MAX_DEGREE];

static void _neural_min_sum_update(
    uint16_t degree,
    int16_t *rel_Lq,
    int16_t *sel_lr,
    int16_t alpha,
    int16_t beta,
    int16_t max_llr,
    int16_t min_llr,
    uint16_t *column_list,
    int16_t *LQ
) {
    int16_t min1 = MAX_LLR_I16;
    int16_t min2 = MAX_LLR_I16;
    int16_t min1_idx = -1;
    int sign = 1;

    for (uint16_t j = 0; j < degree; j++) {
        int16_t abs_val;
        if (rel_Lq[j] >= 0) {
            abs_val = rel_Lq[j];
        } else {
            sign = -sign;
            abs_val = (rel_Lq[j] == MIN_LLR_I16) ? MAX_LLR_I16 : (int16_t)(-rel_Lq[j]);
        }

        if (abs_val < min1) {
            min2 = min1;
            min1 = abs_val;
            min1_idx = (int16_t)j;
        } else if (abs_val < min2) {
            min2 = abs_val;
        }
    }

    if (min1 > max_llr) {
        min1 = max_llr;
    }
    if (min2 > max_llr) {
        min2 = max_llr;
    }

    int16_t min1_offset = (int16_t)(((int32_t)min1 - (int32_t)beta) > 0 ? ((int32_t)min1 - (int32_t)beta) : 0);
    int16_t min2_offset = (int16_t)(((int32_t)min2 - (int32_t)beta) > 0 ? ((int32_t)min2 - (int32_t)beta) : 0);

    int32_t tmp1 = (int32_t)alpha * (int32_t)min1_offset;
    int32_t tmp2 = (int32_t)alpha * (int32_t)min2_offset;
    int16_t new_min1 = (int16_t)((tmp1 + (1 << 14)) >> 15);
    int16_t new_min2 = (int16_t)((tmp2 + (1 << 14)) >> 15);

    for (uint16_t j = 0; j < degree; j++) {
        int16_t new_val = (j == (uint16_t)min1_idx) ? new_min2 : new_min1;
        if (rel_Lq[j] >= 0) {
            sel_lr[j] = (int16_t)(sign * new_val);
        } else {
            sel_lr[j] = (int16_t)(-sign * new_val);
        }

        int32_t new_LQ = (int32_t)rel_Lq[j] + (int32_t)sel_lr[j];
        if (new_LQ > (int32_t)max_llr) {
            new_LQ = (int32_t)max_llr;
        } else if (new_LQ < (int32_t)min_llr) {
            new_LQ = (int32_t)min_llr;
        }

        LQ[column_list[j]] = (int16_t)new_LQ;
    }
}

int32_t layered_neural_minsum_ldpc_decode(
    int16_t *LLr,
    uint8_t *hardbits,
    ldpc_config_t *config,
    ldpc_decoder_cfg_t *decoder_cfg
    
) {
    uint16_t non_zero_elements;
    uint16_t num_rows;
    
    uint16_t bgn = config->bgn;
    uint16_t Zc = config->Zc;

    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_array = config->HBG_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    const int16_t *alpha_list;
    const int16_t *beta_list;

    if (bgn == 1) {
        non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        num_rows = BG1_NUM_ROWS;
        alpha_list = BG1_trained_alpha_list;
        beta_list = BG1_trained_beta_list;
    } else if (bgn == 2) {
        non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        num_rows = BG2_NUM_ROWS;
        alpha_list = BG2_trained_alpha_list;
        beta_list = BG2_trained_beta_list;

    } else {
        return -1;
    }

    memcpy(neural_LQ, LLr, sizeof(int16_t) * (config->N + 2 * Zc));
    memset(neural_Lr, 0, sizeof(int16_t) * non_zero_elements * Zc);

    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            int16_t alpha = alpha_list[iter * num_rows + HBG_row];
            int16_t beta = beta_list[iter * num_rows + HBG_row];

            for (uint16_t zx = 0; zx < Zc; zx++) {
                int16_t *sel_lr = neural_Lr + offset * Zc + zx * degree;

                for (uint16_t j = 0; j < degree; j++) {
                    int16_t HBG_col_idx = HBGcolumns[j];
                    int16_t shift = Vij[j];
                    int16_t column = (int16_t)(HBG_col_idx * Zc + ((shift + zx) % Zc));
                    neural_column_list[j] = (uint16_t)column;
                    neural_rel_Lq[j] = neural_LQ[column] - sel_lr[j];
                }

                _neural_min_sum_update(
                    (uint16_t)degree,
                    neural_rel_Lq,
                    sel_lr,
                    alpha,
                    beta,
                    decoder_cfg->max_llr,
                    decoder_cfg->min_llr,
                    neural_column_list,
                    neural_LQ
                );
            }
        }

        memset(hardbits, 0, (config->B + 7) / 8);
        for (uint16_t i = 0; i < config->B; i++) {
            hardbits[i / 8] = (uint8_t)((hardbits[i / 8] << 1) | (neural_LQ[i] < 0 ? 1 : 0));
        }

        if (calculate_crc(hardbits, config->B / 8, config->crctype) == 0) {
            return 0;
        }
    }

    return 1;
}
