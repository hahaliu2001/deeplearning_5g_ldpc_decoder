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
 */
#define DEGREE_MINSUM_ALPHA_LE4 26623
#define DEGREE_MINSUM_ALPHA_5_TO_8 24575
#define DEGREE_MINSUM_ALPHA_GT8 22527

static int16_t degree_LQ[BG1_NUM_COLS*MAX_ZC];
static int16_t degree_Lr[BG1_MAX_NON_NULL_ELEMENTS*MAX_ZC];
static int16_t degree_rel_Lq[MAX_DEGREE];
static uint16_t degree_column_list[MAX_DEGREE];

static int16_t degree_minsum_alpha(uint16_t degree) {
    if (degree <= 4) {
        return DEGREE_MINSUM_ALPHA_LE4;
    }
    if (degree <= 8) {
        return DEGREE_MINSUM_ALPHA_5_TO_8;
    }
    return DEGREE_MINSUM_ALPHA_GT8;
}

static void degree_min_sum_update(uint16_t degree, int16_t *rel_Lq,
    int16_t *sel_lr, int16_t alpha, int16_t beta, int16_t max_llr, int16_t min_llr,
    uint16_t *column_list, int16_t *LQ) {
    int16_t min1 = MAX_LLR_I16;
    int16_t min2 = MAX_LLR_I16;
    int16_t min1_idx = -1;
    int16_t abs_val;
    int sign = 1;

    for (int j = 0; j < degree; j++) {
        if (rel_Lq[j] >= 0) {
            abs_val = rel_Lq[j];
        } else {
            sign = -sign;
            abs_val = -rel_Lq[j];
        }

        if (abs_val < min1) {
            min2 = min1;
            min1 = abs_val;
            min1_idx = j;
        } else if (abs_val < min2) {
            min2 = abs_val;
        }
    }

    if (min1 > max_llr) min1 = max_llr;
    if (min2 > max_llr) min2 = max_llr;

    int32_t tmp1 = ((int16_t)alpha * (int16_t)((min1 - beta) > 0 ? min1 - beta : 0));
    int32_t tmp2 = ((int16_t)alpha * (int16_t)((min2 - beta) > 0 ? min2 - beta : 0));
    int16_t new_min1 = (int16_t)((tmp1 + (1 << 14)) >> 15);
    int16_t new_min2 = (int16_t)((tmp2 + (1 << 14)) >> 15);

    for (int j = 0; j < degree; j++) {
        int16_t new_val = (j == min1_idx) ? new_min2 : new_min1;
        if (rel_Lq[j] >= 0) {
            sel_lr[j] = sign * new_val;
        } else {
            sel_lr[j] = -sign * new_val;
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

int32_t layered_degree_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits,
    ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    uint16_t non_zero_elements;
    uint16_t num_rows;
    uint16_t bgn = config->bgn;
    uint16_t Zc = config->Zc;

    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_array = config->HBG_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    if (bgn == 1) {
        non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        num_rows = BG1_NUM_ROWS;
    } else {
        non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        num_rows = BG2_NUM_ROWS;
    }

    memcpy(degree_LQ, LLr, sizeof(int16_t)*(config->N + 2*Zc));
    memset(degree_Lr, 0, sizeof(int16_t)*non_zero_elements*Zc);

    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            int16_t alpha = degree_minsum_alpha((uint16_t)degree);

            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            for (uint16_t zx = 0; zx < Zc; zx++) {
                int16_t *sel_lr = degree_Lr + offset*Zc + zx*degree;

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

        for (int i = 0; i < config->B; i++) {
            hardbits[i/8] = (hardbits[i/8] << 1) | (degree_LQ[i] < 0 ? 1 : 0);
        }

        if (calculate_crc(hardbits, config->B/8, config->crctype) == 0) {
            return 0;
        }
    }

    return 1;
}
