#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

// First four parity groups; each group contains Zc bits.
static uint8_t pc[4][MAX_ZC];

// Syndromes from the information/filler part of the first four HBG rows.
// In the optimized encoder derivation, these are the A * c terms.
static uint8_t L[4][MAX_ZC];

// Modulo-2 sum of L[0]..L[3], used to solve the first parity group pc[0].
static uint8_t Lsum[MAX_ZC];

typedef struct {
    uint32_t degree;
    uint16_t *columns;
    uint16_t *shifts;
} hbg_row_t;

static uint16_t encoder_num_rows(uint16_t bgn) {
    return (bgn == 1) ? BG1_NUM_ROWS : BG2_NUM_ROWS;
}

static hbg_row_t get_hbg_row(ldpc_config_t *config, uint16_t row) {
    uint32_t offset = config->HBG_offset_array[row];
    hbg_row_t hbg_row;

    hbg_row.degree = config->HBG_offset_array[row + 1] - offset;
    hbg_row.columns = config->HBG_array + offset;
    hbg_row.shifts = config->Vij_array + offset;
    return hbg_row;
}

// Map a base-graph column and lifted-row index to a codeword bit index.
static uint16_t lifted_column(uint16_t hbg_col, uint16_t shift, uint16_t zx,
    uint16_t Zc) {
    return hbg_col * Zc + ((shift + zx) % Zc);
}

static void prepare_c_sequence(uint8_t *in_bits, uint8_t *out_bits,
    uint16_t B, uint16_t F) {
    // The systematic sequence c starts with B input bits followed by F filler zeros.
    memcpy(out_bits, in_bits, B);
    memset(out_bits + B, 0, F);
}

static void compute_first_four_syndromes(uint8_t *cseq,
    ldpc_config_t *config, uint16_t Kb) {
    uint16_t Zc = config->Zc;

    /*
     * Compute L = A * c for the first four base-graph rows.
     *
     * For each non-zero element in an HBG row:
     *   columns[j] is the connected base-graph column.
     *   shifts[j] is the circular shift value Vij.
     *
     * Only columns < Kb belong to the information/filler part. The first four
     * parity columns are intentionally excluded here because pc[0]..pc[3] are
     * solved separately from the special 4x4 parity submatrix.
     */
    for (uint16_t row = 0; row < 4; row++) {
        hbg_row_t hbg_row = get_hbg_row(config, row);

        for (uint16_t zx = 0; zx < Zc; zx++) {
            uint16_t sum = 0;

            for (uint32_t j = 0; j < hbg_row.degree; j++) {
                uint16_t hbg_col = hbg_row.columns[j];
                if (hbg_col < Kb) {
                    uint16_t column = lifted_column(hbg_col, hbg_row.shifts[j],
                            zx, Zc);
                    sum += cseq[column];
                }
            }

            L[row][zx] = (uint8_t)(sum % 2);
        }
    }
}

static void compute_lsum(uint16_t Zc) {
    // Sum the first four syndromes in GF(2).
    for (uint16_t zx = 0; zx < Zc; zx++) {
        Lsum[zx] = (uint8_t)((L[0][zx] + L[1][zx] +
                L[2][zx] + L[3][zx]) % 2);
    }
}

static void get_first_parity_shifts(ldpc_config_t *config, uint16_t *shift0,
    uint16_t *shift1) {
    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    if (config->bgn == 1) {
        // BG1 uses I1 from row 0 and I2 from row 1.
        *shift0 = Vij_array[HBG_offset_array[0] + 17]; // I1 in BG1 row 0.
        *shift1 = Vij_array[HBG_offset_array[1] + 16]; // I2 in BG1 row 1.
    } else {
        // BG2 uses I3 from row 0 and I4 from row 2.
        *shift0 = Vij_array[HBG_offset_array[0] + 6];  // I3 in BG2 row 0.
        *shift1 = Vij_array[HBG_offset_array[2] + 5];  // I4 in BG2 row 2.
    }
}

static void compute_first_four_parity_groups(ldpc_config_t *config) {
    uint16_t shift0;
    uint16_t shift1;
    uint16_t Zc = config->Zc;

    /*
     * Solve pc[0]..pc[3], the first four Zc-bit parity groups.
     *
     * These groups correspond to HBG columns 22..25 for BG1 and 10..13 for BG2.
     * The first four parity equations can be written as:
     *
     *     L + B * [pc0 pc1 pc2 pc3]^T = 0
     *
     * Here each I* entry is a Zc-by-Zc circular permutation matrix, I is the
     * identity matrix, and 0 is the all-zero matrix.
     *
     *   BG1 B = [ I1 I  0  0 ]      BG2 B = [ I3 I  0  0 ]
     *           [ I2 I  I  0 ]              [ 0  I  I  0 ]
     *           [ 0  0  I  I ]              [ I4 0  I  I ]
     *           [ I1 0  0  I ]              [ I3 0  0  I ]
     *
     * Adding the four equations in GF(2) cancels pc[1], pc[2], and pc[3].
     * What remains is I2*pc[0] for BG1 or I4*pc[0] for BG2, so pc[0] is
     * obtained by applying the corresponding circular shift to Lsum.
     */
    get_first_parity_shifts(config, &shift0, &shift1);

    // pc[0] is Lsum shifted by the I2/I4 permutation.
    for (uint16_t zx = 0; zx < Zc; zx++) {
        pc[0][(shift1 + zx) % Zc] = Lsum[zx];
    }

    // pc[1] = L[0] + I1/I3 * pc[0]
    // pc[3] = L[3] + I1/I3 * pc[0]
    for (uint16_t zx = 0; zx < Zc; zx++) {
        uint8_t shifted_pc0 = pc[0][(shift0 + zx) % Zc];
        pc[1][zx] = (uint8_t)((L[0][zx] + shifted_pc0) % 2);
        pc[3][zx] = (uint8_t)((L[3][zx] + shifted_pc0) % 2);
    }

    // The equation for pc[2] is different for BG1 and BG2.
    for (uint16_t zx = 0; zx < Zc; zx++) {
        if (config->bgn == 1) {
            // BG1: pc[2] = L[2] + pc[3]
            pc[2][zx] = (uint8_t)((L[2][zx] + pc[3][zx]) % 2);
        } else {
            // BG2: pc[2] = L[1] + pc[1]
            pc[2][zx] = (uint8_t)((L[1][zx] + pc[1][zx]) % 2);
        }
    }
}

static void write_first_four_parity_groups(uint8_t *wseq, uint16_t Zc) {
    // Store pc[0]..pc[3] as the first four parity groups in w.
    for (uint16_t zx = 0; zx < Zc; zx++) {
        wseq[zx] = pc[0][zx];
        wseq[Zc + zx] = pc[1][zx];
        wseq[2 * Zc + zx] = pc[2][zx];
        wseq[3 * Zc + zx] = pc[3][zx];
    }
}

static void compute_additional_parity_groups(uint8_t *cseq, uint8_t *wseq,
    ldpc_config_t *config, uint16_t Kb, uint16_t num_rows) {
    uint16_t Zc = config->Zc;

    /*
     * Compute the remaining parity groups from the rest of the HBG rows.
     *
     * For rows >= 4, each parity bit is the row sum over [c, pc[0..3]].
     * Columns < Kb select information/filler bits, while columns Kb..Kb+3
     * select the first four parity groups. Later parity columns are not part
     * of this direct computation.
     */
    for (uint16_t row = 4; row < num_rows; row++) {
        hbg_row_t hbg_row = get_hbg_row(config, row);

        for (uint16_t zx = 0; zx < Zc; zx++) {
            uint16_t parity_sum = 0;

            for (uint32_t j = 0; j < hbg_row.degree; j++) {
                uint16_t hbg_col = hbg_row.columns[j];
                if (hbg_col < Kb + 4) {
                    uint16_t column = lifted_column(hbg_col, hbg_row.shifts[j],
                            zx, Zc);
                    parity_sum += cseq[column];
                }
            }

            wseq[row * Zc + zx] = (uint8_t)(parity_sum % 2);
        }
    }
}

/*
 * Encode one 5G LDPC code block.
 *
 * Inputs:
 *   in_bits: B input bits, including CRC bits.
 *   out_bits: Buffer for systematic bits, filler bits, and parity bits.
 *   ldpc_config: Derived LDPC configuration and base-graph table pointers.
 *
 * Outputs:
 *   out_bits[0:B] contains the input bits.
 *   out_bits[B:K] contains zero-valued filler bits.
 *   out_bits[K:] contains the generated parity sequence w.
 *
 * Returns:
 *   0 on success.
 */
int32_t nr_ldpc_encode(uint8_t *in_bits, uint8_t *out_bits,
    ldpc_config_t *ldpc_config) {
    uint16_t Zc = ldpc_config->Zc;
    uint16_t K = ldpc_config->K;
    uint16_t B = ldpc_config->B;
    uint16_t F = ldpc_config->F;
    uint16_t Kb = K / Zc;
    uint16_t num_rows = encoder_num_rows(ldpc_config->bgn);

    prepare_c_sequence(in_bits, out_bits, B, F);

    uint8_t *cseq = out_bits;
    uint8_t *wseq = out_bits + K;

    compute_first_four_syndromes(cseq, ldpc_config, Kb);
    compute_lsum(Zc);
    compute_first_four_parity_groups(ldpc_config);
    write_first_four_parity_groups(wseq, Zc);
    compute_additional_parity_groups(cseq, wseq, ldpc_config, Kb, num_rows);

    return 0;
}
