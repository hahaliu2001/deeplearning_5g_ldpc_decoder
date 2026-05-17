#ifndef BG1_TRAINED_LDPC_WEIGHTS_H
#define BG1_TRAINED_LDPC_WEIGHTS_H

#include <stdint.h>

#define BG1_TRAINED_LDPC_BGN 1
#define BG1_TRAINED_LDPC_NUM_ITERATIONS 32
#define BG1_TRAINED_LDPC_NUM_ROWS 46

/*
 * Alpha and beta are stored as flat [iteration][HBG row] tables.
 *
 *   table_index = iteration * BG1_TRAINED_LDPC_NUM_ROWS + HBG_row
 *
 * The list size must therefore be NUM_ITERATIONS * NUM_ROWS.
 */
#define BG1_TRAINED_LDPC_ALPHA_BETA_LIST_SIZE 1472

/*
 * LDPC configuration used when these weights were generated. These values are
 * metadata for traceability; the decoder still receives its active LDPC
 * configuration from ldpc_in_config_t/gen_ldpc_parameters().
 */
#define BG1_TRAINED_LDPC_A 488
#define BG1_TRAINED_LDPC_B 512
#define BG1_TRAINED_LDPC_E 2048
#define BG1_TRAINED_LDPC_ZC 24
#define BG1_TRAINED_LDPC_CRC_TYPE 1
#define BG1_TRAINED_LDPC_RV 0
#define BG1_TRAINED_LDPC_QM 1

extern const int16_t BG1_trained_alpha_list[BG1_TRAINED_LDPC_ALPHA_BETA_LIST_SIZE];
extern const int16_t BG1_trained_beta_list[BG1_TRAINED_LDPC_ALPHA_BETA_LIST_SIZE];

#endif /* BG1_TRAINED_LDPC_WEIGHTS_H */
