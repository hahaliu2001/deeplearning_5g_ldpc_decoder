#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "../c_lib/nr_ldpc.h"

/* CUDA version of c_lib/degree_fixedpoint_ldpc_decoder.c.
 *
 * The layered schedule is preserved by launching one kernel per base-graph row.
 * Inside each row, Zc lifted check nodes are processed in parallel because they
 * touch distinct circulant columns for a given base-graph row.
 */
#define DEGREE_MINSUM_ALPHA_LE4 26623
#define DEGREE_MINSUM_ALPHA_5_TO_8 24575
#define DEGREE_MINSUM_ALPHA_GT8 22527

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(cuda_check_status));                  \
            return -1;                                                       \
        }                                                                    \
    } while (0)

static __host__ __device__ int16_t cuda_degree_minsum_alpha(uint16_t degree)
{
    if (degree <= 4) {
        return DEGREE_MINSUM_ALPHA_LE4;
    }
    if (degree <= 8) {
        return DEGREE_MINSUM_ALPHA_5_TO_8;
    }
    return DEGREE_MINSUM_ALPHA_GT8;
}

static __device__ void cuda_degree_min_sum_update(
    uint16_t degree,
    int16_t *rel_Lq,
    int16_t *sel_lr,
    int16_t alpha,
    int16_t beta,
    int16_t max_llr,
    int16_t min_llr,
    uint16_t *column_list,
    int16_t *LQ)
{
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
            abs_val = (int16_t)-rel_Lq[j];
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

    int32_t tmp1 = (int32_t)alpha *
        (int32_t)((min1 - beta) > 0 ? min1 - beta : 0);
    int32_t tmp2 = (int32_t)alpha *
        (int32_t)((min2 - beta) > 0 ? min2 - beta : 0);
    int16_t new_min1 = (int16_t)((tmp1 + (1 << 14)) >> 15);
    int16_t new_min2 = (int16_t)((tmp2 + (1 << 14)) >> 15);

    for (uint16_t j = 0; j < degree; j++) {
        int16_t new_val = (j == (uint16_t)min1_idx) ? new_min2 : new_min1;
        sel_lr[j] = (rel_Lq[j] >= 0) ?
            (int16_t)(sign * new_val) :
            (int16_t)(-sign * new_val);

        int32_t new_LQ = (int32_t)rel_Lq[j] + (int32_t)sel_lr[j];
        if (new_LQ > (int32_t)max_llr) {
            new_LQ = (int32_t)max_llr;
        } else if (new_LQ < (int32_t)min_llr) {
            new_LQ = (int32_t)min_llr;
        }

        LQ[column_list[j]] = (int16_t)new_LQ;
    }
}

static __global__ void cuda_degree_layer_kernel(
    int16_t *LQ,
    int16_t *Lr,
    const uint16_t *Vij_array,
    const uint16_t *HBG_array,
    const uint16_t *HBG_offset_array,
    uint16_t HBG_row,
    uint16_t Zc,
    int16_t beta,
    int16_t max_llr,
    int16_t min_llr)
{
    uint16_t zx = (uint16_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (zx >= Zc) {
        return;
    }

    uint32_t offset = HBG_offset_array[HBG_row];
    uint16_t degree = (uint16_t)(HBG_offset_array[HBG_row + 1] - offset);
    int16_t alpha = cuda_degree_minsum_alpha(degree);

    const uint16_t *HBGcolumns = HBG_array + offset;
    const uint16_t *Vij = Vij_array + offset;
    int16_t *sel_lr = Lr + offset * Zc + zx * degree;

    int16_t rel_Lq[MAX_DEGREE];
    uint16_t column_list[MAX_DEGREE];

    for (uint16_t j = 0; j < degree; j++) {
        uint16_t HBG_col_idx = HBGcolumns[j];
        uint16_t shift = Vij[j];
        uint16_t column = (uint16_t)(HBG_col_idx * Zc + ((shift + zx) % Zc));
        column_list[j] = column;
        rel_Lq[j] = (int16_t)(LQ[column] - sel_lr[j]);
    }

    cuda_degree_min_sum_update(degree, rel_Lq, sel_lr, alpha, beta,
                               max_llr, min_llr, column_list, LQ);
}

static __global__ void cuda_pack_hardbits_kernel(
    const int16_t *LQ,
    uint8_t *hardbits,
    uint16_t B)
{
    uint32_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_bytes = (B + 7u) / 8u;
    if (byte_idx >= num_bytes) {
        return;
    }

    uint8_t out = 0;
    for (uint32_t bit = 0; bit < 8u; bit++) {
        uint32_t idx = byte_idx * 8u + bit;
        out = (uint8_t)(out << 1);
        if (idx < B && LQ[idx] < 0) {
            out = (uint8_t)(out | 1u);
        }
    }
    hardbits[byte_idx] = out;
}

extern "C" int32_t cuda_layered_degree_minsum_ldpc_decode(
    int16_t *LLr,
    uint8_t *hardbits,
    ldpc_config_t *config,
    ldpc_decoder_cfg_t *decoder_cfg)
{
    uint16_t non_zero_elements;
    uint16_t num_rows;
    uint16_t bgn = config->bgn;
    uint16_t Zc = config->Zc;
    uint32_t lq_count = (uint32_t)config->N + 2u * (uint32_t)Zc;

    if (bgn == 1) {
        non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        num_rows = BG1_NUM_ROWS;
    } else {
        non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        num_rows = BG2_NUM_ROWS;
    }

    int16_t *d_LQ = NULL;
    int16_t *d_Lr = NULL;
    uint8_t *d_hardbits = NULL;
    uint16_t *d_Vij_array = NULL;
    uint16_t *d_HBG_array = NULL;
    uint16_t *d_HBG_offset_array = NULL;

    uint32_t hardbit_bytes = ((uint32_t)config->B + 7u) / 8u;
    uint32_t row_offsets = (uint32_t)num_rows + 1u;
    uint32_t lr_count = (uint32_t)non_zero_elements * (uint32_t)Zc;

    CUDA_CHECK(cudaMalloc((void **)&d_LQ, sizeof(int16_t) * lq_count));
    CUDA_CHECK(cudaMalloc((void **)&d_Lr, sizeof(int16_t) * lr_count));
    CUDA_CHECK(cudaMalloc((void **)&d_hardbits, sizeof(uint8_t) * hardbit_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_Vij_array, sizeof(uint16_t) * non_zero_elements));
    CUDA_CHECK(cudaMalloc((void **)&d_HBG_array, sizeof(uint16_t) * non_zero_elements));
    CUDA_CHECK(cudaMalloc((void **)&d_HBG_offset_array, sizeof(uint16_t) * row_offsets));

    CUDA_CHECK(cudaMemcpy(d_LQ, LLr, sizeof(int16_t) * lq_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Lr, 0, sizeof(int16_t) * lr_count));
    CUDA_CHECK(cudaMemcpy(d_Vij_array, config->Vij_array,
                          sizeof(uint16_t) * non_zero_elements,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_HBG_array, config->HBG_array,
                          sizeof(uint16_t) * non_zero_elements,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_HBG_offset_array, config->HBG_offset_array,
                          sizeof(uint16_t) * row_offsets,
                          cudaMemcpyHostToDevice));

    const int threads = 128;
    dim3 layer_blocks((Zc + threads - 1) / threads);
    dim3 hardbit_blocks((hardbit_bytes + threads - 1) / threads);
    int32_t status = 1;

    for (uint16_t iter = 0; iter < decoder_cfg->max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            cuda_degree_layer_kernel<<<layer_blocks, threads>>>(
                d_LQ,
                d_Lr,
                d_Vij_array,
                d_HBG_array,
                d_HBG_offset_array,
                HBG_row,
                Zc,
                (int16_t)decoder_cfg->beta,
                decoder_cfg->max_llr,
                decoder_cfg->min_llr);
            CUDA_CHECK(cudaGetLastError());
        }

        cuda_pack_hardbits_kernel<<<hardbit_blocks, threads>>>(
            d_LQ, d_hardbits, config->B);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hardbits, d_hardbits, sizeof(uint8_t) * hardbit_bytes,
                              cudaMemcpyDeviceToHost));

        if (calculate_crc(hardbits, config->B / 8, config->crctype) == 0) {
            status = 0;
            break;
        }
    }

    cudaFree(d_HBG_offset_array);
    cudaFree(d_HBG_array);
    cudaFree(d_Vij_array);
    cudaFree(d_hardbits);
    cudaFree(d_Lr);
    cudaFree(d_LQ);

    return status;
}

int32_t layered_degree_minsum_ldpc_decode(
    int16_t *LLr,
    uint8_t *hardbits,
    ldpc_config_t *config,
    ldpc_decoder_cfg_t *decoder_cfg)
{
    return cuda_layered_degree_minsum_ldpc_decode(
        LLr, hardbits, config, decoder_cfg);
}
