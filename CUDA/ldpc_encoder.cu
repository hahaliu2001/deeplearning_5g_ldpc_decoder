#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

#define CUDA_CHECK_ENCODE(call)                                              \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA encoder error %s:%d: %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(cuda_check_status));        \
            return -1;                                                       \
        }                                                                    \
    } while (0)

typedef struct {
    uint8_t *in_bits;
    uint8_t *out_bits;
    uint8_t *pc;
    uint16_t *Vij_array;
    uint16_t *HBG_array;
    uint16_t *HBG_offset_array;
    uint32_t in_bits_capacity;
    uint32_t out_bits_capacity;
    uint32_t non_zero_capacity;
    uint32_t row_offsets_capacity;
    const uint16_t *host_Vij_array;
    const uint16_t *host_HBG_array;
    const uint16_t *host_HBG_offset_array;
    uint16_t cached_bgn;
    uint16_t cached_Zc;
    uint16_t cached_iLS;
} cuda_encoder_workspace_t;

static cuda_encoder_workspace_t g_encoder_workspace;

static int cuda_encoder_ensure_workspace(
    cuda_encoder_workspace_t *ws,
    uint32_t in_bits_count,
    uint32_t out_bits_count,
    uint32_t non_zero_elements,
    uint32_t row_offsets)
{
    if (ws->in_bits_capacity < in_bits_count) {
        cudaFree(ws->in_bits);
        ws->in_bits = NULL;
        CUDA_CHECK_ENCODE(cudaMalloc((void **)&ws->in_bits,
            sizeof(uint8_t) * in_bits_count));
        ws->in_bits_capacity = in_bits_count;
    }

    if (ws->out_bits_capacity < out_bits_count) {
        cudaFree(ws->out_bits);
        ws->out_bits = NULL;
        CUDA_CHECK_ENCODE(cudaMalloc((void **)&ws->out_bits,
            sizeof(uint8_t) * out_bits_count));
        ws->out_bits_capacity = out_bits_count;
    }

    if (ws->pc == NULL) {
        CUDA_CHECK_ENCODE(cudaMalloc((void **)&ws->pc,
            sizeof(uint8_t) * 4u * MAX_ZC));
    }

    if (ws->non_zero_capacity < non_zero_elements) {
        cudaFree(ws->Vij_array);
        cudaFree(ws->HBG_array);
        ws->Vij_array = NULL;
        ws->HBG_array = NULL;
        CUDA_CHECK_ENCODE(cudaMalloc((void **)&ws->Vij_array,
            sizeof(uint16_t) * non_zero_elements));
        CUDA_CHECK_ENCODE(cudaMalloc((void **)&ws->HBG_array,
            sizeof(uint16_t) * non_zero_elements));
        ws->non_zero_capacity = non_zero_elements;
        ws->host_Vij_array = NULL;
        ws->host_HBG_array = NULL;
    }

    if (ws->row_offsets_capacity < row_offsets) {
        cudaFree(ws->HBG_offset_array);
        ws->HBG_offset_array = NULL;
        CUDA_CHECK_ENCODE(cudaMalloc((void **)&ws->HBG_offset_array,
            sizeof(uint16_t) * row_offsets));
        ws->row_offsets_capacity = row_offsets;
        ws->host_HBG_offset_array = NULL;
    }

    return 0;
}

static __global__ void cuda_encoder_parity_kernel(
    uint8_t *out_bits,
    uint8_t *pc,
    const uint16_t *Vij_array,
    const uint16_t *HBG_array,
    const uint16_t *HBG_offset_array,
    uint16_t bgn,
    uint16_t num_rows,
    uint16_t Zc,
    uint16_t K,
    uint16_t Kb)
{
    uint16_t zx = (uint16_t)threadIdx.x;

    uint8_t L0 = 0;
    uint8_t L1 = 0;
    uint8_t L2 = 0;
    uint8_t L3 = 0;
    uint8_t *L[4] = {&L0, &L1, &L2, &L3};

    for (uint16_t HBG_row = 0; HBG_row < 4; HBG_row++) {
        uint32_t offset = HBG_offset_array[HBG_row];
        uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
        const uint16_t *HBGcolumns = HBG_array + offset;
        const uint16_t *Vij = Vij_array + offset;

        uint8_t acc = 0;
        for (uint32_t j = 0; j < degree; j++) {
            uint16_t HBG_col_idx = HBGcolumns[j];
            if (HBG_col_idx < Kb) {
                uint16_t shift = Vij[j];
                uint16_t column = (uint16_t)(HBG_col_idx * Zc +
                    ((shift + zx) % Zc));
                acc ^= (uint8_t)(out_bits[column] & 1u);
            }
        }
        *L[HBG_row] = acc;
    }

    uint16_t shift0;
    uint16_t shift1;
    if (bgn == 1) {
        shift0 = Vij_array[HBG_offset_array[0] + 17];
        shift1 = Vij_array[HBG_offset_array[1] + 16];
    } else {
        shift0 = Vij_array[HBG_offset_array[0] + 6];
        shift1 = Vij_array[HBG_offset_array[2] + 5];
    }

    uint8_t *wseq = out_bits + K;
    uint8_t pc0 = (uint8_t)(L0 ^ L1 ^ L2 ^ L3);
    pc[(0 * MAX_ZC) + ((shift1 + zx) % Zc)] = pc0;

    __syncthreads();

    uint8_t shifted_pc0 = pc[(0 * MAX_ZC) + ((shift0 + zx) % Zc)];
    uint8_t pc1 = (uint8_t)(L0 ^ shifted_pc0);
    uint8_t pc3 = (uint8_t)(L3 ^ shifted_pc0);
    uint8_t pc2 = (bgn == 1) ? (uint8_t)(L2 ^ pc3) : (uint8_t)(L1 ^ pc1);

    pc[(1 * MAX_ZC) + zx] = pc1;
    pc[(2 * MAX_ZC) + zx] = pc2;
    pc[(3 * MAX_ZC) + zx] = pc3;

    wseq[zx] = pc[(0 * MAX_ZC) + zx];
    wseq[1 * Zc + zx] = pc1;
    wseq[2 * Zc + zx] = pc2;
    wseq[3 * Zc + zx] = pc3;

    __syncthreads();

    for (uint16_t HBG_row = 4u; HBG_row < num_rows; HBG_row++) {
        uint32_t offset = HBG_offset_array[HBG_row];
        uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
        const uint16_t *HBGcolumns = HBG_array + offset;
        const uint16_t *Vij = Vij_array + offset;

        uint8_t ps = 0;
        for (uint32_t j = 0; j < degree; j++) {
            uint16_t HBG_col_idx = HBGcolumns[j];
            if (HBG_col_idx < Kb + 4u) {
                uint16_t shift = Vij[j];
                uint16_t column = (uint16_t)(HBG_col_idx * Zc +
                    ((shift + zx) % Zc));
                ps ^= (uint8_t)(out_bits[column] & 1u);
            }
        }

        wseq[Zc * HBG_row + zx] = ps;
    }
}

static int32_t nr_ldpc_encode_device_input(const uint8_t *device_in_bits,
    uint8_t *host_out_bits, uint8_t *device_out_bits,
    ldpc_config_t *ldpc_config) {
    uint16_t bgn = ldpc_config->bgn;
    uint16_t Zc = ldpc_config->Zc;
    uint16_t K = ldpc_config->K;
    uint16_t B = ldpc_config->B;
    uint16_t Kb = K / Zc; //22 for bg1, 10 for bg2
    uint16_t *Vij_array = ldpc_config->Vij_array;
    uint16_t *HBG_array = ldpc_config->HBG_array;
    uint16_t *HBG_offset_array = ldpc_config->HBG_offset_array;
    uint16_t num_rows = (bgn == 1) ? BG1_NUM_ROWS : BG2_NUM_ROWS;
    uint16_t non_zero_elements = (bgn == 1) ?
        BG1_MAX_NON_NULL_ELEMENTS : BG2_MAX_NON_NULL_ELEMENTS;
    uint32_t out_count = (uint32_t)K + (uint32_t)num_rows * (uint32_t)Zc;
    uint32_t row_offsets = (uint32_t)num_rows + 1u;
    cuda_encoder_workspace_t *ws = &g_encoder_workspace;

    if (cuda_encoder_ensure_workspace(ws, B,
            device_out_bits == NULL ? out_count : 0u, non_zero_elements,
            row_offsets) != 0) {
        return -1;
    }
    uint8_t *encoder_out_bits = device_out_bits != NULL ?
        device_out_bits : ws->out_bits;

    if (ws->host_Vij_array != Vij_array || ws->cached_bgn != bgn ||
            ws->cached_Zc != Zc || ws->cached_iLS != ldpc_config->iLS) {
        CUDA_CHECK_ENCODE(cudaMemcpy(ws->Vij_array, Vij_array,
            sizeof(uint16_t) * non_zero_elements, cudaMemcpyHostToDevice));
        ws->host_Vij_array = Vij_array;
    }
    if (ws->host_HBG_array != HBG_array || ws->cached_bgn != bgn) {
        CUDA_CHECK_ENCODE(cudaMemcpy(ws->HBG_array, HBG_array,
            sizeof(uint16_t) * non_zero_elements, cudaMemcpyHostToDevice));
        ws->host_HBG_array = HBG_array;
    }
    if (ws->host_HBG_offset_array != HBG_offset_array ||
            ws->cached_bgn != bgn) {
        CUDA_CHECK_ENCODE(cudaMemcpy(ws->HBG_offset_array, HBG_offset_array,
            sizeof(uint16_t) * row_offsets, cudaMemcpyHostToDevice));
        ws->host_HBG_offset_array = HBG_offset_array;
    }
    ws->cached_bgn = bgn;
    ws->cached_Zc = Zc;
    ws->cached_iLS = ldpc_config->iLS;

    CUDA_CHECK_ENCODE(cudaMemset(ws->pc, 0, sizeof(uint8_t) * 4u * MAX_ZC));

    if (encoder_out_bits != device_in_bits) {
        CUDA_CHECK_ENCODE(cudaMemcpy(encoder_out_bits, device_in_bits,
            sizeof(uint8_t) * B, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK_ENCODE(cudaMemset(encoder_out_bits + B, 0,
        sizeof(uint8_t) * (K - B)));

    cuda_encoder_parity_kernel<<<1, Zc>>>(
        encoder_out_bits, ws->pc, ws->Vij_array, ws->HBG_array,
        ws->HBG_offset_array,
        bgn, num_rows, Zc, K, Kb);
    CUDA_CHECK_ENCODE(cudaGetLastError());

    if (host_out_bits != NULL) {
        CUDA_CHECK_ENCODE(cudaMemcpy(host_out_bits, encoder_out_bits,
            sizeof(uint8_t) * out_count, cudaMemcpyDeviceToHost));
    }

    return 0;
}

int32_t cuda_nr_ldpc_encode_to_device(const uint8_t *device_in_bits,
    uint8_t *device_out_bits, ldpc_config_t *ldpc_config) {
    if (device_in_bits == NULL || device_out_bits == NULL ||
            ldpc_config == NULL) {
        return -1;
    }

    return nr_ldpc_encode_device_input(device_in_bits, NULL, device_out_bits,
        ldpc_config);
}
