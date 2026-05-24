#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

#define CUDA_CHECK_RATE_RECOVER_INT(call)                                    \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA rate-recover error %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(cuda_check_status));        \
            return -1;                                                       \
        }                                                                    \
    } while (0)

typedef struct {
    int16_t *input_llr;
    int16_t *harq_memory;
    uint32_t input_capacity;
    uint32_t harq_capacity;
} cuda_rate_recover_workspace_t;

static cuda_rate_recover_workspace_t g_rate_recover_workspace;

static int cuda_rate_recover_ensure_workspace(
    cuda_rate_recover_workspace_t *ws,
    uint32_t E,
    uint32_t harq_len)
{
    if (ws->input_capacity < E) {
        cudaFree(ws->input_llr);
        ws->input_llr = NULL;
        CUDA_CHECK_RATE_RECOVER_INT(cudaMalloc((void **)&ws->input_llr,
            sizeof(int16_t) * E));
        ws->input_capacity = E;
    }

    if (ws->harq_capacity < harq_len) {
        cudaFree(ws->harq_memory);
        ws->harq_memory = NULL;
        CUDA_CHECK_RATE_RECOVER_INT(cudaMalloc((void **)&ws->harq_memory,
            sizeof(int16_t) * harq_len));
        ws->harq_capacity = harq_len;
    }

    return 0;
}

static __global__ void cuda_rate_recover_kernel(
    int16_t *harq_memory,
    const int16_t *input_llr,
    uint16_t E,
    uint16_t N,
    uint16_t k0,
    uint16_t Zc,
    uint16_t B,
    uint16_t K,
    uint16_t F,
    int16_t filling_bit_value)
{
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= N) {
        return;
    }

    if (pos < F) {
        harq_memory[B + pos] = filling_bit_value;
    }

    uint32_t filler_start = (uint32_t)B - 2u * (uint32_t)Zc;
    uint32_t filler_end = (uint32_t)K - 2u * (uint32_t)Zc;
    if (pos >= filler_start && pos < filler_end) {
        return;
    }

    uint32_t compact_N = (uint32_t)N - (uint32_t)F;
    uint32_t compact_pos = pos;
    if (pos >= filler_end) {
        compact_pos -= F;
    }

    uint32_t compact_k0;
    if (k0 < filler_start) {
        compact_k0 = k0;
    } else if (k0 < filler_end) {
        compact_k0 = filler_start;
    } else {
        compact_k0 = (uint32_t)k0 - (uint32_t)F;
    }

    uint32_t selected_idx = (compact_pos + compact_N - compact_k0) % compact_N;
    int32_t llr_sum = 0;
    for (uint32_t k = selected_idx; k < E; k += compact_N) {
        llr_sum += input_llr[k];
    }

    harq_memory[pos + 2u * (uint32_t)Zc] =
        (int16_t)((int32_t)harq_memory[pos + 2u * (uint32_t)Zc] + llr_sum);
}

int32_t cuda_nr_ldpc_rate_recover_harq_combine(
    ldpc_config_t *config,
    int16_t filling_bit_value,
    int16_t *input_LLr)
{
    if (config == NULL || input_LLr == NULL) {
        return -1;
    }

    uint16_t E = config->E;
    uint16_t N = config->N;
    uint16_t Zc = config->Zc;
    uint32_t harq_len = (uint32_t)N + 2u * (uint32_t)Zc;
    cuda_rate_recover_workspace_t *ws = &g_rate_recover_workspace;

    if (cuda_rate_recover_ensure_workspace(ws, E, harq_len) != 0) {
        return -1;
    }

    CUDA_CHECK_RATE_RECOVER_INT(cudaMemcpy(ws->input_llr, input_LLr,
        sizeof(int16_t) * E, cudaMemcpyHostToDevice));

    if (config->HARQ_on != 0) {
        CUDA_CHECK_RATE_RECOVER_INT(cudaMemcpy(ws->harq_memory, HARQ_memory,
            sizeof(int16_t) * harq_len, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK_RATE_RECOVER_INT(cudaMemset(ws->harq_memory, 0,
            sizeof(int16_t) * harq_len));
    }

    const int threads = 256;
    cuda_rate_recover_kernel<<<(N + threads - 1) / threads, threads>>>(
        ws->harq_memory,
        ws->input_llr,
        E,
        N,
        config->k0,
        Zc,
        config->B,
        config->K,
        config->F,
        filling_bit_value);
    CUDA_CHECK_RATE_RECOVER_INT(cudaGetLastError());

    CUDA_CHECK_RATE_RECOVER_INT(cudaMemcpy(HARQ_memory, ws->harq_memory,
        sizeof(int16_t) * harq_len, cudaMemcpyDeviceToHost));

    return 0;
}
