#include <stdio.h>
#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

#define CUDA_CHECK_RATE_MATCH_INT(call)                                      \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA rate-match error %s:%d: %s\n", __FILE__,   \
                    __LINE__, cudaGetErrorString(cuda_check_status));        \
            return -1;                                                       \
        }                                                                    \
    } while (0)

typedef struct {
    uint8_t *outbytes;
    uint32_t outbytes_capacity;
} cuda_rate_match_workspace_t;

static cuda_rate_match_workspace_t g_rate_match_workspace;

static int cuda_rate_match_ensure_workspace(
    cuda_rate_match_workspace_t *ws,
    uint32_t out_bytes)
{
    if (ws->outbytes_capacity < out_bytes) {
        cudaFree(ws->outbytes);
        ws->outbytes = NULL;
        CUDA_CHECK_RATE_MATCH_INT(cudaMalloc((void **)&ws->outbytes,
            sizeof(uint8_t) * out_bytes));
        ws->outbytes_capacity = out_bytes;
    }

    return 0;
}

static __global__ void cuda_rate_match_pack_kernel(
    const uint8_t *inbits,
    uint8_t *outbytes,
    uint16_t E,
    uint16_t compact_N,
    uint16_t compact_k0,
    uint16_t filler_start,
    uint16_t Zc,
    uint16_t F,
    uint16_t Qm)
{
    uint32_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_bytes = (E + 7u) / 8u;
    if (byte_idx >= num_bytes) {
        return;
    }

    uint8_t out = 0;
    uint16_t symbols = (uint16_t)(E / Qm);

    for (uint32_t bit = 0; bit < 8u; bit++) {
        uint32_t stream_idx = byte_idx * 8u + bit;
        out = (uint8_t)(out << 1);
        if (stream_idx < E) {
            uint16_t j = (uint16_t)(stream_idx / Qm);
            uint16_t i = (uint16_t)(stream_idx % Qm);
            uint16_t selected_idx = (uint16_t)(i * symbols + j);
            uint32_t compact_pos = (uint32_t)compact_k0 + selected_idx;
            if (compact_pos >= compact_N) {
                compact_pos %= compact_N;
            }

            // Compact buffer layout:
            //   [inbits[2*Zc:B]][inbits[K:N + 2*Zc]]
            // Positions after the systematic non-filler span map past the
            // filler bits in the original encoder output.
            uint32_t original_pos = compact_pos;
            if (compact_pos >= filler_start) {
                original_pos = compact_pos + F;
            }

            out |= (uint8_t)(inbits[2 * Zc + original_pos] & 1u);
        }
    }
    outbytes[byte_idx] = out;
}

int32_t cuda_nr_ldpc_rate_match_from_device(const uint8_t *device_inbits,
    uint8_t *outbytes, ldpc_config_t *config)
{
    if (device_inbits == NULL || outbytes == NULL || config == NULL) {
        return -1;
    }

    uint16_t E = config->E;
    uint16_t N = config->N;
    uint16_t k0 = config->k0;
    uint16_t Zc = config->Zc;
    uint16_t B = config->B;
    uint16_t Qm = config->Qm;
    uint16_t K = config->K;
    uint16_t F = config->F;
    uint16_t compact_N = (uint16_t)(N - F);
    uint16_t filler_start = (uint16_t)(B - 2 * Zc);
    uint16_t filler_end = (uint16_t)(K - 2 * Zc);
    uint16_t compact_k0;
    uint32_t out_bytes = ((uint32_t)E + 7u) / 8u;
    cuda_rate_match_workspace_t *ws = &g_rate_match_workspace;

    if (k0 < filler_start) {
        compact_k0 = k0;
    } else if (k0 < filler_end) {
        compact_k0 = filler_start;
    } else {
        compact_k0 = (uint16_t)(k0 - F);
    }

    if (cuda_rate_match_ensure_workspace(ws, out_bytes) != 0) {
        return -1;
    }

    const int threads = 128;
    cuda_rate_match_pack_kernel<<<(out_bytes + threads - 1) / threads, threads>>>(
        device_inbits, ws->outbytes, E, compact_N, compact_k0, filler_start,
        Zc, F, Qm);
    CUDA_CHECK_RATE_MATCH_INT(cudaGetLastError());

    CUDA_CHECK_RATE_MATCH_INT(cudaMemcpy(outbytes, ws->outbytes, sizeof(uint8_t) * out_bytes,
        cudaMemcpyDeviceToHost));

    return 0;
}
