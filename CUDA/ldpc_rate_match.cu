#include <stdio.h>
#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

#define CUDA_CHECK_RATE_MATCH(call)                                          \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA rate-match error %s:%d: %s\n", __FILE__,   \
                    __LINE__, cudaGetErrorString(cuda_check_status));        \
            return;                                                          \
        }                                                                    \
    } while (0)

static __global__ void cuda_rate_match_select_kernel(
    const uint8_t *inbits,
    uint8_t *ek,
    uint16_t E,
    uint16_t N,
    uint16_t k0,
    uint16_t Zc,
    uint16_t B,
    uint16_t K)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const uint8_t *rm_in = inbits + 2 * Zc;
    uint16_t k = 0;
    uint16_t j = 0;
    while (k < E) {
        uint16_t pos = (uint16_t)((k0 + j) % N);
        if (pos < B - 2 * Zc || pos >= K - 2 * Zc) {
            ek[k++] = rm_in[pos];
        }
        j++;
    }
}

static __global__ void cuda_rate_match_pack_kernel(
    const uint8_t *ek,
    uint8_t *outbytes,
    uint16_t E,
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
            out |= (uint8_t)(ek[i * symbols + j] & 1u);
        }
    }
    outbytes[byte_idx] = out;
}

/*5g ldpc rate matching
dn: input sequence with size = N
ek: rat matching output with size = E/8 bytes
*/
void nr_ldpc_rate_match(uint8_t *inbits, uint8_t *outbytes,ldpc_config_t *config) {
    //read parameters from config
    uint16_t E = config->E;
    uint16_t N = config->N;
    uint16_t k0 = config->k0;
    uint16_t Zc = config->Zc;
    uint16_t B = config->B;
    uint16_t Qm = config->Qm;
    uint16_t K = config->K;
    uint32_t in_count = (uint32_t)N + 2u * (uint32_t)Zc;
    uint32_t out_bytes = ((uint32_t)E + 7u) / 8u;

    uint8_t *d_inbits = NULL;
    uint8_t *d_ek = NULL;
    uint8_t *d_outbytes = NULL;

    CUDA_CHECK_RATE_MATCH(cudaMalloc((void **)&d_inbits, sizeof(uint8_t) * in_count));
    CUDA_CHECK_RATE_MATCH(cudaMalloc((void **)&d_ek, sizeof(uint8_t) * E));
    CUDA_CHECK_RATE_MATCH(cudaMalloc((void **)&d_outbytes, sizeof(uint8_t) * out_bytes));

    CUDA_CHECK_RATE_MATCH(cudaMemcpy(d_inbits, inbits, sizeof(uint8_t) * in_count,
        cudaMemcpyHostToDevice));

    cuda_rate_match_select_kernel<<<1, 1>>>(
        d_inbits, d_ek, E, N, k0, Zc, B, K);
    CUDA_CHECK_RATE_MATCH(cudaGetLastError());

    const int threads = 128;
    cuda_rate_match_pack_kernel<<<(out_bytes + threads - 1) / threads, threads>>>(
        d_ek, d_outbytes, E, Qm);
    CUDA_CHECK_RATE_MATCH(cudaGetLastError());

    CUDA_CHECK_RATE_MATCH(cudaMemcpy(outbytes, d_outbytes, sizeof(uint8_t) * out_bytes,
        cudaMemcpyDeviceToHost));

    cudaFree(d_outbytes);
    cudaFree(d_ek);
    cudaFree(d_inbits);

}
