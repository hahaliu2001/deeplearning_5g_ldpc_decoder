#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "nr_crc_internal.h"
#include "nr_ldpc.h"

#define CUDA_CRC_THREADS 4
#define CUDA_CRC_WORD_TABLE_ENTRIES (4u * CRC_TABLE_SIZE)
#define CUDA_CRC_DEVICE_TABLE_ENTRIES \
    (CUDA_CRC_WORD_TABLE_ENTRIES + CRC_TABLE_SIZE)

typedef struct {
    uint8_t *input_bytes;
    uint8_t *out_bits;
    uint32_t *crc_contribution_table;
    uint32_t input_capacity;
    uint32_t out_capacity;
    uint32_t crc_contribution_capacity;
} cuda_crc_workspace_t;

static cuda_crc_workspace_t g_cuda_crc_workspace;

static int cuda_crc_check(cudaError_t status, const char *file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA CRC error %s:%d: %s\n", file, line,
                cudaGetErrorString(status));
        return -1;
    }
    return 0;
}

#define CUDA_CRC_CHECK(call)                                                 \
    do {                                                                     \
        if (cuda_crc_check((call), __FILE__, __LINE__) != 0) {               \
            return -1;                                                       \
        }                                                                    \
    } while (0)

static int cuda_crc_ensure_workspace(cuda_crc_workspace_t *ws,
    uint32_t input_bytes, uint32_t out_bits) {
    if (ws->input_capacity < input_bytes) {
        cudaFree(ws->input_bytes);
        ws->input_bytes = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->input_bytes,
                sizeof(uint8_t) * input_bytes));
        ws->input_capacity = input_bytes;
    }

    if (ws->out_capacity < out_bits) {
        cudaFree(ws->out_bits);
        ws->out_bits = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->out_bits,
                sizeof(uint8_t) * out_bits));
        ws->out_capacity = out_bits;
    }

    return 0;
}

static int cuda_crc_copy_tables(cuda_crc_workspace_t *ws,
    const crc_spec_t *spec) {
    if (ws->crc_contribution_capacity < CUDA_CRC_DEVICE_TABLE_ENTRIES) {
        cudaFree(ws->crc_contribution_table);
        ws->crc_contribution_table = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->crc_contribution_table,
                sizeof(uint32_t) * CUDA_CRC_DEVICE_TABLE_ENTRIES));
        ws->crc_contribution_capacity = CUDA_CRC_DEVICE_TABLE_ENTRIES;
    }

    CUDA_CRC_CHECK(cudaMemcpy(ws->crc_contribution_table,
            spec->word_contribution_table,
            sizeof(uint32_t) * CUDA_CRC_WORD_TABLE_ENTRIES,
            cudaMemcpyHostToDevice));
    CUDA_CRC_CHECK(cudaMemcpy(
            ws->crc_contribution_table + CUDA_CRC_WORD_TABLE_ENTRIES,
            spec->table, sizeof(uint32_t) * CRC_TABLE_SIZE,
            cudaMemcpyHostToDevice));

    return 0;
}

static __device__ uint32_t cuda_crc_advance_zero_bytes(uint32_t crc,
    uint32_t zero_bytes, const uint32_t *crc_table, uint32_t shift,
    uint32_t mask) {
    for (uint32_t byte_idx = 0; byte_idx < zero_bytes; byte_idx++) {
        uint8_t index = (uint8_t)((crc >> shift) & 0xFFu);
        crc = ((crc << 8) ^ crc_table[index]) & mask;
    }

    return crc & mask;
}

static __device__ void cuda_append_crc_bits(uint8_t *out_bits, uint32_t A,
    uint32_t B, uint32_t crc) {
    uint32_t crc_len = B - A;
    uint32_t first_bit_pos = crc_len - 1u;

    for (uint32_t i = 0; i < crc_len; i++) {
        uint32_t bit_pos = first_bit_pos - i;
        out_bits[A + i] = (uint8_t)((crc >> bit_pos) & 1u);
    }
}

static __global__ void cuda_crc_encode_shared_kernel(const uint8_t *input_bytes,
    uint8_t *out_bits, const uint32_t *crc_contribution_table, uint32_t A,
    uint32_t B, uint32_t input_len, uint32_t crc_shift, uint32_t mask,
    uint32_t max_input_bytes) {
    extern __shared__ uint8_t shared_mem[];
    uint8_t *shared_input_bytes = shared_mem;
    uint32_t *shared_word_table =
        (uint32_t *)(shared_input_bytes + max_input_bytes);
    uint32_t *shared_crc_table =
        shared_word_table + CUDA_CRC_WORD_TABLE_ENTRIES;
    uint32_t *shared_lane_crc =
        shared_crc_table + CRC_TABLE_SIZE;
    uint8_t *shared_bits = (uint8_t *)(shared_lane_crc + CUDA_CRC_THREADS);
    uint32_t word_count = (input_len + 3u) / 4u;

    for (uint32_t i = threadIdx.x; i < input_len; i += blockDim.x) {
        shared_input_bytes[i] = input_bytes[i];
    }

    for (uint32_t i = threadIdx.x; i < CUDA_CRC_DEVICE_TABLE_ENTRIES;
            i += blockDim.x) {
        shared_word_table[i] = crc_contribution_table[i];
    }
    __syncthreads();

    uint32_t lane = threadIdx.x;
    uint32_t lane_crc = 0;
    for (uint32_t word_idx = 0; word_idx < word_count; word_idx++) {
        uint32_t byte_idx = word_idx * 4u;
        uint32_t chunk_len = input_len - byte_idx;
        if (chunk_len > 4u) {
            chunk_len = 4u;
        }

        lane_crc = cuda_crc_advance_zero_bytes(lane_crc, chunk_len,
            shared_crc_table, crc_shift, mask);

        if (lane < chunk_len) {
            uint8_t value = shared_input_bytes[byte_idx + lane];
            uint32_t remaining = chunk_len - 1u - lane;
            lane_crc ^= shared_word_table[remaining * CRC_TABLE_SIZE + value];
        }
    }

    shared_lane_crc[lane] = lane_crc & mask;
    __syncthreads();

    if (threadIdx.x == 0) {
        shared_lane_crc[0] = (shared_lane_crc[0] ^ shared_lane_crc[1] ^
            shared_lane_crc[2] ^ shared_lane_crc[3]) & mask;
    }
    __syncthreads();

    for (uint32_t bit_idx = threadIdx.x; bit_idx < A; bit_idx += blockDim.x) {
        uint32_t byte_idx = bit_idx / 8u;
        uint8_t bit_pos = (uint8_t)(7u - (bit_idx % 8u));
        shared_bits[bit_idx] =
            (uint8_t)((shared_input_bytes[byte_idx] >> bit_pos) & 1u);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        cuda_append_crc_bits(shared_bits, A, B, shared_lane_crc[0]);
    }
    __syncthreads();

    for (uint32_t bit_idx = threadIdx.x; bit_idx < B; bit_idx += blockDim.x) {
        out_bits[bit_idx] = shared_bits[bit_idx];
    }
}

int32_t cuda_crc_encode_bits_to_device(const uint8_t *input_bytes,
    uint8_t **device_out_bits, uint16_t A, uint16_t B, int crctype) {
    if (input_bytes == NULL || device_out_bits == NULL || (A % 8u) != 0 ||
            B < A || B > MAX_ZC * 22u || crctype < CRC16_TYPE ||
            crctype > CRC24C_TYPE) {
        return -1;
    }

    uint32_t input_len = (uint32_t)A / 8u;
    crc_spec_t spec;
    if (!cuda_crc_get_spec(crctype, &spec)) {
        return -1;
    }

    cuda_crc_workspace_t *ws = &g_cuda_crc_workspace;
    if (cuda_crc_ensure_workspace(ws, input_len, B) != 0) {
        return -1;
    }

    uint32_t max_input_bytes = (MAX_ZC * 22u) / 8u;
    if (cuda_crc_copy_tables(ws, &spec) != 0) {
        return -1;
    }

    CUDA_CRC_CHECK(cudaMemcpy(ws->input_bytes, input_bytes,
            sizeof(uint8_t) * input_len, cudaMemcpyHostToDevice));

    size_t shared_bytes = max_input_bytes +
        CUDA_CRC_DEVICE_TABLE_ENTRIES * sizeof(uint32_t) +
        CUDA_CRC_THREADS * sizeof(uint32_t) + (size_t)MAX_ZC * 22u;
    cuda_crc_encode_shared_kernel<<<1, CUDA_CRC_THREADS, shared_bytes>>>(
            ws->input_bytes, ws->out_bits, ws->crc_contribution_table, A, B,
            input_len, spec.width - 8u, spec.mask, max_input_bytes);

    CUDA_CRC_CHECK(cudaGetLastError());
    *device_out_bits = ws->out_bits;

    return 0;
}

int32_t cuda_crc_encode_bits(const uint8_t *input_bytes, uint8_t *out_bits,
    uint16_t A, uint16_t B, int crctype) {
    if (input_bytes == NULL || out_bits == NULL || (A % 8u) != 0 ||
            B < A || crctype < CRC16_TYPE || crctype > CRC24C_TYPE) {
        return -1;
    }

    uint8_t *device_out_bits = NULL;
    if (cuda_crc_encode_bits_to_device(input_bytes, &device_out_bits, A, B,
            crctype) != 0) {
        return -1;
    }

    CUDA_CRC_CHECK(cudaMemcpy(out_bits, device_out_bits, sizeof(uint8_t) * B,
            cudaMemcpyDeviceToHost));

    return 0;
}
