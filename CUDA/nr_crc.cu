#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "nr_crc_internal.h"
#include "nr_ldpc.h"

#define CUDA_CRC_WORD_BYTES 4u
#define CUDA_CRC_WORD_TABLE_ENTRIES (CUDA_CRC_WORD_BYTES * CRC_TABLE_SIZE)
#define CUDA_CRC_TABLE_ENTRIES (CUDA_CRC_WORD_TABLE_ENTRIES + CRC_TABLE_SIZE)

// Reused device allocations for the single-block CRC kernel.
typedef struct {
    uint8_t *device_input;
    uint32_t *device_tables;
    uint32_t *device_crc;
    uint32_t input_capacity;
    uint32_t crc_capacity;
    uint32_t table_capacity;
    int table_crctype;
} cuda_crc_workspace_t;

static cuda_crc_workspace_t g_cuda_crc_workspace;

// Convert CUDA runtime failures into the integer status convention used by the
// surrounding C/CUDA encoder code.
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

// Grow workspace allocations lazily. Existing buffers are reused across calls
// to avoid paying cudaMalloc/cudaFree for every CRC attachment.
static int cuda_crc_ensure_workspace(cuda_crc_workspace_t *ws,
    uint32_t input_bytes, uint32_t crc_count) {
    if (ws->input_capacity < input_bytes) {
        cudaFree(ws->device_input);
        ws->device_input = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->device_input,
                sizeof(uint8_t) * input_bytes));
        ws->input_capacity = input_bytes;
    }

    if (ws->crc_capacity < crc_count) {
        cudaFree(ws->device_crc);
        ws->device_crc = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->device_crc,
                sizeof(uint32_t) * crc_count));
        ws->crc_capacity = crc_count;
    }

    return 0;
}

// Copy the selected CRC tables to one contiguous device buffer:
//   [4-byte contribution table][byte-wise lookup table]
//
// Tables are stable constexpr host data, so the workspace only refreshes the
// device copy when the requested CRC type changes.
static int cuda_crc_copy_tables(cuda_crc_workspace_t *ws,
    const crc_spec_t *spec, int crctype) {
    if (ws->table_capacity < CUDA_CRC_TABLE_ENTRIES) {
        cudaFree(ws->device_tables);
        ws->device_tables = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->device_tables,
                sizeof(uint32_t) * CUDA_CRC_TABLE_ENTRIES));
        ws->table_capacity = CUDA_CRC_TABLE_ENTRIES;
        ws->table_crctype = -1;
    }

    if (ws->table_crctype == crctype) {
        return 0;
    }

    CUDA_CRC_CHECK(cudaMemcpy(ws->device_tables,
            spec->word_contribution_table,
            sizeof(uint32_t) * CUDA_CRC_WORD_TABLE_ENTRIES,
            cudaMemcpyHostToDevice));
    CUDA_CRC_CHECK(cudaMemcpy(
            ws->device_tables + CUDA_CRC_WORD_TABLE_ENTRIES,
            spec->table, sizeof(uint32_t) * CRC_TABLE_SIZE,
            cudaMemcpyHostToDevice));
    ws->table_crctype = crctype;

    return 0;
}

// Update the CRC by one big-endian 32-bit payload word.
//
// For CRC24, the three active CRC bytes are folded into the first three data
// bytes before indexing the precomputed contribution table. For CRC16, only
// two CRC bytes exist, so d2 and d3 are pure data contributions.
static __device__ uint32_t cuda_crc_update32(uint32_t crc, uint32_t word,
    const uint32_t *word_table, uint32_t width, uint32_t mask) {
    uint8_t d0 = (uint8_t)((word >> 24) & 0xFFu);
    uint8_t d1 = (uint8_t)((word >> 16) & 0xFFu);
    uint8_t d2 = (uint8_t)((word >> 8) & 0xFFu);
    uint8_t d3 = (uint8_t)(word & 0xFFu);

    uint8_t c0 = (uint8_t)((crc >> (width - 8u)) & 0xFFu);
    uint8_t c1 = (uint8_t)((crc >> (width - 16u)) & 0xFFu);
    uint8_t c2 = (uint8_t)((width == CRC24_WIDTH) ? (crc & 0xFFu) : 0u);

    uint32_t next_crc =
        word_table[3u * CRC_TABLE_SIZE + (c0 ^ d0)] ^
        word_table[2u * CRC_TABLE_SIZE + (c1 ^ d1)];

    next_crc ^= word_table[1u * CRC_TABLE_SIZE +
        ((width == CRC24_WIDTH) ? (c2 ^ d2) : d2)];
    next_crc ^= word_table[d3];

    return next_crc & mask;
}

// Byte-wise update for the 0..3 trailing input bytes after all complete words.
static __device__ uint32_t cuda_crc_update8(uint32_t crc, uint8_t value,
    const uint32_t *crc_table, uint32_t width, uint32_t mask) {
    uint32_t shift = width - 8u;
    uint8_t index = (uint8_t)(((crc >> shift) ^ value) & 0xFFu);
    return ((crc << 8) ^ crc_table[index]) & mask;
}

static __device__ void cuda_crc_write_byte_bits(uint8_t *out_bits,
    uint32_t bit_offset, uint8_t value) {
    for (uint32_t bit = 0; bit < 8u; bit++) {
        uint8_t bit_pos = (uint8_t)(7u - bit);
        out_bits[bit_offset + bit] = (uint8_t)((value >> bit_pos) & 0x01u);
    }
}

// One-thread CRC encoder that outputs one byte per bit.
//
// The payload still arrives as packed host bytes, but the LDPC encoder's hot
// loops consume individual bits. This kernel writes payload bits and CRC bits
// directly as uint8_t values 0 or 1, avoiding a separate unpack kernel.
static __global__ void cuda_crc_encode_bits_kernel(const uint8_t *input_bytes,
    uint8_t *out_bits, uint32_t *out_crc, const uint32_t *tables,
    uint32_t input_len, uint32_t crc_len_bits, uint32_t width,
    uint32_t mask) {
    const uint32_t *word_table = tables;
    const uint32_t *crc_table = tables + CUDA_CRC_WORD_TABLE_ENTRIES;
    uint32_t crc = 0;
    uint32_t byte_idx = 0;

    for (; byte_idx + CUDA_CRC_WORD_BYTES <= input_len;
            byte_idx += CUDA_CRC_WORD_BYTES) {
        uint8_t b0 = input_bytes[byte_idx];
        uint8_t b1 = input_bytes[byte_idx + 1u];
        uint8_t b2 = input_bytes[byte_idx + 2u];
        uint8_t b3 = input_bytes[byte_idx + 3u];
        uint32_t word = ((uint32_t)b0 << 24) | ((uint32_t)b1 << 16) |
            ((uint32_t)b2 << 8) | (uint32_t)b3;

        cuda_crc_write_byte_bits(out_bits, byte_idx * 8u, b0);
        cuda_crc_write_byte_bits(out_bits, (byte_idx + 1u) * 8u, b1);
        cuda_crc_write_byte_bits(out_bits, (byte_idx + 2u) * 8u, b2);
        cuda_crc_write_byte_bits(out_bits, (byte_idx + 3u) * 8u, b3);
        crc = cuda_crc_update32(crc, word, word_table, width, mask);
    }

    for (; byte_idx < input_len; byte_idx++) {
        uint8_t value = input_bytes[byte_idx];
        cuda_crc_write_byte_bits(out_bits, byte_idx * 8u, value);
        crc = cuda_crc_update8(crc, value, crc_table, width, mask);
    }

    uint32_t crc_bit_offset = input_len * 8u;
    for (uint32_t bit = 0; bit < crc_len_bits; bit++) {
        uint32_t crc_bit_pos = crc_len_bits - 1u - bit;
        out_bits[crc_bit_offset + bit] =
            (uint8_t)((crc >> crc_bit_pos) & 0x01u);
    }

    if (out_crc != NULL) {
        *out_crc = crc;
    }
}

// Attach an NR CRC and write payload+CRC bits into caller-owned device memory.
//
// Each output bit is stored as one uint8_t value, 0 or 1. This representation
// matches the LDPC encoder kernels, so the top-level encoder can skip the
// packed-byte-to-bit unpack step.
//
// The caller owns device_out_bits and must allocate at least B bytes before
// calling. The input payload still starts in host memory and is copied into the
// CRC workspace's device_input buffer before the kernel runs.
//
// Optional output:
//   crc: host copy of the integer CRC remainder when non-NULL.
int32_t cuda_crc_encode_bits_to_device(const uint8_t *input_bytes,
    uint8_t *device_out_bits, uint32_t *crc, uint16_t A, uint16_t B,
    int crctype) {
    if (input_bytes == NULL || device_out_bits == NULL || (A % 8u) != 0 ||
            (B % 8u) != 0 || B < A || B > MAX_ZC * 22u ||
            crctype < CRC16_TYPE || crctype > CRC24C_TYPE) {
        return -1;
    }

    uint32_t input_len = (uint32_t)A / 8u;
    uint32_t crc_len_bits = (uint32_t)B - (uint32_t)A;

    crc_spec_t spec;
    if (!cuda_crc_get_spec(crctype, &spec)) {
        return -1;
    }

    cuda_crc_workspace_t *ws = &g_cuda_crc_workspace;
    if (cuda_crc_ensure_workspace(ws, input_len, 1u) != 0) {
        return -1;
    }
    if (cuda_crc_copy_tables(ws, &spec, crctype) != 0) {
        return -1;
    }

    CUDA_CRC_CHECK(cudaMemcpy(ws->device_input, input_bytes,
            sizeof(uint8_t) * input_len, cudaMemcpyHostToDevice));

    cuda_crc_encode_bits_kernel<<<1, 1>>>(ws->device_input, device_out_bits,
            ws->device_crc, ws->device_tables, input_len, crc_len_bits,
            spec.width, spec.mask);
    CUDA_CRC_CHECK(cudaGetLastError());

    if (crc != NULL) {
        CUDA_CRC_CHECK(cudaMemcpy(crc, ws->device_crc, sizeof(uint32_t),
                cudaMemcpyDeviceToHost));
    }

    return 0;
}
