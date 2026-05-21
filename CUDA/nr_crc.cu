#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "nr_crc_internal.h"
#include "nr_ldpc.h"

#define CUDA_CRC_WORD_BYTES 4u
#define CUDA_CRC_WORD_TABLE_ENTRIES (CUDA_CRC_WORD_BYTES * CRC_TABLE_SIZE)
#define CUDA_CRC_TABLE_ENTRIES (CUDA_CRC_WORD_TABLE_ENTRIES + CRC_TABLE_SIZE)

// Reused device allocations for the single-block CRC kernel.
//
// The output buffer is owned by this workspace. Callers that request
// device_out_bytes receive this pointer and must treat it as valid until the
// next cuda_crc_encode_bytes() call that grows or overwrites the workspace.
typedef struct {
    uint8_t *device_input;
    uint8_t *device_output;
    uint32_t *device_tables;
    uint32_t *device_crc;
    uint32_t input_capacity;
    uint32_t output_capacity;
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

static int cuda_crc_validate_request(const uint8_t *input_bytes,
    const uint8_t *out_bytes, uint8_t **device_out_bytes, const uint32_t *crc,
    uint16_t A, uint16_t B, int crctype) {
    if (input_bytes == NULL) {
        return -1;
    }
    if (out_bytes == NULL && device_out_bytes == NULL && crc == NULL) {
        return -1;
    }
    if ((A % 8u) != 0 || (B % 8u) != 0 || B < A || B > MAX_ZC * 22u) {
        return -1;
    }
    if (crctype < CRC16_TYPE || crctype > CRC24C_TYPE) {
        return -1;
    }
    return 0;
}

// Grow workspace allocations lazily. Existing buffers are reused across calls
// to avoid paying cudaMalloc/cudaFree for every CRC attachment.
static int cuda_crc_ensure_workspace(cuda_crc_workspace_t *ws,
    uint32_t input_bytes, uint32_t output_bytes, uint32_t crc_count) {
    if (ws->input_capacity < input_bytes) {
        cudaFree(ws->device_input);
        ws->device_input = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->device_input,
                sizeof(uint8_t) * input_bytes));
        ws->input_capacity = input_bytes;
    }

    if (ws->output_capacity < output_bytes) {
        cudaFree(ws->device_output);
        ws->device_output = NULL;
        CUDA_CRC_CHECK(cudaMalloc((void **)&ws->device_output,
                sizeof(uint8_t) * output_bytes));
        ws->output_capacity = output_bytes;
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

// Append the CRC remainder MSB-first as whole bytes.
//
// This path assumes A and B are byte-aligned, which is validated by the public
// entry point before the kernel is launched.
static __device__ void cuda_crc_append_bytes(uint8_t *out_bytes,
    uint32_t input_len, uint32_t crc_len_bytes, uint32_t crc) {
    for (uint32_t i = 0; i < crc_len_bytes; i++) {
        uint32_t shift = (crc_len_bytes - 1u - i) * 8u;
        out_bytes[input_len + i] = (uint8_t)((crc >> shift) & 0xFFu);
    }
}

// One-thread CRC encoder.
//
// The LDPC encoder consumes byte-packed input, so this kernel copies the
// payload bytes into the output buffer, computes the CRC, appends the CRC
// bytes, and stores the integer CRC for optional host retrieval.
static __global__ void cuda_crc_encode_kernel(const uint8_t *input_bytes,
    uint8_t *out_bytes, uint32_t *out_crc, const uint32_t *tables,
    uint32_t input_len, uint32_t crc_len_bytes, uint32_t width,
    uint32_t mask) {
    const uint32_t *word_table = tables;
    const uint32_t *crc_table = tables + CUDA_CRC_WORD_TABLE_ENTRIES;
    uint32_t crc = 0;
    uint32_t byte_idx = 0;

    for (; byte_idx + CUDA_CRC_WORD_BYTES <= input_len;
            byte_idx += CUDA_CRC_WORD_BYTES) {
        uint32_t word = ((uint32_t)input_bytes[byte_idx] << 24) |
            ((uint32_t)input_bytes[byte_idx + 1u] << 16) |
            ((uint32_t)input_bytes[byte_idx + 2u] << 8) |
            (uint32_t)input_bytes[byte_idx + 3u];

        out_bytes[byte_idx] = input_bytes[byte_idx];
        out_bytes[byte_idx + 1u] = input_bytes[byte_idx + 1u];
        out_bytes[byte_idx + 2u] = input_bytes[byte_idx + 2u];
        out_bytes[byte_idx + 3u] = input_bytes[byte_idx + 3u];
        crc = cuda_crc_update32(crc, word, word_table, width, mask);
    }

    for (; byte_idx < input_len; byte_idx++) {
        uint8_t value = input_bytes[byte_idx];
        out_bytes[byte_idx] = value;
        crc = cuda_crc_update8(crc, value, crc_table, width, mask);
    }

    cuda_crc_append_bytes(out_bytes, input_len, crc_len_bytes, crc);
    *out_crc = crc;
}

// Batched variant used when many equal-size CRC blocks can share one launch.
// Each CUDA thread handles one complete payload+CRC operation.
static __global__ void cuda_crc_encode_batch_kernel(const uint8_t *input_bytes,
    uint8_t *out_bytes, uint32_t *out_crc, const uint32_t *tables,
    uint32_t input_len, uint32_t output_len, uint32_t crc_len_bytes,
    uint32_t block_count, uint32_t width, uint32_t mask) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= block_count) {
        return;
    }

    const uint8_t *block_input = input_bytes + block_idx * input_len;
    uint8_t *block_output = out_bytes + block_idx * output_len;
    const uint32_t *word_table = tables;
    const uint32_t *crc_table = tables + CUDA_CRC_WORD_TABLE_ENTRIES;
    uint32_t crc = 0;
    uint32_t byte_idx = 0;

    for (; byte_idx + CUDA_CRC_WORD_BYTES <= input_len;
            byte_idx += CUDA_CRC_WORD_BYTES) {
        uint32_t word = ((uint32_t)block_input[byte_idx] << 24) |
            ((uint32_t)block_input[byte_idx + 1u] << 16) |
            ((uint32_t)block_input[byte_idx + 2u] << 8) |
            (uint32_t)block_input[byte_idx + 3u];

        block_output[byte_idx] = block_input[byte_idx];
        block_output[byte_idx + 1u] = block_input[byte_idx + 1u];
        block_output[byte_idx + 2u] = block_input[byte_idx + 2u];
        block_output[byte_idx + 3u] = block_input[byte_idx + 3u];
        crc = cuda_crc_update32(crc, word, word_table, width, mask);
    }

    for (; byte_idx < input_len; byte_idx++) {
        uint8_t value = block_input[byte_idx];
        block_output[byte_idx] = value;
        crc = cuda_crc_update8(crc, value, crc_table, width, mask);
    }

    cuda_crc_append_bytes(block_output, input_len, crc_len_bytes, crc);
    out_crc[block_idx] = crc;
}

// Fulfill the caller's requested outputs. The function supports host output,
// device output, CRC value, or any combination of those.
static int cuda_crc_copy_outputs(const cuda_crc_workspace_t *ws,
    uint8_t *out_bytes, uint8_t **device_out_bytes, uint32_t *crc,
    uint32_t output_len) {
    if (device_out_bytes != NULL) {
        *device_out_bytes = ws->device_output;
    }
    if (out_bytes != NULL) {
        CUDA_CRC_CHECK(cudaMemcpy(out_bytes, ws->device_output,
                sizeof(uint8_t) * output_len, cudaMemcpyDeviceToHost));
    }
    if (crc != NULL) {
        CUDA_CRC_CHECK(cudaMemcpy(crc, ws->device_crc, sizeof(uint32_t),
                cudaMemcpyDeviceToHost));
    }
    return 0;
}

// Attach an NR CRC to byte-aligned payload bytes using CUDA.
//
// Outputs are optional:
//   out_bytes:        host copy of payload+CRC bytes
//   device_out_bytes: device pointer to workspace payload+CRC bytes
//   crc:              host copy of the integer CRC remainder
//
// At least one output must be requested. The returned device pointer aliases
// internal workspace memory and is overwritten by later calls.
int32_t cuda_crc_encode_bytes(const uint8_t *input_bytes, uint8_t *out_bytes,
    uint8_t **device_out_bytes, uint32_t *crc, uint16_t A, uint16_t B,
    int crctype) {
    if (cuda_crc_validate_request(input_bytes, out_bytes, device_out_bytes,
            crc, A, B, crctype) != 0) {
        return -1;
    }

    uint32_t input_len = (uint32_t)A / 8u;
    uint32_t output_len = (uint32_t)B / 8u;
    uint32_t crc_len_bytes = output_len - input_len;

    crc_spec_t spec;
    if (!cuda_crc_get_spec(crctype, &spec)) {
        return -1;
    }

    cuda_crc_workspace_t *ws = &g_cuda_crc_workspace;
    if (cuda_crc_ensure_workspace(ws, input_len, output_len, 1u) != 0) {
        return -1;
    }
    if (cuda_crc_copy_tables(ws, &spec, crctype) != 0) {
        return -1;
    }

    CUDA_CRC_CHECK(cudaMemcpy(ws->device_input, input_bytes,
            sizeof(uint8_t) * input_len, cudaMemcpyHostToDevice));

    cuda_crc_encode_kernel<<<1, 1>>>(ws->device_input, ws->device_output,
            ws->device_crc, ws->device_tables, input_len, crc_len_bytes,
            spec.width, spec.mask);
    CUDA_CRC_CHECK(cudaGetLastError());

    return cuda_crc_copy_outputs(ws, out_bytes, device_out_bytes, crc,
        output_len);
}

int32_t cuda_crc_encode_bytes_batch(const uint8_t *input_bytes,
    uint8_t *out_bytes, uint32_t *crc, uint16_t A, uint16_t B, int crctype,
    uint32_t block_count) {
    if (input_bytes == NULL || block_count == 0u || (A % 8u) != 0 ||
            (B % 8u) != 0 || B < A || B > MAX_ZC * 22u ||
            crctype < CRC16_TYPE || crctype > CRC24C_TYPE) {
        return -1;
    }

    uint32_t input_len = (uint32_t)A / 8u;
    uint32_t output_len = (uint32_t)B / 8u;
    uint32_t crc_len_bytes = output_len - input_len;
    uint32_t total_input_bytes = input_len * block_count;
    uint32_t total_output_bytes = output_len * block_count;

    crc_spec_t spec;
    if (!cuda_crc_get_spec(crctype, &spec)) {
        return -1;
    }

    cuda_crc_workspace_t *ws = &g_cuda_crc_workspace;
    if (cuda_crc_ensure_workspace(ws, total_input_bytes,
            total_output_bytes, block_count) != 0) {
        return -1;
    }
    if (cuda_crc_copy_tables(ws, &spec, crctype) != 0) {
        return -1;
    }

    CUDA_CRC_CHECK(cudaMemcpy(ws->device_input, input_bytes,
            sizeof(uint8_t) * total_input_bytes, cudaMemcpyHostToDevice));

    const uint32_t threads = 128u;
    cuda_crc_encode_batch_kernel<<<
        (block_count + threads - 1u) / threads, threads>>>(
        ws->device_input, ws->device_output, ws->device_crc,
        ws->device_tables, input_len, output_len, crc_len_bytes,
        block_count, spec.width, spec.mask);
    CUDA_CRC_CHECK(cudaGetLastError());

    if (out_bytes != NULL) {
        CUDA_CRC_CHECK(cudaMemcpy(out_bytes, ws->device_output,
                sizeof(uint8_t) * total_output_bytes,
                cudaMemcpyDeviceToHost));
    }
    if (crc != NULL) {
        CUDA_CRC_CHECK(cudaMemcpy(crc, ws->device_crc,
                sizeof(uint32_t) * block_count, cudaMemcpyDeviceToHost));
    }

    return 0;
}
