#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "nr_crc_internal.h"
#include "nr_ldpc.h"

#define DECODE_CRC_PASS 0
#define DECODE_CRC_FAIL 1

#define CUDA_CHECK_LDPC_DEC_INT(call)                                        \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA fixed LDPC decoder error %s:%d: %s\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(cuda_check_status)); \
            return -1;                                                       \
        }                                                                    \
    } while (0)

typedef struct {
    int16_t *LQ;
    int16_t *Lr;
    uint8_t *hardbit_bits;
    uint8_t *hardbits;
    int32_t *decode_status;
    uint16_t *Vij_array;
    uint16_t *HBG_array;
    uint16_t *HBG_offset_array;
    uint32_t lq_capacity;
    uint32_t lr_capacity;
    uint32_t hardbit_bits_capacity;
    uint32_t hardbits_capacity;
    uint32_t status_capacity;
    uint32_t table_capacity;
    uint32_t offset_capacity;
} cuda_fixed_decoder_workspace_t;

static cuda_fixed_decoder_workspace_t g_fixed_decoder_workspace;

// Return the compressed base-graph table dimensions for BG1 or BG2.
static void get_base_graph_dimensions(uint16_t bgn, uint16_t *non_zero_elements,
    uint16_t *num_rows) {
    if (bgn == 1) {
        *non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        *num_rows = BG1_NUM_ROWS;
    } else {
        *non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        *num_rows = BG2_NUM_ROWS;
    }
}

// Lazily grow reusable device buffers for one fixed-point decoder call.
// The workspace is global so repeated code blocks avoid cudaMalloc churn.
static int ensure_fixed_workspace(cuda_fixed_decoder_workspace_t *ws,
    uint32_t lq_len, uint32_t lr_len, uint32_t hardbit_bits_len,
    uint32_t hardbytes, uint32_t table_len, uint32_t offset_len) {
    if (ws->lq_capacity < lq_len) {
        cudaFree(ws->LQ);
        ws->LQ = NULL;
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->LQ,
            sizeof(int16_t) * lq_len));
        ws->lq_capacity = lq_len;
    }

    if (ws->lr_capacity < lr_len) {
        cudaFree(ws->Lr);
        ws->Lr = NULL;
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->Lr,
            sizeof(int16_t) * lr_len));
        ws->lr_capacity = lr_len;
    }

    if (ws->hardbit_bits_capacity < hardbit_bits_len) {
        cudaFree(ws->hardbit_bits);
        ws->hardbit_bits = NULL;
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->hardbit_bits,
            sizeof(uint8_t) * hardbit_bits_len));
        ws->hardbit_bits_capacity = hardbit_bits_len;
    }

    if (ws->hardbits_capacity < hardbytes) {
        cudaFree(ws->hardbits);
        ws->hardbits = NULL;
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->hardbits,
            sizeof(uint8_t) * hardbytes));
        ws->hardbits_capacity = hardbytes;
    }

    if (ws->status_capacity < 1u) {
        cudaFree(ws->decode_status);
        ws->decode_status = NULL;
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->decode_status,
            sizeof(int32_t)));
        ws->status_capacity = 1u;
    }

    if (ws->table_capacity < table_len) {
        cudaFree(ws->Vij_array);
        cudaFree(ws->HBG_array);
        ws->Vij_array = NULL;
        ws->HBG_array = NULL;
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->Vij_array,
            sizeof(uint16_t) * table_len));
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->HBG_array,
            sizeof(uint16_t) * table_len));
        ws->table_capacity = table_len;
    }

    if (ws->offset_capacity < offset_len) {
        cudaFree(ws->HBG_offset_array);
        ws->HBG_offset_array = NULL;
        CUDA_CHECK_LDPC_DEC_INT(cudaMalloc((void **)&ws->HBG_offset_array,
            sizeof(uint16_t) * offset_len));
        ws->offset_capacity = offset_len;
    }

    return 0;
}

// Clip an accumulated belief to the configured fixed-point LLR range.
static __device__ int16_t clip_llr_device(int32_t value, int16_t max_llr,
    int16_t min_llr) {
    if (value > (int32_t)max_llr) {
        return max_llr;
    }
    if (value < (int32_t)min_llr) {
        return min_llr;
    }
    return (int16_t)value;
}

// Apply normalized/offset min-sum scaling in Q15.
static __device__ int16_t min_sum_scale_device(int16_t magnitude,
    int16_t alpha, int16_t beta) {
    int32_t offset_magnitude = (int32_t)magnitude - (int32_t)beta;
    if (offset_magnitude < 0) {
        offset_magnitude = 0;
    }

    int32_t scaled = (int32_t)alpha * offset_magnitude;
    return (int16_t)((scaled + (1 << 14)) >> 15);
}

// Return abs(value) while folding its sign into the row parity accumulator.
static __device__ int16_t abs_llr_and_update_sign_device(int16_t value,
    int *sign) {
    if (value >= 0) {
        return value;
    }

    *sign = -*sign;
    return (value == MIN_LLR_I16) ? MAX_LLR_I16 : (int16_t)(-value);
}

// Resolve the public CRC type ID to the parameters needed by the bitwise
// device CRC checker. This avoids device access to host-only CRC tables.
static __device__ int get_crc_spec_device(uint16_t crctype, uint32_t *width,
    uint32_t *polynomial, uint32_t *top_bit, uint32_t *mask) {
    switch (crctype) {
        case CRC16_TYPE:
            *width = CRC16_WIDTH;
            *polynomial = CRC16_POLY;
            *top_bit = CRC16_TOP_BIT;
            *mask = CRC16_MASK;
            return 1;
        case CRC24A_TYPE:
            *width = CRC24_WIDTH;
            *polynomial = CRC24A_POLY;
            *top_bit = CRC24_TOP_BIT;
            *mask = CRC24_MASK;
            return 1;
        case CRC24B_TYPE:
            *width = CRC24_WIDTH;
            *polynomial = CRC24B_POLY;
            *top_bit = CRC24_TOP_BIT;
            *mask = CRC24_MASK;
            return 1;
        case CRC24C_TYPE:
            *width = CRC24_WIDTH;
            *polynomial = CRC24C_POLY;
            *top_bit = CRC24_TOP_BIT;
            *mask = CRC24_MASK;
            return 1;
        default:
            return 0;
    }
}

// Compute the NR CRC remainder over packed MSB-first bytes on device.
// Payload+CRC bytes should produce remainder 0 when decoding succeeds.
static __device__ uint32_t calculate_crc_device(const uint8_t *data,
    uint32_t len, uint16_t crctype) {
    uint32_t width;
    uint32_t polynomial;
    uint32_t top_bit;
    uint32_t mask;
    if (!get_crc_spec_device(crctype, &width, &polynomial, &top_bit, &mask)) {
        return 1u;
    }

    uint32_t crc = 0;
    uint32_t shift = width - 8u;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= (uint32_t)data[i] << shift;
        for (uint32_t bit = 0; bit < 8u; bit++) {
            crc = (crc & top_bit) ? ((crc << 1) ^ polynomial) : (crc << 1);
        }
        crc &= mask;
    }

    return crc & mask;
}

// Each zx thread owns the hard decisions for bit positions zx + n * Zc.
// The intermediate hardbit_bits buffer stores one decoded bit per byte so
// threads can write independently before the packed-byte step.
static __device__ void make_hardbit_bits_from_lq_device(const int16_t *LQ,
    uint8_t *hardbit_bits, uint16_t B, uint16_t thread_idx,
    uint16_t thread_count) {
    for (uint32_t bit_idx = thread_idx; bit_idx < B; bit_idx += thread_count) {
        hardbit_bits[bit_idx] = (uint8_t)(LQ[bit_idx] < 0 ? 1u : 0u);
    }
}

// Pack one-bit-per-byte hard decisions into the public MSB-first byte format.
// BG1 needs at most 3 bytes per zx thread; BG2 needs at most 2.
// Each thread owns complete output bytes, so no atomics are needed.
static __device__ void pack_hardbit_bits_device(const uint8_t *hardbit_bits,
    uint8_t *hardbits, uint16_t B, uint16_t thread_idx,
    uint16_t thread_count, uint16_t bytes_per_thread) {
    uint32_t num_bytes = ((uint32_t)B + 7u) / 8u;
    for (uint16_t byte_group = 0; byte_group < bytes_per_thread;
            byte_group++) {
        uint32_t byte_idx = (uint32_t)thread_idx +
            (uint32_t)byte_group * (uint32_t)thread_count;
        if (byte_idx >= num_bytes) {
            return;
        }

        uint8_t out = 0;
        for (uint32_t bit = 0; bit < 8u; bit++) {
            uint32_t bit_idx = byte_idx * 8u + bit;
            out <<= 1;
            if (bit_idx < B) {
                out |= hardbit_bits[bit_idx];
            }
        }
        hardbits[byte_idx] = out;
    }
}

// Layered fixed-point min-sum decoder.
//
// One block is launched with Zc threads. Thread zx processes the zx-th lifted
// check node in each base-graph row. After each row, all threads synchronize
// because layered decoding immediately feeds updated LQ values into later rows.
// After each full iteration, the kernel packs the first B hard decisions and
// checks CRC on device for early termination.
static __global__ void layered_fixed_minsum_decode_kernel(int16_t *LQ,
    int16_t *Lr,
    const uint16_t *Vij_array, const uint16_t *HBG_array,
    const uint16_t *HBG_offset_array, uint16_t num_rows, uint16_t Zc,
    uint16_t max_iterations, int16_t alpha, int16_t beta, int16_t max_llr,
    int16_t min_llr, uint8_t *hardbit_bits, uint8_t *hardbits, uint16_t B,
    uint16_t crctype, uint16_t pack_bytes_per_thread,
    int32_t *decode_status) {
    uint16_t zx = (uint16_t)threadIdx.x;
    if (zx >= Zc) {
        return;
    }

    __shared__ int decode_done;
    if (zx == 0) {
        decode_done = 0;
        *decode_status = DECODE_CRC_FAIL;
    }
    __syncthreads();

    for (uint16_t iter = 0; iter < max_iterations; iter++) {
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            const uint16_t *HBGcolumns = HBG_array + offset;
            const uint16_t *Vij = Vij_array + offset;
            int16_t *row_lr = Lr + offset * Zc + zx * degree;

            int16_t rel_Lq[MAX_DEGREE];
            uint16_t column_list[MAX_DEGREE];
            int16_t min1 = MAX_LLR_I16;
            int16_t min2 = MAX_LLR_I16;
            int16_t min1_idx = -1;
            int sign = 1;

            // Remove the previous check message from each connected variable
            // belief to form variable-to-check inputs for this row.
            for (uint16_t j = 0; j < degree; j++) {
                uint16_t column = (uint16_t)(HBGcolumns[j] * Zc +
                    ((Vij[j] + zx) % Zc));
                column_list[j] = column;
                rel_Lq[j] = (int16_t)(LQ[column] - row_lr[j]);

                int16_t abs_val =
                    abs_llr_and_update_sign_device(rel_Lq[j], &sign);
                if (abs_val < min1) {
                    min2 = min1;
                    min1 = abs_val;
                    min1_idx = (int16_t)j;
                } else if (abs_val < min2) {
                    min2 = abs_val;
                }
            }

            // Compute the two smallest magnitudes and row parity for the
            // normalized/offset min-sum check-node update.
            if (min1 > max_llr) min1 = max_llr;
            if (min2 > max_llr) min2 = max_llr;

            int16_t new_min1 = min_sum_scale_device(min1, alpha, beta);
            int16_t new_min2 = min_sum_scale_device(min2, alpha, beta);

            // Write the outgoing check-to-variable messages for this row.
            for (uint16_t j = 0; j < degree; j++) {
                int16_t new_val = (j == min1_idx) ? new_min2 : new_min1;
                row_lr[j] =
                    (rel_Lq[j] >= 0) ? (int16_t)(sign * new_val)
                                     : (int16_t)(-sign * new_val);
            }

            // Layered update: immediately refresh variable beliefs for
            // following rows in this iteration.
            for (uint16_t j = 0; j < degree; j++) {
                int32_t updated_lq = (int32_t)rel_Lq[j] + (int32_t)row_lr[j];
                LQ[column_list[j]] =
                    clip_llr_device(updated_lq, max_llr, min_llr);
            }

            __syncthreads();
        }

        make_hardbit_bits_from_lq_device(LQ, hardbit_bits, B, zx, Zc);
        __syncthreads();

        pack_hardbit_bits_device(hardbit_bits, hardbits, B, zx, Zc,
            pack_bytes_per_thread);
        __syncthreads();

        // Thread 0 checks the packed first B bits and releases the block early
        // once the decoded code block passes CRC.
        if (zx == 0 &&
                calculate_crc_device(hardbits, (uint32_t)B / 8u, crctype) ==
                    0u) {
            *decode_status = DECODE_CRC_PASS;
            decode_done = 1;
        }

        __syncthreads();
        if (decode_done) {
            break;
        }
    }
}

// Host entry point for the CUDA layered fixed-point min-sum decoder.
// Copies tables/input to device, launches the one-block layered kernel, then
// copies the final packed hardbits and CRC status back to host memory.
int32_t cuda_layered_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits,
    ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    uint16_t non_zero_elements;
    uint16_t num_rows;
    get_base_graph_dimensions(config->bgn, &non_zero_elements, &num_rows);

    uint32_t lq_len = (uint32_t)config->N + 2u * (uint32_t)config->Zc;
    uint32_t lr_len = (uint32_t)non_zero_elements * (uint32_t)config->Zc;
    uint32_t hardbit_bits_len = (uint32_t)config->B;
    uint32_t hardbytes = ((uint32_t)config->B + 7u) / 8u;
    uint32_t offset_len = (uint32_t)num_rows + 1u;
    cuda_fixed_decoder_workspace_t *ws = &g_fixed_decoder_workspace;

    if (ensure_fixed_workspace(ws, lq_len, lr_len, hardbit_bits_len, hardbytes,
            non_zero_elements, offset_len) != 0) {
        return -1;
    }

    CUDA_CHECK_LDPC_DEC_INT(cudaMemcpy(ws->LQ, LLr, sizeof(int16_t) * lq_len,
        cudaMemcpyHostToDevice));
    CUDA_CHECK_LDPC_DEC_INT(cudaMemset(ws->Lr, 0, sizeof(int16_t) * lr_len));
    CUDA_CHECK_LDPC_DEC_INT(cudaMemcpy(ws->Vij_array, config->Vij_array,
        sizeof(uint16_t) * non_zero_elements, cudaMemcpyHostToDevice));
    CUDA_CHECK_LDPC_DEC_INT(cudaMemcpy(ws->HBG_array, config->HBG_array,
        sizeof(uint16_t) * non_zero_elements, cudaMemcpyHostToDevice));
    CUDA_CHECK_LDPC_DEC_INT(cudaMemcpy(ws->HBG_offset_array,
        config->HBG_offset_array, sizeof(uint16_t) * offset_len,
        cudaMemcpyHostToDevice));

    const int row_threads = config->Zc;
    const uint16_t pack_bytes_per_thread = (config->bgn == 1) ? 3u : 2u;

    // Run all layered min-sum rows and iterations inside one Zc-thread block.
    // A block barrier after each row preserves the layered schedule ordering.
    layered_fixed_minsum_decode_kernel<<<1, row_threads>>>(
        ws->LQ, ws->Lr, ws->Vij_array, ws->HBG_array, ws->HBG_offset_array,
        num_rows, config->Zc,
        decoder_cfg->max_iterations, decoder_cfg->alpha, decoder_cfg->beta,
        decoder_cfg->max_llr, decoder_cfg->min_llr, ws->hardbit_bits,
        ws->hardbits, config->B, config->crctype, pack_bytes_per_thread,
        ws->decode_status);
    CUDA_CHECK_LDPC_DEC_INT(cudaGetLastError());
    CUDA_CHECK_LDPC_DEC_INT(cudaMemcpy(hardbits, ws->hardbits,
        sizeof(uint8_t) * hardbytes, cudaMemcpyDeviceToHost));

    int32_t decode_status = DECODE_CRC_FAIL;
    CUDA_CHECK_LDPC_DEC_INT(cudaMemcpy(&decode_status, ws->decode_status,
        sizeof(int32_t), cudaMemcpyDeviceToHost));

    return decode_status;
}
