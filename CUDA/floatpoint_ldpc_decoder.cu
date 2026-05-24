#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "nr_crc_internal.h"
#include "nr_ldpc.h"

#define BP_ATANH_CLIP 19.07f
#define FLOAT_LLR_SCALE 128.0f
#define FLOAT_DECODER_THREADS_PER_BLOCK 256
#define DECODE_CRC_PASS 0
#define DECODE_CRC_FAIL 1

#define CUDA_CHECK_FLOAT_DEC_INT(call)                                       \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA float LDPC decoder error %s:%d: %s\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(cuda_check_status)); \
            return -1;                                                       \
        }                                                                    \
    } while (0)

typedef struct {
    int16_t *input_llr;
    float *LQ;
    float *Lr;
    uint8_t *hardbit_bits;
    uint8_t *hardbits;
    int32_t *decode_status;
    uint16_t *Vij_array;
    uint16_t *HBG_array;
    uint16_t *HBG_offset_array;
    uint32_t input_capacity;
    uint32_t lq_capacity;
    uint32_t lr_capacity;
    uint32_t hardbit_bits_capacity;
    uint32_t hardbits_capacity;
    uint32_t status_capacity;
    uint32_t table_capacity;
    uint32_t offset_capacity;
} cuda_float_decoder_workspace_t;

static cuda_float_decoder_workspace_t g_float_decoder_workspace;

// Integer ceil(value / divisor), used for one-dimensional CUDA launch sizes.
static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

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

// Lazily grow reusable device buffers for one float decoder call.
// The workspace is global so repeated code blocks avoid cudaMalloc churn.
static int ensure_float_workspace(cuda_float_decoder_workspace_t *ws,
    uint32_t lq_len, uint32_t lr_len, uint32_t hardbit_bits_len,
    uint32_t hardbytes, uint32_t table_len, uint32_t offset_len) {
    if (ws->input_capacity < lq_len) {
        cudaFree(ws->input_llr);
        ws->input_llr = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->input_llr,
            sizeof(int16_t) * lq_len));
        ws->input_capacity = lq_len;
    }

    if (ws->lq_capacity < lq_len) {
        cudaFree(ws->LQ);
        ws->LQ = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->LQ,
            sizeof(float) * lq_len));
        ws->lq_capacity = lq_len;
    }

    if (ws->lr_capacity < lr_len) {
        cudaFree(ws->Lr);
        ws->Lr = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->Lr,
            sizeof(float) * lr_len));
        ws->lr_capacity = lr_len;
    }

    if (ws->hardbit_bits_capacity < hardbit_bits_len) {
        cudaFree(ws->hardbit_bits);
        ws->hardbit_bits = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->hardbit_bits,
            sizeof(uint8_t) * hardbit_bits_len));
        ws->hardbit_bits_capacity = hardbit_bits_len;
    }

    if (ws->hardbits_capacity < hardbytes) {
        cudaFree(ws->hardbits);
        ws->hardbits = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->hardbits,
            sizeof(uint8_t) * hardbytes));
        ws->hardbits_capacity = hardbytes;
    }

    if (ws->status_capacity < 1u) {
        cudaFree(ws->decode_status);
        ws->decode_status = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->decode_status,
            sizeof(int32_t)));
        ws->status_capacity = 1u;
    }

    if (ws->table_capacity < table_len) {
        cudaFree(ws->Vij_array);
        cudaFree(ws->HBG_array);
        ws->Vij_array = NULL;
        ws->HBG_array = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->Vij_array,
            sizeof(uint16_t) * table_len));
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->HBG_array,
            sizeof(uint16_t) * table_len));
        ws->table_capacity = table_len;
    }

    if (ws->offset_capacity < offset_len) {
        cudaFree(ws->HBG_offset_array);
        ws->HBG_offset_array = NULL;
        CUDA_CHECK_FLOAT_DEC_INT(cudaMalloc((void **)&ws->HBG_offset_array,
            sizeof(uint16_t) * offset_len));
        ws->offset_capacity = offset_len;
    }

    return 0;
}

// Convert int16 channel LLRs into the float belief scale used by BP.
// This runs as a separate simple kernel so the main decoder starts from LQ.
static __global__ void float_init_lq_kernel(float *LQ, const int16_t *LLr,
    uint32_t lq_len) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < lq_len) {
        LQ[idx] = (float)LLr[idx] / FLOAT_LLR_SCALE;
    }
}

// Convert a product of tanh-domain incoming messages back into an LLR-domain
// check-to-variable BP message. Inputs are clipped at +/-1 to avoid atanhf()
// infinities on saturated products.
static __device__ float bp_message_from_product_device(float product) {
    if (product >= 1.0f) {
        return 2.0f * BP_ATANH_CLIP;
    }
    if (product <= -1.0f) {
        return -2.0f * BP_ATANH_CLIP;
    }
    return 2.0f * atanhf(product);
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
static __device__ void make_hardbit_bits_from_lq_device(const float *LQ,
    uint8_t *hardbit_bits, uint16_t B, uint16_t thread_idx,
    uint16_t thread_count) {
    for (uint32_t bit_idx = thread_idx; bit_idx < B; bit_idx += thread_count) {
        hardbit_bits[bit_idx] = (uint8_t)(LQ[bit_idx] < 0.0f ? 1u : 0u);
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

// Layered floating-point BP decoder.
//
// One block is launched with Zc threads. Thread zx processes the zx-th lifted
// check node in each base-graph row. After each row, all threads synchronize
// because layered decoding immediately feeds updated LQ values into later rows.
// After each full iteration, the kernel packs the first B hard decisions and
// checks CRC on device for early termination.
static __global__ void layered_float_bp_decode_kernel(float *LQ, float *Lr,
    const uint16_t *Vij_array, const uint16_t *HBG_array,
    const uint16_t *HBG_offset_array, uint16_t num_rows, uint16_t Zc,
    uint16_t max_iterations, uint8_t *hardbit_bits, uint8_t *hardbits,
    uint16_t B, uint16_t crctype, uint16_t pack_bytes_per_thread,
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
            const uint16_t *hbg_columns = HBG_array + offset;
            const uint16_t *vij = Vij_array + offset;
            float *row_lr = Lr + offset * Zc + zx * degree;

            float rel_Lq[MAX_DEGREE];
            uint16_t column_list[MAX_DEGREE];
            float tanh_values[MAX_DEGREE];
            float prefix_products[MAX_DEGREE + 1];
            float suffix_products[MAX_DEGREE + 1];
            float nonzero_product = 1.0f;
            int16_t zero_idx = -1;
            int16_t num_zeros = 0;

            // Remove the previous check message from each connected variable
            // belief to form variable-to-check inputs for this row.
            for (uint16_t j = 0; j < degree; j++) {
                uint16_t column = (uint16_t)(hbg_columns[j] * Zc +
                    ((vij[j] + zx) % Zc));
                column_list[j] = column;
                rel_Lq[j] = LQ[column] - row_lr[j];

                if (rel_Lq[j] == 0.0f) {
                    num_zeros++;
                    zero_idx = (int16_t)j;
                    tanh_values[j] = 0.0f;
                } else {
                    tanh_values[j] = tanhf(rel_Lq[j] / 2.0f);
                    nonzero_product *= tanh_values[j];
                }
            }

            // Compute all outgoing BP check messages. Zero inputs need special
            // handling because product/tanh division would be undefined.
            if (num_zeros == 1) {
                for (uint16_t j = 0; j < degree; j++) {
                    row_lr[j] = (j == (uint16_t)zero_idx)
                        ? bp_message_from_product_device(nonzero_product)
                        : 0.0f;
                }
            } else if (num_zeros >= 2) {
                for (uint16_t j = 0; j < degree; j++) {
                    row_lr[j] = 0.0f;
                }
            } else {
                prefix_products[0] = 1.0f;
                for (uint16_t j = 0; j < degree; j++) {
                    prefix_products[j + 1] = prefix_products[j] * tanh_values[j];
                }

                suffix_products[degree] = 1.0f;
                for (int j = (int)degree - 1; j >= 0; j--) {
                    suffix_products[j] = suffix_products[j + 1] * tanh_values[j];
                }

                for (uint16_t j = 0; j < degree; j++) {
                    float extrinsic_product =
                        prefix_products[j] * suffix_products[j + 1];
                    row_lr[j] = bp_message_from_product_device(
                        extrinsic_product);
                }
            }

            // Layered update: write the new check message back into Lr and
            // immediately refresh the variable belief LQ for following rows.
            for (uint16_t j = 0; j < degree; j++) {
                LQ[column_list[j]] = rel_Lq[j] + row_lr[j];
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

// Host entry point for the CUDA layered float BP decoder.
// Copies tables/input to device, launches the one-block layered kernel, then
// copies the final packed hardbits and CRC status back to host memory.
int32_t cuda_layered_float_bp_ldpc_decode(int16_t *LLr, uint8_t *hardbits,
    ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    uint16_t non_zero_elements;
    uint16_t num_rows;
    get_base_graph_dimensions(config->bgn, &non_zero_elements, &num_rows);

    uint32_t lq_len = (uint32_t)config->N + 2u * (uint32_t)config->Zc;
    uint32_t lr_len = (uint32_t)non_zero_elements * (uint32_t)config->Zc;
    uint32_t hardbit_bits_len = (uint32_t)config->B;
    uint32_t hardbytes = ((uint32_t)config->B + 7u) / 8u;
    uint32_t offset_len = (uint32_t)num_rows + 1u;
    cuda_float_decoder_workspace_t *ws = &g_float_decoder_workspace;

    if (ensure_float_workspace(ws, lq_len, lr_len, hardbit_bits_len, hardbytes,
            non_zero_elements, offset_len) != 0) {
        return -1;
    }

    CUDA_CHECK_FLOAT_DEC_INT(cudaMemcpy(ws->input_llr, LLr,
        sizeof(int16_t) * lq_len, cudaMemcpyHostToDevice));
    float_init_lq_kernel<<<ceil_div_u32(lq_len,
            FLOAT_DECODER_THREADS_PER_BLOCK),
        FLOAT_DECODER_THREADS_PER_BLOCK>>>(ws->LQ, ws->input_llr, lq_len);
    CUDA_CHECK_FLOAT_DEC_INT(cudaGetLastError());
    CUDA_CHECK_FLOAT_DEC_INT(cudaMemset(ws->Lr, 0, sizeof(float) * lr_len));
    CUDA_CHECK_FLOAT_DEC_INT(cudaMemcpy(ws->Vij_array, config->Vij_array,
        sizeof(uint16_t) * non_zero_elements, cudaMemcpyHostToDevice));
    CUDA_CHECK_FLOAT_DEC_INT(cudaMemcpy(ws->HBG_array, config->HBG_array,
        sizeof(uint16_t) * non_zero_elements, cudaMemcpyHostToDevice));
    CUDA_CHECK_FLOAT_DEC_INT(cudaMemcpy(ws->HBG_offset_array,
        config->HBG_offset_array, sizeof(uint16_t) * offset_len,
        cudaMemcpyHostToDevice));

    const int row_threads = config->Zc;
    const uint16_t pack_bytes_per_thread = (config->bgn == 1) ? 3u : 2u;

    // Run all layered BP rows and iterations inside one Zc-thread block.
    // A block barrier after each row preserves the layered schedule ordering.
    layered_float_bp_decode_kernel<<<1, row_threads>>>(
        ws->LQ, ws->Lr, ws->Vij_array, ws->HBG_array, ws->HBG_offset_array,
        num_rows, config->Zc, decoder_cfg->max_iterations, ws->hardbit_bits,
        ws->hardbits, config->B, config->crctype, pack_bytes_per_thread,
        ws->decode_status);
    CUDA_CHECK_FLOAT_DEC_INT(cudaGetLastError());
    CUDA_CHECK_FLOAT_DEC_INT(cudaMemcpy(hardbits, ws->hardbits,
        sizeof(uint8_t) * hardbytes, cudaMemcpyDeviceToHost));

    int32_t decode_status = DECODE_CRC_FAIL;
    CUDA_CHECK_FLOAT_DEC_INT(cudaMemcpy(&decode_status, ws->decode_status,
        sizeof(int32_t), cudaMemcpyDeviceToHost));

    return decode_status;
}
