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

static __global__ void cuda_encoder_init_kernel(
    const uint8_t *in_bits,
    uint8_t *out_bits,
    uint16_t B,
    uint16_t K,
    uint16_t out_count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_count) {
        return;
    }

    if (idx < B) {
        out_bits[idx] = in_bits[idx];
    } else {
        out_bits[idx] = 0;
    }
}

static __global__ void cuda_unpack_input_bytes_kernel(
    const uint8_t *in_bytes,
    uint8_t *out_bits,
    uint16_t B)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    uint32_t byte_idx = idx / 8u;
    uint8_t bit_pos = (uint8_t)(7u - (idx % 8u));
    out_bits[idx] = (uint8_t)((in_bytes[byte_idx] >> bit_pos) & 0x01u);
}

static __global__ void cuda_encoder_first_parity_kernel(
    uint8_t *out_bits,
    uint8_t *pc,
    const uint16_t *Vij_array,
    const uint16_t *HBG_array,
    const uint16_t *HBG_offset_array,
    uint16_t bgn,
    uint16_t Zc,
    uint16_t K,
    uint16_t Kb)
{
    uint16_t zx = (uint16_t)(blockIdx.x * blockDim.x + threadIdx.x);
    bool active = zx < Zc;

    uint8_t L0 = 0;
    uint8_t L1 = 0;
    uint8_t L2 = 0;
    uint8_t L3 = 0;
    uint8_t *L[4] = {&L0, &L1, &L2, &L3};

    if (active) {
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
    if (active) {
        pc[(0 * MAX_ZC) + ((shift1 + zx) % Zc)] = pc0;
    }

    __syncthreads();

    if (!active) {
        return;
    }

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
}

static __global__ void cuda_encoder_remaining_parity_kernel(
    uint8_t *out_bits,
    const uint16_t *Vij_array,
    const uint16_t *HBG_array,
    const uint16_t *HBG_offset_array,
    uint16_t num_rows,
    uint16_t Zc,
    uint16_t K,
    uint16_t Kb)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t work_items = (uint32_t)(num_rows - 4u) * (uint32_t)Zc;
    if (idx >= work_items) {
        return;
    }

    uint16_t HBG_row = (uint16_t)(4u + idx / Zc);
    uint16_t zx = (uint16_t)(idx % Zc);
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

    out_bits[K + Zc * HBG_row + zx] = ps;
}

static int32_t nr_ldpc_encode_device_input(const uint8_t *device_in_bits,
    uint8_t *out_bits, ldpc_config_t *ldpc_config) {
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

    if (cuda_encoder_ensure_workspace(ws, B, out_count, non_zero_elements,
            row_offsets) != 0) {
        return -1;
    }

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

    const int threads = 128;
    cuda_encoder_init_kernel<<<(out_count + threads - 1) / threads, threads>>>(
        device_in_bits, ws->out_bits, B, K, (uint16_t)out_count);
    CUDA_CHECK_ENCODE(cudaGetLastError());

    cuda_encoder_first_parity_kernel<<<1, MAX_ZC>>>(
        ws->out_bits, ws->pc, ws->Vij_array, ws->HBG_array,
        ws->HBG_offset_array,
        bgn, Zc, K, Kb);
    CUDA_CHECK_ENCODE(cudaGetLastError());

    uint32_t remaining_items = (uint32_t)(num_rows - 4u) * (uint32_t)Zc;
    cuda_encoder_remaining_parity_kernel<<<
        (remaining_items + threads - 1) / threads, threads>>>(
        ws->out_bits, ws->Vij_array, ws->HBG_array, ws->HBG_offset_array,
        num_rows, Zc, K, Kb);
    CUDA_CHECK_ENCODE(cudaGetLastError());

    CUDA_CHECK_ENCODE(cudaMemcpy(out_bits, ws->out_bits, sizeof(uint8_t) * out_count,
        cudaMemcpyDeviceToHost));

    return 0;
}

int32_t cuda_nr_ldpc_encode_from_device(const uint8_t *device_in_bits,
    uint8_t *out_bits, ldpc_config_t *ldpc_config) {
    if (device_in_bits == NULL || out_bits == NULL || ldpc_config == NULL) {
        return -1;
    }

    return nr_ldpc_encode_device_input(device_in_bits, out_bits, ldpc_config);
}

int32_t cuda_nr_ldpc_encode_packed_from_device(const uint8_t *device_in_bytes,
    uint8_t *out_bits, ldpc_config_t *ldpc_config) {
    cuda_encoder_workspace_t *ws = &g_encoder_workspace;
    const int threads = 128;

    if (device_in_bytes == NULL || out_bits == NULL || ldpc_config == NULL) {
        return -1;
    }
    if (cuda_encoder_ensure_workspace(ws, ldpc_config->B, 0, 0, 0) != 0) {
        return -1;
    }

    cuda_unpack_input_bytes_kernel<<<
        (ldpc_config->B + threads - 1) / threads, threads>>>(
        device_in_bytes, ws->in_bits, ldpc_config->B);
    CUDA_CHECK_ENCODE(cudaGetLastError());

    return nr_ldpc_encode_device_input(ws->in_bits, out_bits, ldpc_config);
}

/*LDPC encoder
in_bit: input bit stream of size B
out_bits: B input bits + F filling bits + w sequence, 
*/
int32_t nr_ldpc_encode(uint8_t *in_bits, uint8_t *out_bits, ldpc_config_t *ldpc_config){
    cuda_encoder_workspace_t *ws = &g_encoder_workspace;

    if (in_bits == NULL || out_bits == NULL || ldpc_config == NULL) {
        return -1;
    }
    if (cuda_encoder_ensure_workspace(ws, ldpc_config->B, 0, 0, 0) != 0) {
        return -1;
    }

    CUDA_CHECK_ENCODE(cudaMemcpy(ws->in_bits, in_bits,
        sizeof(uint8_t) * ldpc_config->B, cudaMemcpyHostToDevice));

    return nr_ldpc_encode_device_input(ws->in_bits, out_bits, ldpc_config);
}

int32_t bak_nr_ldpc_encode(uint8_t *in_bits, uint8_t *out_bits, ldpc_config_t *ldpc_config){

    uint16_t num_rows;

    //read parameters from config
    uint16_t bgn = ldpc_config->bgn;
    uint16_t Zc = ldpc_config->Zc;
    uint16_t K = ldpc_config->K;
    uint16_t B = ldpc_config->B;
    uint16_t F = ldpc_config->F; //number of filling bit
    uint16_t Kb = K / Zc; //22 for bg1, 10 for bg2
    uint16_t *Vij_array = ldpc_config->Vij_array;
    uint16_t *HBG_array = ldpc_config->HBG_array;
    uint16_t *HBG_offset_array = ldpc_config->HBG_offset_array;
    
    if(bgn == 1){
        num_rows = BG1_NUM_ROWS;
    } else {
        num_rows = BG2_NUM_ROWS;
    }

    //first copy in_bit to out_bits, padding 0 for filling bit
    memcpy(out_bits, in_bits, B);
    memset(out_bits+B, 0, F);
    uint8_t *cseq = out_bits; //point to c sequence
    uint8_t *wseq = out_bits+K; //point to w sequence

    //ldpc encoding, refer to LDPC_encoder_optimization.pdf
    uint8_t L[4];
    uint8_t pc[4];
    uint8_t ps[BG1_NUM_ROWS - 4];
    for (uint16_t zx = 0; zx < Zc; zx++){
        //first 4 row is to calculate L
        for(int HBG_row =0; HBG_row < 4; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            //columns indicate the column index j for the non-null elements in this row, 
            //and Vij indicate the corresponding Vij value for each non-null element
            //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
            //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            L[HBG_row] = 0;
            //calculate A @ c
            for(uint32_t j=0; j < degree; j++) {
                int16_t HBG_col_idx = HBGcolumns[j];
                if(HBG_col_idx < Kb) {
                    //this is c bits
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    L[HBG_row] = (L[HBG_row]+cseq[column]) % 2;
                }
                
            }
        }

        //second calculate pc1-pc4
        pc[0] = (L[0] + L[1] + L[2] + L[3]) % 2;
        pc[1] = (L[1] + L[2] + L[3]) % 2;
        pc[3] = (L[0] + L[1] + L[2]) % 2;   
        if(bgn==1) {
            pc[2] = (L[0] + L[1]) % 2;
        } else {
            pc[2] = (L[2] + L[3]) % 2;
        }

        //write pc to wseq
        wseq[zx + Zc * 0] = pc[0];
        wseq[zx + Zc * 1] = pc[0];
        wseq[zx + Zc * 2] = pc[0];
        wseq[zx + Zc * 3] = pc[0];

        //additionaL PARITY bits pa
        for(int HBG_row =4; HBG_row < num_rows; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            //columns indicate the column index j for the non-null elements in this row, 
            //and Vij indicate the corresponding Vij value for each non-null element
            //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
            //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            ps[HBG_row-4] = 0;
            for(uint32_t j=0; j < degree; j++) {
                int16_t HBG_col_idx = HBGcolumns[j];
                //C@c + D@pc
                if(HBG_col_idx < Kb + 4) {
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    ps[HBG_row-4] = (ps[HBG_row-4]+cseq[column]) % 2;
                }
                
            }        

            //write ps to wseq
            wseq[zx + Zc*HBG_row] = ps[HBG_row-4];
        }

    }
    return 0;
}
