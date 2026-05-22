#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

#define CUDA_CHECK_TOP_ENCODE(call)                                          \
    do {                                                                     \
        cudaError_t cuda_check_status = (call);                              \
        if (cuda_check_status != cudaSuccess) {                              \
            fprintf(stderr, "CUDA top encoder error %s:%d: %s\n", __FILE__,  \
                    __LINE__, cudaGetErrorString(cuda_check_status));        \
            return -1;                                                       \
        }                                                                    \
    } while (0)

typedef struct {
    uint8_t *ldpc_bits;
} cuda_ldpc_5g_encode_workspace_t;

static cuda_ldpc_5g_encode_workspace_t g_cuda_ldpc_5g_encode_workspace;

extern int32_t cuda_crc_encode_bits_to_device(const uint8_t *input_bytes,
    uint8_t *device_out_bits, uint32_t *crc, uint16_t A, uint16_t B,
    int crctype);
extern int32_t cuda_nr_ldpc_encode_to_device(const uint8_t *device_in_bits,
    uint8_t *device_out_bits, ldpc_config_t *ldpc_config);
extern int32_t cuda_nr_ldpc_rate_match_from_device(
    const uint8_t *device_inbits, uint8_t *outbytes, ldpc_config_t *config);

static int32_t cuda_ldpc_5g_encode_ensure_workspace(
    cuda_ldpc_5g_encode_workspace_t *ws)
{
    if (ws->ldpc_bits == NULL) {
        CUDA_CHECK_TOP_ENCODE(cudaMalloc((void **)&ws->ldpc_bits,
            sizeof(uint8_t) * MAX_ZC * BG1_NUM_COLS));
    }

    return 0;
}

/********* LDPC Decoder Main Function *********/
int32_t c_ldpc_5g_encode(uint8_t *infobytes, uint8_t *rm_bytes, ldpc_in_config_t *inconfig) {
    ldpc_config_t ldpc_config;

    //first generate the ldpc parameters based on the input configuration, which will be used for rate recovery and LDPC decoding
    if(gen_ldpc_parameters(&ldpc_config, inconfig) != 0) {
        fprintf(stderr, "Invalid ldpc configuration: \n");
        fprintf(stderr, "bgn= %d, A=%d,crctype=%d,E=%d, rv=%d, Qm=%d, HARQ_on=%d \n",
            inconfig->bgn, inconfig->A, inconfig->crctype, inconfig->E, inconfig->rv, inconfig->Qm, inconfig->HARQ_on);
        return -2; // Invalid ldpc configuration
    }

    cuda_ldpc_5g_encode_workspace_t *ws = &g_cuda_ldpc_5g_encode_workspace;
    if (cuda_ldpc_5g_encode_ensure_workspace(ws) != 0) {
        return -1;
    }

    // CRC writes payload+CRC bits to device memory for LDPC consumption.
    if (cuda_crc_encode_bits_to_device(infobytes, ws->ldpc_bits, NULL,
            ldpc_config.A, ldpc_config.B, ldpc_config.crctype) != 0) {
        return -1;
    }

    // LDPC runs in-place: it preserves [0, B), zeros filler bits, and writes
    // parity bits after K in the same device buffer.
    if (cuda_nr_ldpc_encode_to_device(ws->ldpc_bits, ws->ldpc_bits,
            &ldpc_config) != 0) {
        return -1;
    }

    // Rate matching reads encoded device bits and copies packed output bytes out.
    if (cuda_nr_ldpc_rate_match_from_device(ws->ldpc_bits, rm_bytes,
            &ldpc_config) != 0) {
        return -1;
    }

    return 0;
}
