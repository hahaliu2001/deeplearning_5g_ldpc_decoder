#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

extern int32_t cuda_nr_ldpc_encode_to_device(const uint8_t *device_in_bits,
    uint8_t *device_out_bits, ldpc_config_t *ldpc_config);

typedef int (*gen_ldpc_parameters_fn_t)(ldpc_config_t *params,
    ldpc_in_config_t *inconfig);
typedef int32_t (*nr_ldpc_encode_fn_t)(uint8_t *in_bits, uint8_t *out_bits,
    ldpc_config_t *ldpc_config);

typedef struct {
    void *handle;
    gen_ldpc_parameters_fn_t gen_params;
    nr_ldpc_encode_fn_t encode;
} c_ref_api_t;

static int load_c_reference(c_ref_api_t *api)
{
    const char *candidates[] = {
        "c_lib/libnrLDPC.so",
        "../c_lib/libnrLDPC.so",
        "./libnrLDPC.so"
    };

    memset(api, 0, sizeof(*api));
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        api->handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (api->handle != NULL) {
            break;
        }
    }

    if (api->handle == NULL) {
        fprintf(stderr, "Failed to open c_lib/libnrLDPC.so: %s\n", dlerror());
        return 1;
    }

    api->gen_params = (gen_ldpc_parameters_fn_t)dlsym(api->handle,
        "gen_ldpc_parameters");
    api->encode = (nr_ldpc_encode_fn_t)dlsym(api->handle, "nr_ldpc_encode");
    if (api->gen_params == NULL || api->encode == NULL) {
        fprintf(stderr, "Failed to load C LDPC encoder symbols\n");
        dlclose(api->handle);
        memset(api, 0, sizeof(*api));
        return 1;
    }

    return 0;
}

static uint8_t make_input_bit(uint32_t bit, const ldpc_config_t *config)
{
    uint32_t value = bit * 13u + (uint32_t)config->bgn * 17u +
        (uint32_t)config->crctype * 19u + (uint32_t)config->Zc * 3u;
    value ^= value >> 5;
    value ^= value << 7;
    return (uint8_t)(value & 0x01u);
}

static void fill_input_bits(uint8_t *bits, const ldpc_config_t *config)
{
    for (uint32_t i = 0; i < config->B; i++) {
        bits[i] = make_input_bit(i, config);
    }
}

static uint32_t encoder_output_bits(const ldpc_config_t *config)
{
    uint16_t num_rows = config->bgn == 1 ? BG1_NUM_ROWS : BG2_NUM_ROWS;
    return (uint32_t)config->K + (uint32_t)num_rows * (uint32_t)config->Zc;
}

static int run_one_case(const c_ref_api_t *ref, ldpc_in_config_t inconfig)
{
    ldpc_config_t cuda_config;
    ldpc_config_t c_config;
    uint8_t input_bits[MAX_ZC * 22u];
    uint8_t c_out_bits[MAX_ZC * BG1_NUM_COLS];
    uint8_t cuda_out_bits[MAX_ZC * BG1_NUM_COLS];
    uint8_t *device_bits = NULL;

    if (gen_ldpc_parameters(&cuda_config, &inconfig) != 0) {
        fprintf(stderr, "CUDA gen_ldpc_parameters failed\n");
        return 1;
    }
    if (ref->gen_params(&c_config, &inconfig) != 0) {
        fprintf(stderr, "C gen_ldpc_parameters failed\n");
        return 1;
    }

    uint32_t out_bits = encoder_output_bits(&cuda_config);
    fill_input_bits(input_bits, &cuda_config);
    memset(c_out_bits, 0xA5, sizeof(c_out_bits));
    memset(cuda_out_bits, 0x5A, sizeof(cuda_out_bits));

    if (ref->encode(input_bits, c_out_bits, &c_config) != 0) {
        fprintf(stderr, "C nr_ldpc_encode failed\n");
        return 1;
    }

    cudaError_t status = cudaMalloc((void **)&device_bits,
        sizeof(uint8_t) * out_bits);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        return 1;
    }

    status = cudaMemcpy(device_bits, input_bits, sizeof(uint8_t) * cuda_config.B,
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf(stderr, "Input copy failed: %s\n", cudaGetErrorString(status));
        cudaFree(device_bits);
        return 1;
    }

    if (cuda_nr_ldpc_encode_to_device(device_bits, device_bits,
            &cuda_config) != 0) {
        fprintf(stderr, "CUDA LDPC encode failed\n");
        cudaFree(device_bits);
        return 1;
    }

    status = cudaMemcpy(cuda_out_bits, device_bits, sizeof(uint8_t) * out_bits,
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "Output copy failed: %s\n", cudaGetErrorString(status));
        cudaFree(device_bits);
        return 1;
    }
    cudaFree(device_bits);

    if (memcmp(cuda_out_bits, c_out_bits, out_bits) != 0) {
        for (uint32_t i = 0; i < out_bits; i++) {
            if (cuda_out_bits[i] != c_out_bits[i]) {
                fprintf(stderr,
                    "Mismatch: bgn=%u A=%u crctype=%u Zc=%u bit=%u got=%u expected=%u\n",
                    inconfig.bgn, inconfig.A, inconfig.crctype,
                    cuda_config.Zc, i, cuda_out_bits[i], c_out_bits[i]);
                break;
            }
        }
        return 1;
    }

    printf("matched bgn=%u A=%u crctype=%u Zc=%u B=%u K=%u out=%u bits\n",
        inconfig.bgn, inconfig.A, inconfig.crctype, cuda_config.Zc,
        cuda_config.B, cuda_config.K, out_bits);
    return 0;
}

int main()
{
    const ldpc_in_config_t cases[] = {
        {1, 488, 1, 1536, 0, 1, 0},
        {1, 1024, 0, 2048, 1, 2, 0},
        {1, 2400, 2, 4800, 2, 4, 0},
        {2, 176, 1, 512, 0, 1, 0},
        {2, 640, 3, 1280, 1, 2, 0},
        {2, 1200, 0, 2400, 2, 4, 0},
    };
    c_ref_api_t ref;

    if (load_c_reference(&ref) != 0) {
        return 1;
    }

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_one_case(&ref, cases[i]) != 0) {
            dlclose(ref.handle);
            return 1;
        }
    }

    dlclose(ref.handle);
    printf("All CUDA LDPC encoder tests passed\n");
    return 0;
}
