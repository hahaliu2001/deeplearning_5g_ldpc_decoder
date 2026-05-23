#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

#define PROFILE_WARMUP_ITERATIONS 10
#define PROFILE_ITERATIONS 1000
#define MAX_INFO_BITS 8448
#define MAX_RM_BYTES (MAX_INFO_BITS * 256)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

static uint8_t g_info_bytes[MAX_INFO_BITS / 8];
static uint8_t g_cuda_rmbytes[MAX_RM_BYTES];
static uint8_t g_c_rmbytes[MAX_RM_BYTES];
static ldpc_in_config_t g_inconfig;

typedef int32_t (*ldpc_encode_fn_t)(uint8_t *infobytes, uint8_t *rm_bytes,
    ldpc_in_config_t *inconfig);

typedef struct {
    ldpc_encode_fn_t top_encode;
} encoder_api_t;

static double now_seconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void clear_output(uint8_t *buffer, size_t num_bytes)
{
    memset(buffer, 0, num_bytes);
}

static uint8_t make_info_bit(uint32_t bit, const ldpc_in_config_t *config)
{
    // Deterministic pseudo-random payload bit that changes with the test case.
    uint32_t value = bit * 13u + (uint32_t)config->bgn * 17u +
        (uint32_t)config->crctype * 19u + (uint32_t)config->rv * 23u +
        (uint32_t)config->Qm * 29u + (uint32_t)config->E * 3u;
    value ^= value >> 5;
    value ^= value << 7;
    return (uint8_t)(value & 0x01u);
}

static void generate_info_bytes(const ldpc_in_config_t *config)
{
    size_t num_bytes = ((size_t)config->A + 7u) / 8u;

    /*
     * The top-level encoder API consumes payload bits packed MSB-first:
     * payload bit 0 is bit 7 of byte 0, matching numpy.packbits(..., "big").
     */
    memset(g_info_bytes, 0, sizeof(g_info_bytes));
    for (uint16_t i = 0; i < config->A; i++) {
        uint8_t bit = make_info_bit(i, config);
        uint16_t byte_idx = i / 8u;
        uint16_t bit_pos = 7u - (i % 8u);

        g_info_bytes[byte_idx] |= (uint8_t)(bit << bit_pos);
    }

    if (num_bytes < sizeof(g_info_bytes)) {
        memset(g_info_bytes + num_bytes, 0, sizeof(g_info_bytes) - num_bytes);
    }
}

static int load_c_lib_encoder(void **handle, encoder_api_t *api)
{
    const char *candidates[] = {
        "c_lib/libnrLDPC.so",
        "../c_lib/libnrLDPC.so",
        "./libnrLDPC.so"
    };

    *handle = NULL;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        *handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (*handle != NULL) {
            break;
        }
    }

    if (*handle == NULL) {
        fprintf(stderr, "Failed to open c_lib/libnrLDPC.so: %s\n", dlerror());
        return 1;
    }

    dlerror();
    api->top_encode = (ldpc_encode_fn_t)dlsym(*handle, "c_ldpc_5g_encode");

    if (api->top_encode == NULL) {
        fprintf(stderr, "Failed to load c_lib encoder symbol\n");
        dlclose(*handle);
        *handle = NULL;
        return 1;
    }

    return 0;
}

static double profile_encoder(const char *name, ldpc_encode_fn_t encode_fn,
    uint8_t *out_bytes, int needs_cuda_sync)
{
    int32_t result = 0;
    size_t out_bytes_len = ((size_t)g_inconfig.E + 7u) / 8u;

    for (int i = 0; i < PROFILE_WARMUP_ITERATIONS; i++) {
        clear_output(out_bytes, out_bytes_len);
        result = encode_fn(g_info_bytes, out_bytes, &g_inconfig);
        if (result != 0) {
            fprintf(stderr, "%s warmup failed with status %d\n", name, result);
            return -1.0;
        }
        if (needs_cuda_sync) {
            cudaDeviceSynchronize();
        }
    }

    double start = now_seconds();
    for (int i = 0; i < PROFILE_ITERATIONS; i++) {
        clear_output(out_bytes, out_bytes_len);
        result = encode_fn(g_info_bytes, out_bytes, &g_inconfig);
        if (result != 0) {
            fprintf(stderr, "%s profiling failed with status %d\n", name, result);
            return -1.0;
        }
        if (needs_cuda_sync) {
            cudaDeviceSynchronize();
        }
    }
    double elapsed = now_seconds() - start;

    return elapsed * 1000.0 / (double)PROFILE_ITERATIONS;
}

static int run_one_case(const encoder_api_t *c_lib_api,
    const ldpc_in_config_t *config)
{
    size_t rm_bytes_len = ((size_t)config->E + 7u) / 8u;

    g_inconfig = *config;
    generate_info_bytes(&g_inconfig);
    clear_output(g_cuda_rmbytes, rm_bytes_len);
    clear_output(g_c_rmbytes, rm_bytes_len);

    int32_t cuda_status = c_ldpc_5g_encode(g_info_bytes, g_cuda_rmbytes,
        &g_inconfig);
    if (cuda_status != 0) {
        fprintf(stderr,
            "CUDA encoder failed: bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u status=%d\n",
            g_inconfig.bgn, g_inconfig.A, g_inconfig.crctype, g_inconfig.E,
            g_inconfig.rv, g_inconfig.Qm, cuda_status);
        return 1;
    }

    int32_t c_status = c_lib_api->top_encode(g_info_bytes, g_c_rmbytes,
        &g_inconfig);
    if (c_status != 0) {
        fprintf(stderr,
            "c_lib encoder failed: bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u status=%d\n",
            g_inconfig.bgn, g_inconfig.A, g_inconfig.crctype, g_inconfig.E,
            g_inconfig.rv, g_inconfig.Qm, c_status);
        return 1;
    }

    if (memcmp(g_cuda_rmbytes, g_c_rmbytes, rm_bytes_len) != 0) {
        // Report the first mismatched packed output byte to make debugging fast.
        for (size_t i = 0; i < rm_bytes_len; i++) {
            if (g_cuda_rmbytes[i] != g_c_rmbytes[i]) {
                fprintf(stderr,
                    "Mismatch: bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u byte=%zu got=0x%02x expected=0x%02x\n",
                    g_inconfig.bgn, g_inconfig.A, g_inconfig.crctype,
                    g_inconfig.E, g_inconfig.rv, g_inconfig.Qm, i,
                    g_cuda_rmbytes[i], g_c_rmbytes[i]);
                break;
            }
        }
        return 1;
    }

    printf("matched bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u (%zu bytes)\n",
        g_inconfig.bgn, g_inconfig.A, g_inconfig.crctype, g_inconfig.E,
        g_inconfig.rv, g_inconfig.Qm, rm_bytes_len);

    return 0;
}

int main() {
    /*
     * These cases verify the complete CUDA 5G encoder wrapper:
     * CRC attachment, LDPC encoding, rate matching, interleaving, and packing.
     * The C library is used as the bit-exact reference.
     */
    const ldpc_in_config_t cases[] = {
        {1, 488, 1, 1536, 0, 1, 0},
        {1, 1024, 0, 2048, 1, 2, 0},
        {1, 2400, 2, 4800, 2, 4, 0},
        {1, 8424, 1, 10240, 3, 8, 0},
        {2, 176, 1, 512, 0, 1, 0},
        {2, 640, 3, 1280, 1, 2, 0},
        {2, 1200, 0, 2400, 2, 4, 0},
        {2, 3816, 2, 5120, 3, 8, 0},
    };
    void *c_lib_handle = NULL;
    encoder_api_t c_lib_api;
    memset(&c_lib_api, 0, sizeof(c_lib_api));

    if (load_c_lib_encoder(&c_lib_handle, &c_lib_api) != 0) {
        return 1;
    }

    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        if (run_one_case(&c_lib_api, &cases[i]) != 0) {
            dlclose(c_lib_handle);
            return 1;
        }
    }

    /*
     * Timing is measured through the public top-level APIs. For CUDA this
     * includes host/device copies, kernel launches, and synchronization.
     */
    g_inconfig = cases[3];
    generate_info_bytes(&g_inconfig);
    double cuda_avg_ms = profile_encoder("CUDA", c_ldpc_5g_encode,
        g_cuda_rmbytes, 1);
    double c_avg_ms = profile_encoder("c_lib", c_lib_api.top_encode,
        g_c_rmbytes, 0);

    if (cuda_avg_ms < 0.0 || c_avg_ms < 0.0) {
        dlclose(c_lib_handle);
        return 1;
    }

    printf("LDPC encoder timing for bgn=%u A=%u E=%u rv=%u Qm=%u "
        "over %d iterations after %d warmup runs\n",
        g_inconfig.bgn, g_inconfig.A, g_inconfig.E, g_inconfig.rv,
        g_inconfig.Qm, PROFILE_ITERATIONS, PROFILE_WARMUP_ITERATIONS);
    printf("CUDA encoder average: %.6f ms\n", cuda_avg_ms);
    printf("c_lib encoder average: %.6f ms\n", c_avg_ms);
    printf("Speedup CUDA/c_lib: %.3fx\n", c_avg_ms / cuda_avg_ms);
    printf("All CUDA 5G LDPC encoder tests passed\n");

    dlclose(c_lib_handle);
    return 0;
}
