#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

/*
 * Runtime CUDA-vs-c_lib decoder smoke test.
 *
 * The test generates random transport-block bits, encodes them with the C
 * reference encoder, sends the rate-matched bits through the shared C channel
 * model, and then decodes the same LLRs with both CUDA and c_lib. The final
 * section profiles both decoder entry points on one representative case.
 */
#define PROFILE_WARMUP_ITERATIONS 2
#define PROFILE_ITERATIONS 20
#define MAX_INFO_BITS 8448
#define MAX_RM_BYTES (MAX_INFO_BITS * 256)
#define CHANNEL_SNR_DB 0.0f
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#ifndef RTLD_DEEPBIND
#define RTLD_DEEPBIND 0
#endif

extern "C" int32_t cuda_ldpc_5g_decode(int16_t *LLr, uint8_t *hardbits,
    ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config);

typedef int32_t (*ldpc_decode_fn_t)(int16_t *LLr, uint8_t *hardbits,
    ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config);
typedef int32_t (*ldpc_encode_fn_t)(uint8_t *infobytes, uint8_t *rm_bytes,
    ldpc_in_config_t *inconfig);
typedef int (*tx_rx_process_fn_t)(const uint8_t *inbytes, size_t num_bits,
    uint16_t Qm, float noise_power_db, int16_t *llr_out);

// c_lib is loaded dynamically so this CUDA-linked executable can compare
// against the independently built C shared library.
typedef struct {
    ldpc_decode_fn_t top_decode;
    ldpc_encode_fn_t top_encode;
    tx_rx_process_fn_t tx_rx_process;
} c_lib_api_t;

// Static buffers keep the test simple and avoid allocation noise in timing.
static uint8_t g_info_bytes[MAX_INFO_BITS / 8];
static uint8_t g_encoded_bytes[MAX_RM_BYTES];
static int16_t g_channel_llr[BG1_NUM_COLS * MAX_ZC];
static uint8_t g_cuda_hardbits[MAX_INFO_BITS / 8];
static uint8_t g_c_hardbits[MAX_INFO_BITS / 8];
static ldpc_in_config_t g_inconfig;
static ldpc_decoder_cfg_t g_decoder_config;

static double now_seconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Decode functions append/shift hard bits into this buffer, so always clear it.
static void clear_hardbits(uint8_t *buffer)
{
    memset(buffer, 0, sizeof(g_cuda_hardbits));
}

// Generate MSB-first packed random information bits, matching numpy.packbits().
static void generate_info_bytes(const ldpc_in_config_t *config)
{
    size_t num_bytes = ((size_t)config->A + 7u) / 8u;

    memset(g_info_bytes, 0, sizeof(g_info_bytes));
    for (uint16_t i = 0; i < config->A; i++) {
        uint8_t bit = (uint8_t)(rand() & 0x01);
        uint16_t byte_idx = i / 8u;
        uint16_t bit_pos = 7u - (i % 8u);

        g_info_bytes[byte_idx] |= (uint8_t)(bit << bit_pos);
    }

    if (num_bytes < sizeof(g_info_bytes)) {
        memset(g_info_bytes + num_bytes, 0,
            sizeof(g_info_bytes) - num_bytes);
    }
}

// Read one bit from an MSB-first packed byte array.
static uint8_t get_packed_bit(const uint8_t *bytes, uint16_t bit_idx)
{
    uint16_t byte_idx = bit_idx / 8u;
    uint8_t bit_pos = (uint8_t)(7u - (bit_idx % 8u));
    return (uint8_t)((bytes[byte_idx] >> bit_pos) & 0x01u);
}

// Compare only the transport-block payload. CRC bits are checked by decoder CRC.
static int verify_payload_bits(const uint8_t *decoded,
    const ldpc_in_config_t *config)
{
    for (uint16_t bit = 0; bit < config->A; bit++) {
        if (get_packed_bit(decoded, bit) != get_packed_bit(g_info_bytes, bit)) {
            fprintf(stderr, "Payload mismatch at bit %u\n", bit);
            return 1;
        }
    }
    return 0;
}

// Keep this aligned with scripts/*_ldpc_encoder_decoder.py min-sum settings.
static void init_decoder_config(uint16_t algo)
{
    g_decoder_config.algo = algo;
    g_decoder_config.max_iterations = 32;
    g_decoder_config.alpha = 22936;
    g_decoder_config.beta = 0;
    g_decoder_config.filling_bit_value = 128 * 10;
    g_decoder_config.max_llr = 32767;
    g_decoder_config.min_llr = -32767;
}

static uint16_t crc_len_bits(uint16_t crctype)
{
    return (crctype == 0) ? 16u : 24u;
}

// Load the C reference encode/decode/channel symbols from common build paths.
static int load_c_lib(void **handle, c_lib_api_t *api)
{
    const char *candidates[] = {
        "c_lib/libnrLDPC.so",
        "../c_lib/libnrLDPC.so",
        "./libnrLDPC.so"
    };

    *handle = NULL;
    for (size_t i = 0; i < ARRAY_SIZE(candidates); i++) {
        *handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        if (*handle != NULL) {
            break;
        }
    }

    if (*handle == NULL) {
        fprintf(stderr, "Failed to open c_lib/libnrLDPC.so: %s\n", dlerror());
        return 1;
    }

    api->top_decode = (ldpc_decode_fn_t)dlsym(*handle, "c_ldpc_5g_decode");
    api->top_encode = (ldpc_encode_fn_t)dlsym(*handle, "c_ldpc_5g_encode");
    api->tx_rx_process =
        (tx_rx_process_fn_t)dlsym(*handle, "tx_rx_process");

    if (api->top_decode == NULL || api->top_encode == NULL ||
            api->tx_rx_process == NULL) {
        fprintf(stderr, "Failed to load one or more c_lib symbols\n");
        dlclose(*handle);
        *handle = NULL;
        return 1;
    }

    return 0;
}

/*
 * Build one channel realization:
 *   random payload -> c_lib encoder -> C tx_rx_process -> int16 LLRs.
 *
 * tx_rx_process takes noise_power_db, not SNR. For normalized symbols and
 * Qm=1, SNR 10 dB corresponds to noise power -10 dB. Higher Qm tests should
 * include the same Qm adjustment used in the Python simulation.
 */
static int generate_channel_llrs(const c_lib_api_t *api,
    const ldpc_in_config_t *config)
{
    float noise_power_db = -CHANNEL_SNR_DB;

    generate_info_bytes(config);
    memset(g_encoded_bytes, 0, sizeof(g_encoded_bytes));
    memset(g_channel_llr, 0, sizeof(g_channel_llr));

    int32_t status = api->top_encode(g_info_bytes, g_encoded_bytes,
        (ldpc_in_config_t *)config);
    if (status != 0) {
        fprintf(stderr,
            "c_lib encoder failed: bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u status=%d\n",
            config->bgn, config->A, config->crctype, config->E, config->rv,
            config->Qm, status);
        return 1;
    }

    status = api->tx_rx_process(g_encoded_bytes, config->E, config->Qm,
        noise_power_db, g_channel_llr);
    if (status != 0) {
        fprintf(stderr,
            "tx_rx_process failed: bgn=%u A=%u E=%u Qm=%u status=%d\n",
            config->bgn, config->A, config->E, config->Qm, status);
        return 1;
    }

    return 0;
}

// Run one decoder entry point on the current global LLR/config buffers.
static int32_t run_decode_once(ldpc_decode_fn_t decode_fn, uint8_t *hardbits,
    int need_cuda_sync)
{
    clear_hardbits(hardbits);
    int32_t status = decode_fn(g_channel_llr, hardbits, &g_inconfig,
        &g_decoder_config);
    if (need_cuda_sync) {
        cudaDeviceSynchronize();
    }
    return status;
}

// Decode one generated block with both implementations and compare outputs.
static int run_one_case(const c_lib_api_t *api,
    const ldpc_in_config_t *config)
{
    size_t hardbit_bytes;

    g_inconfig = *config;
    srand(1000u + (uint32_t)config->A + (uint32_t)config->E +
        (uint32_t)config->Qm * 17u);

    if (generate_channel_llrs(api, &g_inconfig) != 0) {
        return 1;
    }

    // Run c_lib first so a CUDA failure cannot hide the reference result.
    int32_t c_status = run_decode_once(api->top_decode, g_c_hardbits, 0);
    int32_t cuda_status = run_decode_once(cuda_ldpc_5g_decode,
        g_cuda_hardbits, 1);
    if (cuda_status != 0 || c_status != 0) {
        fprintf(stderr,
            "Decoder failed: bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u CUDA=%d c_lib=%d\n",
            config->bgn, config->A, config->crctype, config->E, config->rv,
            config->Qm, cuda_status, c_status);
        return 1;
    }

    hardbit_bytes = ((size_t)config->A + crc_len_bits(config->crctype) + 7u) /
        8u;
    if (memcmp(g_cuda_hardbits, g_c_hardbits, hardbit_bytes) != 0) {
        fprintf(stderr,
            "CUDA/c_lib hardbit mismatch: bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u\n",
            config->bgn, config->A, config->crctype, config->E, config->rv,
            config->Qm);
        return 1;
    }

    if (verify_payload_bits(g_cuda_hardbits, config) != 0) {
        fprintf(stderr,
            "Decoded payload mismatch: bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u\n",
            config->bgn, config->A, config->crctype, config->E, config->rv,
            config->Qm);
        return 1;
    }

    printf("decoded bgn=%u A=%u crc=%u E=%u rv=%u Qm=%u at SNR %.1f dB\n",
        config->bgn, config->A, config->crctype, config->E, config->rv,
        config->Qm, CHANNEL_SNR_DB);
    return 0;
}

// Measure average decoder latency. CUDA timing includes explicit synchronization.
static double profile_decoder(const char *name, ldpc_decode_fn_t decode_fn,
    uint8_t *hardbits, int need_cuda_sync, int32_t *decode_status)
{
    int32_t status = 0;

    for (int i = 0; i < PROFILE_WARMUP_ITERATIONS; i++) {
        clear_hardbits(hardbits);
        status = decode_fn(g_channel_llr, hardbits, &g_inconfig,
            &g_decoder_config);
        if (need_cuda_sync) {
            cudaDeviceSynchronize();
        }
        if (status < 0) {
            fprintf(stderr, "%s warmup failed with status %d\n", name, status);
            *decode_status = status;
            return -1.0;
        }
    }

    double start = now_seconds();
    for (int i = 0; i < PROFILE_ITERATIONS; i++) {
        clear_hardbits(hardbits);
        status = decode_fn(g_channel_llr, hardbits, &g_inconfig,
            &g_decoder_config);
        if (need_cuda_sync) {
            cudaDeviceSynchronize();
        }
        if (status < 0) {
            fprintf(stderr, "%s profiling failed with status %d\n", name,
                status);
            *decode_status = status;
            return -1.0;
        }
    }
    double elapsed = now_seconds() - start;

    *decode_status = status;
    return elapsed * 1000.0 / (double)PROFILE_ITERATIONS;
}

int main()
{
    /*
     * Cases are intentionally rv=0 and conservative rates. More aggressive
     * modulation/rate combinations can be legitimate channel failures for both
     * implementations, which makes them poor smoke-test comparisons.
     */
    const ldpc_in_config_t cases[] = {
        {1, 488, 1, 1536, 0, 1, 0},
        {1, 1024, 0, 3072, 0, 1, 0},
        {1, 2400, 2, 8192, 0, 1, 0},
        {1, 4096-24, 1, 16384, 0, 1, 0},
        {2, 176, 1, 512, 0, 1, 0},
        {2, 640, 3, 2048, 0, 1, 0},
        {2, 1200, 0, 4096, 0, 1, 0},
        {2, 2400, 2, 8192, 0, 1, 0},
    };
    void *c_lib_handle = NULL;
    c_lib_api_t c_lib_api;
    memset(&c_lib_api, 0, sizeof(c_lib_api));

    init_decoder_config(1);

    if (load_c_lib(&c_lib_handle, &c_lib_api) != 0) {
        return 1;
    }

    for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
        if (run_one_case(&c_lib_api, &cases[i]) != 0) {
            dlclose(c_lib_handle);
            return 1;
        }
    }

    g_inconfig = cases[3];
    srand(2000u);
    if (generate_channel_llrs(&c_lib_api, &g_inconfig) != 0) {
        dlclose(c_lib_handle);
        return 1;
    }

    int32_t cuda_status = 1;
    int32_t c_status = 1;
    double cuda_avg_ms = profile_decoder("CUDA", cuda_ldpc_5g_decode,
        g_cuda_hardbits, 1, &cuda_status);
    double c_avg_ms = profile_decoder("c_lib", c_lib_api.top_decode,
        g_c_hardbits, 0, &c_status);

    if (cuda_avg_ms < 0.0 || c_avg_ms < 0.0) {
        dlclose(c_lib_handle);
        return 1;
    }

    printf("LDPC decoder timing for bgn=%u A=%u E=%u rv=%u Qm=%u "
        "algo=%u SNR=%.1f dB over %d iterations after %d warmup runs\n",
        g_inconfig.bgn, g_inconfig.A, g_inconfig.E, g_inconfig.rv,
        g_inconfig.Qm, g_decoder_config.algo, CHANNEL_SNR_DB,
        PROFILE_ITERATIONS, PROFILE_WARMUP_ITERATIONS);
    printf("CUDA decoder average: %.6f ms, status %d\n", cuda_avg_ms,
        cuda_status);
    printf("c_lib decoder average: %.6f ms, status %d\n", c_avg_ms,
        c_status);
    printf("Speedup CUDA/c_lib: %.3fx\n", c_avg_ms / cuda_avg_ms);
    printf("All CUDA 5G LDPC decoder tests passed\n");

    dlclose(c_lib_handle);
    return 0;
}
