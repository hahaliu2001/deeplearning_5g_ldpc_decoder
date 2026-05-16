#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <dlfcn.h>
#include <time.h>

#include <cuda_runtime.h>

#include "nr_ldpc.h"

#define PROFILE_WARMUP_ITERATIONS 2
#define PROFILE_ITERATIONS 20

#ifndef RTLD_DEEPBIND
#define RTLD_DEEPBIND 0
#endif

uint8_t info_bits[8448]; // 
int16_t LLr[BG1_NUM_COLS*MAX_ZC]; // LLRs for variable nodes, size should match the number of variable nodes in the code
uint8_t info_bytes[8448/8]; // Size should be ceil(K/8) bytes for storing K bits
uint8_t cuda_hardbits[8448/8]; // Decoded hardbits, size should be ceil(K/8) bytes for storing K bits
uint8_t c_hardbits[8448/8]; // Decoded hardbits, size should be ceil(K/8) bytes for storing K bits
ldpc_in_config_t inconfig;
ldpc_decoder_cfg_t decoder_config;

typedef int32_t (*ldpc_decode_fn_t)(int16_t *LLr, uint8_t *hardbits,
    ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config);
typedef int (*gen_ldpc_parameters_fn_t)(ldpc_config_t *params,
    ldpc_in_config_t *inconfig);
typedef void (*rate_recover_fn_t)(ldpc_config_t *config,
    int16_t filling_bit_value, int16_t *input_LLr);
typedef int32_t (*layered_decode_fn_t)(int16_t *LLr, uint8_t *hardbits,
    ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg);

typedef struct {
    ldpc_decode_fn_t top_decode;
    gen_ldpc_parameters_fn_t gen_params;
    rate_recover_fn_t rate_recover;
    layered_decode_fn_t layered_bp;
    layered_decode_fn_t layered_minsum;
    layered_decode_fn_t layered_neural;
    layered_decode_fn_t layered_degree;
    int16_t *harq_memory;
} decoder_api_t;

typedef struct {
    double gen_params_ms;
    double rate_recover_ms;
    double ldpc_decode_ms;
    double total_ms;
    int32_t status;
} decoder_profile_t;

static double now_seconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void clear_hardbits(uint8_t *buffer)
{
    memset(buffer, 0, sizeof(cuda_hardbits));
}

static layered_decode_fn_t select_layered_decoder(const decoder_api_t *api)
{
    if (decoder_config.algo == 0) {
        return api->layered_bp;
    }
    if (decoder_config.algo == 1) {
        return api->layered_minsum;
    }
    if (decoder_config.algo == 2) {
        return api->layered_neural;
    }
    if (decoder_config.algo == 3) {
        return api->layered_degree;
    }
    return NULL;
}

static int load_c_lib_decoder(void **handle, decoder_api_t *api)
{
    const char *candidates[] = {
        "c_lib/libnrLDPC.so",
        "../c_lib/libnrLDPC.so",
        "./libnrLDPC.so"
    };

    *handle = NULL;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
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
    api->gen_params = (gen_ldpc_parameters_fn_t)dlsym(*handle, "gen_ldpc_parameters");
    api->rate_recover = (rate_recover_fn_t)dlsym(*handle,
        "nr_ldpc_rate_recover_harq_combine");
    api->layered_bp = (layered_decode_fn_t)dlsym(*handle,
        "layered_BP_ldpc_decode");
    api->layered_minsum = (layered_decode_fn_t)dlsym(*handle,
        "layered_minsum_ldpc_decode");
    api->layered_neural = (layered_decode_fn_t)dlsym(*handle,
        "layered_neural_minsum_ldpc_decode");
    api->layered_degree = (layered_decode_fn_t)dlsym(*handle,
        "layered_degree_minsum_ldpc_decode");
    api->harq_memory = (int16_t *)dlsym(*handle, "HARQ_memory");

    if (api->top_decode == NULL || api->gen_params == NULL ||
            api->rate_recover == NULL || select_layered_decoder(api) == NULL ||
            api->harq_memory == NULL) {
        fprintf(stderr, "Failed to load one or more c_lib decoder symbols\n");
        dlclose(*handle);
        *handle = NULL;
        return 1;
    }

    return 0;
}

static double profile_decoder(const char *name, ldpc_decode_fn_t decode_fn,
    uint8_t *out_bits, int32_t *decode_status)
{
    int32_t result = 0;

    for (int i = 0; i < PROFILE_WARMUP_ITERATIONS; i++) {
        clear_hardbits(out_bits);
        result = decode_fn(LLr, out_bits, &inconfig, &decoder_config);
        if (result < 0) {
            fprintf(stderr, "%s warmup failed with status %d\n", name, result);
            *decode_status = result;
            return -1.0;
        }
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }

    double start = now_seconds();
    for (int i = 0; i < PROFILE_ITERATIONS; i++) {
        clear_hardbits(out_bits);
        result = decode_fn(LLr, out_bits, &inconfig, &decoder_config);
        if (result < 0) {
            fprintf(stderr, "%s profiling failed with status %d\n", name, result);
            *decode_status = result;
            return -1.0;
        }
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }
    double elapsed = now_seconds() - start;

    *decode_status = result;
    return elapsed * 1000.0 / (double)PROFILE_ITERATIONS;
}

static int profiled_decode_once(const decoder_api_t *api, uint8_t *out_bits,
    decoder_profile_t *profile)
{
    ldpc_config_t ldpc_config;
    layered_decode_fn_t layered_decode = select_layered_decoder(api);
    double t0;

    if (layered_decode == NULL) {
        return -3;
    }

    t0 = now_seconds();
    if (api->gen_params(&ldpc_config, &inconfig) != 0) {
        return -2;
    }
    profile->gen_params_ms += now_seconds() - t0;

    t0 = now_seconds();
    api->rate_recover(&ldpc_config, decoder_config.filling_bit_value, LLr);
    profile->rate_recover_ms += now_seconds() - t0;

    t0 = now_seconds();
    profile->status = layered_decode(api->harq_memory, out_bits, &ldpc_config,
        &decoder_config);
    profile->ldpc_decode_ms += now_seconds() - t0;

    return profile->status;
}

static int profile_decoder_stages(const char *name, const decoder_api_t *api,
    uint8_t *out_bits, decoder_profile_t *profile)
{
    memset(profile, 0, sizeof(*profile));
    profile->status = 1;

    for (int i = 0; i < PROFILE_WARMUP_ITERATIONS; i++) {
        decoder_profile_t warmup_profile;
        memset(&warmup_profile, 0, sizeof(warmup_profile));
        clear_hardbits(out_bits);
        profiled_decode_once(api, out_bits, &warmup_profile);
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }

    double start = now_seconds();
    for (int i = 0; i < PROFILE_ITERATIONS; i++) {
        clear_hardbits(out_bits);
        profiled_decode_once(api, out_bits, profile);
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }
    profile->total_ms = now_seconds() - start;

    profile->gen_params_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->rate_recover_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->ldpc_decode_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->total_ms *= 1000.0 / (double)PROFILE_ITERATIONS;

    return 0;
}

static void print_stage_profile(const char *name, const decoder_profile_t *p)
{
    printf("%s staged timing average:\n", name);
    printf("  gen params:   %.6f ms\n", p->gen_params_ms);
    printf("  rate recover: %.6f ms\n", p->rate_recover_ms);
    printf("  LDPC decode:  %.6f ms\n", p->ldpc_decode_ms);
    printf("  total:        %.6f ms\n", p->total_ms);
    printf("  final status: %d\n", p->status);
}

int32_t load_test1_testdata_from_binary_file() {
    char cwd[PATH_MAX];
    char prefolder[PATH_MAX]; // Define it here so it exists for the whole function
    char full_path[PATH_MAX+64];

    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        // basename() extracts the folder name from the full path
        char *folder_name = basename(cwd);

        if (strcmp(folder_name, "c_lib") == 0) {
            printf("Confirmed: You are inside the 'c_lib' folder.\n");
            strcpy(prefolder, "../out/"); 
        } else {
            printf("Notice: Current folder is '%s', not 'c_lib'.\n", folder_name);
            strcpy(prefolder, "out/");
        }
    } else {
        perror("getcwd() error");
        return 1;
    }

    // Load LLRs from binary file
    snprintf(full_path, sizeof(full_path), "%sLLR_bits.bin", prefolder);
    FILE *f = fopen(full_path, "rb");
    if (f == NULL) {
        perror("Failed to open LLR_bits.bin");
        return 1;
    }
    fread(LLr, sizeof(int16_t), sizeof(LLr) / sizeof(LLr[0]), f); // Read LLR values
    fclose(f);
    
    //load info bits from binary file for testing
    snprintf(full_path, sizeof(full_path), "%sinfo_bits.bin", prefolder);
    f = fopen(full_path, "rb");
    if (f == NULL) {
        perror("Failed to open info_bits.bin");
        return 1;
    }
    fread(info_bits, sizeof(uint8_t), sizeof(info_bits), f); // Read ceil(K/8) bytes of info bits
    fclose(f);
    
    //load info bytes from binary file for testing
    snprintf(full_path, sizeof(full_path), "%sinfo_bytes.bin", prefolder);
    f = fopen(full_path, "rb");
    if (f == NULL) {
        perror("Failed to open info_bytes.bin");
        return 1;
    }
    fread(info_bytes, sizeof(uint8_t), sizeof(info_bytes), f); // Read ceil(K/8) bytes of info bytes
    fclose(f);

    inconfig.A = 512-24;
    inconfig.bgn = 1;
    inconfig.crctype = 1; // crc24a
    inconfig.E = 512*3;
    inconfig.rv = 0;
    inconfig.Qm = 1; // BPSK
    inconfig.HARQ_on = 0;

    decoder_config.algo = 3; // 3 for degree-based min-sum CUDA kernel path
    decoder_config.max_iterations = 32;
    decoder_config.alpha = 32767; // Scaling factor for normalized min-sum approximation, scaled by 32767 for fixed-point representation
    decoder_config.beta = 0 * 32767; // Offset for offset min-sum approximation, scaled by 32767 for fixed-point representation
    decoder_config.filling_bit_value = 128*20; // the LLR value for filling bits in rate recovery, which can be set to 0 for HARQ combining
    decoder_config.max_llr = 32767; // the maximum LLR value for input to LDPC decoder, which is set to 127 for int8 implementation in C
    decoder_config.min_llr = -32768; // the minimum LLR value for input to LDPC decoder, which is set to -128 for int8 implementation in C

    return 0;

}

int main() {
    // read LLRs and info bits from binary files for testing C implementation
    int32_t status = load_test1_testdata_from_binary_file();
    if (status != 0) {
        fprintf(stderr, "Failed to load test data from binary files\n");
        return 1;
    }

    void *c_lib_handle = NULL;
    decoder_api_t cuda_api = {
        c_ldpc_5g_decode,
        gen_ldpc_parameters,
        nr_ldpc_rate_recover_harq_combine,
        layered_BP_ldpc_decode,
        layered_minsum_ldpc_decode,
        layered_neural_minsum_ldpc_decode,
        layered_degree_minsum_ldpc_decode,
        HARQ_memory
    };
    decoder_api_t c_lib_api;
    memset(&c_lib_api, 0, sizeof(c_lib_api));

    if (load_c_lib_decoder(&c_lib_handle, &c_lib_api) != 0) {
        return 1;
    }

    int32_t cuda_status = 1;
    int32_t c_status = 1;
    double cuda_avg_ms = profile_decoder("CUDA", c_ldpc_5g_decode,
        cuda_hardbits, &cuda_status);
    double c_avg_ms = profile_decoder("c_lib", c_lib_api.top_decode,
        c_hardbits, &c_status);

    if (cuda_avg_ms < 0.0 || c_avg_ms < 0.0) {
        dlclose(c_lib_handle);
        return 1;
    }

    ldpc_config_t ldpc_config;
    if (gen_ldpc_parameters(&ldpc_config, &inconfig) != 0) {
        dlclose(c_lib_handle);
        return 1;
    }
    size_t hardbit_bytes = ((size_t)ldpc_config.B + 7u) / 8u;
    if (memcmp(cuda_hardbits, c_hardbits, hardbit_bytes) != 0) {
        fprintf(stderr, "CUDA and c_lib decoder hardbits differ!\n");
        dlclose(c_lib_handle);
        return 1;
    }

    decoder_profile_t cuda_profile;
    decoder_profile_t c_profile;
    profile_decoder_stages("CUDA", &cuda_api, cuda_hardbits, &cuda_profile);
    profile_decoder_stages("c_lib", &c_lib_api, c_hardbits, &c_profile);

    if (memcmp(cuda_hardbits, c_hardbits, hardbit_bytes) != 0) {
        fprintf(stderr, "CUDA and c_lib staged decoder hardbits differ!\n");
        dlclose(c_lib_handle);
        return 1;
    }

    printf("LDPC decoder timing over %d iterations after %d warmup runs\n",
        PROFILE_ITERATIONS, PROFILE_WARMUP_ITERATIONS);
    printf("Decoder algorithm: %u\n", decoder_config.algo);
    printf("CUDA decoder average: %.6f ms, status %d\n",
        cuda_avg_ms, cuda_status);
    printf("c_lib decoder average: %.6f ms, status %d\n",
        c_avg_ms, c_status);
    printf("Speedup CUDA/c_lib: %.3fx\n", c_avg_ms / cuda_avg_ms);
    printf("Output compare: matched %zu hardbit bytes\n", hardbit_bytes);
    print_stage_profile("CUDA", &cuda_profile);
    print_stage_profile("c_lib", &c_profile);

    dlclose(c_lib_handle);
    return 0;
}
