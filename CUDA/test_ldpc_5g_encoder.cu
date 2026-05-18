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

#define PROFILE_WARMUP_ITERATIONS 10
#define PROFILE_ITERATIONS 1000

extern int32_t cuda_crc_encode_bits(const uint8_t *input_bytes,
    uint8_t *out_bits, uint16_t A, uint16_t B, int crctype);

uint8_t info_bits[8448]; // 
uint8_t info_bytes[8448/8]; // Size should be ceil(K/8) bytes for storing K bits
ldpc_in_config_t inconfig;
uint8_t cuda_rmbytes[8448*256];
uint8_t c_rmbytes[8448*256];

typedef int32_t (*ldpc_encode_fn_t)(uint8_t *infobytes, uint8_t *rm_bytes,
    ldpc_in_config_t *inconfig);
typedef int (*gen_ldpc_parameters_fn_t)(ldpc_config_t *params,
    ldpc_in_config_t *inconfig);
typedef uint32_t (*calculate_crc_fn_t)(const uint8_t *data, int len,
    int crctype);
typedef int32_t (*cuda_crc_encode_bits_fn_t)(const uint8_t *input_bytes,
    uint8_t *out_bits, uint16_t A, uint16_t B, int crctype);
typedef int32_t (*nr_ldpc_encode_fn_t)(uint8_t *in_bits, uint8_t *out_bits,
    ldpc_config_t *ldpc_config);
typedef void (*nr_ldpc_rate_match_fn_t)(uint8_t *inbits, uint8_t *outbytes,
    ldpc_config_t *config);

typedef struct {
    ldpc_encode_fn_t top_encode;
    gen_ldpc_parameters_fn_t gen_params;
    calculate_crc_fn_t crc;
    cuda_crc_encode_bits_fn_t cuda_crc_encode_bits;
    nr_ldpc_encode_fn_t ldpc_encode;
    nr_ldpc_rate_match_fn_t rate_match;
} encoder_api_t;

typedef struct {
    double gen_params_ms;
    double crc_ms;
    double unpack_ms;
    double ldpc_encode_ms;
    double rate_match_ms;
    double total_ms;
} encoder_profile_t;

static double now_seconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void clear_output(uint8_t *buffer)
{
    memset(buffer, 0, sizeof(cuda_rmbytes));
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
    api->gen_params = (gen_ldpc_parameters_fn_t)dlsym(*handle, "gen_ldpc_parameters");
    api->crc = (calculate_crc_fn_t)dlsym(*handle, "calculate_crc");
    api->cuda_crc_encode_bits = NULL;
    api->ldpc_encode = (nr_ldpc_encode_fn_t)dlsym(*handle, "nr_ldpc_encode");
    api->rate_match = (nr_ldpc_rate_match_fn_t)dlsym(*handle, "nr_ldpc_rate_match");

    if (api->top_encode == NULL || api->gen_params == NULL ||
            api->crc == NULL || api->ldpc_encode == NULL ||
            api->rate_match == NULL) {
        fprintf(stderr, "Failed to load one or more c_lib encoder symbols\n");
        dlclose(*handle);
        *handle = NULL;
        return 1;
    }

    return 0;
}

static double profile_encoder(const char *name, ldpc_encode_fn_t encode_fn,
    uint8_t *out_bytes)
{
    int32_t result = 0;

    for (int i = 0; i < PROFILE_WARMUP_ITERATIONS; i++) {
        clear_output(out_bytes);
        result = encode_fn(info_bytes, out_bytes, &inconfig);
        if (result != 0) {
            fprintf(stderr, "%s warmup failed with status %d\n", name, result);
            return -1.0;
        }
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }

    double start = now_seconds();
    for (int i = 0; i < PROFILE_ITERATIONS; i++) {
        clear_output(out_bytes);
        result = encode_fn(info_bytes, out_bytes, &inconfig);
        if (result != 0) {
            fprintf(stderr, "%s profiling failed with status %d\n", name, result);
            return -1.0;
        }
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }
    double elapsed = now_seconds() - start;

    return elapsed * 1000.0 / (double)PROFILE_ITERATIONS;
}

static int profiled_encode_once(const encoder_api_t *api, uint8_t *out_bytes,
    encoder_profile_t *profile)
{
    uint8_t ldpc_in_bit[MAX_ZC * 22];
    uint8_t ldpc_out_bits[MAX_ZC * BG1_NUM_COLS];
    ldpc_config_t ldpc_config;
    uint32_t crc;
    double t0;

    memset(ldpc_in_bit, 0, sizeof(ldpc_in_bit));
    memset(ldpc_out_bits, 0, sizeof(ldpc_out_bits));

    t0 = now_seconds();
    if (api->gen_params(&ldpc_config, &inconfig) != 0) {
        return -2;
    }
    profile->gen_params_ms += now_seconds() - t0;

    if (api->cuda_crc_encode_bits != NULL) {
        t0 = now_seconds();
        if (api->cuda_crc_encode_bits(info_bytes, ldpc_in_bit, ldpc_config.A,
                ldpc_config.B, ldpc_config.crctype) != 0) {
            return 1;
        }
        profile->crc_ms += now_seconds() - t0;
    } else {
        t0 = now_seconds();
        crc = api->crc(info_bytes, inconfig.A / 8, inconfig.crctype);
        profile->crc_ms += now_seconds() - t0;

        t0 = now_seconds();
        for (int i = 0; i < inconfig.A; i++) {
            int byte_idx = i / 8;
            int bit_pos = 7 - (i % 8);
            ldpc_in_bit[i] = (info_bytes[byte_idx] >> bit_pos) & 0x01;
        }

        int first_bit_pos = ldpc_config.B - ldpc_config.A - 1;
        for (int i = 0; i < ldpc_config.B - ldpc_config.A; i++) {
            int bit_pos = first_bit_pos - i;
            ldpc_in_bit[ldpc_config.A + i] = (crc >> bit_pos) & 1;
        }
        profile->unpack_ms += now_seconds() - t0;
    }

    t0 = now_seconds();
    if (api->ldpc_encode(ldpc_in_bit, ldpc_out_bits, &ldpc_config) != 0) {
        return 1;
    }
    profile->ldpc_encode_ms += now_seconds() - t0;

    t0 = now_seconds();
    api->rate_match(ldpc_out_bits, out_bytes, &ldpc_config);
    profile->rate_match_ms += now_seconds() - t0;

    return 0;
}

static int profile_encoder_stages(const char *name, const encoder_api_t *api,
    uint8_t *out_bytes, encoder_profile_t *profile)
{
    memset(profile, 0, sizeof(*profile));

    for (int i = 0; i < PROFILE_WARMUP_ITERATIONS; i++) {
        encoder_profile_t warmup_profile;
        memset(&warmup_profile, 0, sizeof(warmup_profile));
        clear_output(out_bytes);
        if (profiled_encode_once(api, out_bytes, &warmup_profile) != 0) {
            fprintf(stderr, "%s staged warmup failed\n", name);
            return 1;
        }
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }

    double start = now_seconds();
    for (int i = 0; i < PROFILE_ITERATIONS; i++) {
        clear_output(out_bytes);
        if (profiled_encode_once(api, out_bytes, profile) != 0) {
            fprintf(stderr, "%s staged profiling failed\n", name);
            return 1;
        }
        if (strcmp(name, "CUDA") == 0) {
            cudaDeviceSynchronize();
        }
    }
    profile->total_ms = now_seconds() - start;

    profile->gen_params_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->crc_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->unpack_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->ldpc_encode_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->rate_match_ms *= 1000.0 / (double)PROFILE_ITERATIONS;
    profile->total_ms *= 1000.0 / (double)PROFILE_ITERATIONS;

    return 0;
}

static void print_stage_profile(const char *name, const encoder_profile_t *p)
{
    printf("%s staged timing average:\n", name);
    printf("  gen params:  %.6f ms\n", p->gen_params_ms);
    printf("  CRC:         %.6f ms\n", p->crc_ms);
    printf("  unpack bits: %.6f ms\n", p->unpack_ms);
    printf("  LDPC encode: %.6f ms\n", p->ldpc_encode_ms);
    printf("  rate match:  %.6f ms\n", p->rate_match_ms);
    printf("  total:       %.6f ms\n", p->total_ms);
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
    
    //load info bits from binary file for testing
    snprintf(full_path, sizeof(full_path), "%sencode_info_bits.bin", prefolder);
    FILE *f  = fopen(full_path, "rb");
    if (f == NULL) {
        perror("Failed to open info_bits.bin");
        return 1;
    }
    fread(info_bits, sizeof(uint8_t), sizeof(info_bits), f); // Read ceil(K/8) bytes of info bits
    fclose(f);
    
    //load info bytes from binary file for testing
    snprintf(full_path, sizeof(full_path), "%sencode_info_bytes.bin", prefolder);
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
    encoder_api_t cuda_api = {
        c_ldpc_5g_encode,
        gen_ldpc_parameters,
        calculate_crc,
        cuda_crc_encode_bits,
        nr_ldpc_encode,
        nr_ldpc_rate_match
    };
    encoder_api_t c_lib_api;
    memset(&c_lib_api, 0, sizeof(c_lib_api));

    if (load_c_lib_encoder(&c_lib_handle, &c_lib_api) != 0) {
        return 1;
    }

    double cuda_avg_ms = profile_encoder("CUDA", c_ldpc_5g_encode, cuda_rmbytes);
    double c_avg_ms = profile_encoder("c_lib", c_lib_api.top_encode, c_rmbytes);

    if (cuda_avg_ms < 0.0 || c_avg_ms < 0.0) {
        dlclose(c_lib_handle);
        return 1;
    }

    size_t rm_bytes_len = ((size_t)inconfig.E + 7u) / 8u;
    if (memcmp(cuda_rmbytes, c_rmbytes, rm_bytes_len) != 0) {
        fprintf(stderr, "CUDA and c_lib encoder outputs differ!\n");
        dlclose(c_lib_handle);
        return 1;
    }

    encoder_profile_t cuda_profile;
    encoder_profile_t c_profile;
    if (profile_encoder_stages("CUDA", &cuda_api, cuda_rmbytes, &cuda_profile) != 0 ||
            profile_encoder_stages("c_lib", &c_lib_api, c_rmbytes, &c_profile) != 0) {
        dlclose(c_lib_handle);
        return 1;
    }

    if (memcmp(cuda_rmbytes, c_rmbytes, rm_bytes_len) != 0) {
        fprintf(stderr, "CUDA and c_lib staged encoder outputs differ!\n");
        dlclose(c_lib_handle);
        return 1;
    }

    printf("LDPC encoder timing over %d iterations after %d warmup runs\n",
        PROFILE_ITERATIONS, PROFILE_WARMUP_ITERATIONS);
    printf("CUDA encoder average: %.6f ms\n", cuda_avg_ms);
    printf("c_lib encoder average: %.6f ms\n", c_avg_ms);
    printf("Speedup CUDA/c_lib: %.3fx\n", c_avg_ms / cuda_avg_ms);
    printf("Output compare: matched %zu rate-matched bytes\n", rm_bytes_len);
    print_stage_profile("CUDA", &cuda_profile);
    print_stage_profile("c_lib", &c_profile);

    dlclose(c_lib_handle);
    return 0;
}
