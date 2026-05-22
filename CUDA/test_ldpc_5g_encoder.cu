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

uint8_t info_bits[8448]; // 
uint8_t info_bytes[8448/8]; // Size should be ceil(K/8) bytes for storing K bits
ldpc_in_config_t inconfig;
uint8_t cuda_rmbytes[8448*256];
uint8_t c_rmbytes[8448*256];

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

    if (api->top_encode == NULL) {
        fprintf(stderr, "Failed to load c_lib encoder symbol\n");
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

    printf("LDPC encoder timing over %d iterations after %d warmup runs\n",
        PROFILE_ITERATIONS, PROFILE_WARMUP_ITERATIONS);
    printf("CUDA encoder average: %.6f ms\n", cuda_avg_ms);
    printf("c_lib encoder average: %.6f ms\n", c_avg_ms);
    printf("Speedup CUDA/c_lib: %.3fx\n", c_avg_ms / cuda_avg_ms);
    printf("Output compare: matched %zu rate-matched bytes\n", rm_bytes_len);

    dlclose(c_lib_handle);
    return 0;
}
