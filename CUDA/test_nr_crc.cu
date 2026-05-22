#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "nr_crc_internal.h"
#include "nr_ldpc.h"

extern int32_t cuda_crc_encode_bits_to_device(const uint8_t *input_bytes,
    uint8_t *device_out_bits, uint32_t *crc, uint16_t A, uint16_t B,
    int crctype);

// Keep the benchmark short enough for routine test runs while still doing
// enough repetitions to smooth out timer noise.
#define CRC_BENCH_WARMUP 5u
#define CRC_BENCH_ITERATIONS 200u

static double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *crc_type_name(int crctype) {
    switch (crctype) {
        case CRC16_TYPE:
            return "CRC16";
        case CRC24A_TYPE:
            return "CRC24A";
        case CRC24B_TYPE:
            return "CRC24B";
        case CRC24C_TYPE:
            return "CRC24C";
        default:
            return "UNKNOWN";
    }
}

static uint32_t make_payload_byte(uint32_t index, int crctype,
    uint32_t input_len) {
    uint32_t value = index * 37u + input_len * 11u + (uint32_t)crctype * 19u;
    value ^= value >> 7;
    value ^= value << 3;
    return value & 0xFFu;
}

static uint32_t crc_len_bytes_for_type(int crctype) {
    return crctype == CRC16_TYPE ? 2u : 3u;
}

static int check_device_bit_output(const uint8_t *payload, uint32_t input_len,
    const uint8_t *encoded_bits, uint32_t output_bits, uint32_t reported_crc,
    int crctype) {
    uint32_t crc_len_bits = crc_len_bytes_for_type(crctype) * 8u;
    uint32_t expected_crc = calculate_crc(payload, (int)input_len, crctype);

    if (reported_crc != expected_crc) {
        fprintf(stderr,
            "CRC bit-output value mismatch: type=%d len=%u got=0x%08x expected=0x%08x\n",
            crctype, input_len, reported_crc, expected_crc);
        return 1;
    }

    for (uint32_t bit = 0; bit < input_len * 8u; bit++) {
        uint32_t byte_idx = bit / 8u;
        uint8_t bit_pos = (uint8_t)(7u - (bit % 8u));
        uint8_t expected_bit =
            (uint8_t)((payload[byte_idx] >> bit_pos) & 0x01u);
        if (encoded_bits[bit] != expected_bit) {
            fprintf(stderr,
                "Payload bit mismatch: type=%d len=%u bit=%u got=%u expected=%u\n",
                crctype, input_len, bit, encoded_bits[bit], expected_bit);
            return 1;
        }
    }

    for (uint32_t bit = 0; bit < crc_len_bits; bit++) {
        uint32_t crc_bit_pos = crc_len_bits - 1u - bit;
        uint8_t expected_bit =
            (uint8_t)((expected_crc >> crc_bit_pos) & 0x01u);
        uint32_t out_bit = input_len * 8u + bit;
        if (encoded_bits[out_bit] != expected_bit) {
            fprintf(stderr,
                "CRC bit mismatch: type=%d len=%u bit=%u got=%u expected=%u\n",
                crctype, input_len, bit, encoded_bits[out_bit],
                expected_bit);
            return 1;
        }
    }

    if (output_bits != (input_len * 8u + crc_len_bits)) {
        fprintf(stderr, "Internal bit-output test length mismatch\n");
        return 1;
    }

    return 0;
}

static int run_one_case(int crctype, uint32_t input_len) {
    uint8_t payload[MAX_ZC * 22u / 8u];
    uint8_t device_encoded_bits[MAX_ZC * 22u];
    uint8_t *device_out_bits = NULL;
    uint32_t device_crc = 0;
    uint32_t crc_len_bytes = crc_len_bytes_for_type(crctype);
    uint32_t output_len = input_len + crc_len_bytes;
    uint32_t output_bits = output_len * 8u;

    // Use deterministic pseudo-random payloads so failures are reproducible
    // and every CRC type/length combination exercises different byte values.
    for (uint32_t i = 0; i < input_len; i++) {
        payload[i] = (uint8_t)make_payload_byte(i, crctype, input_len);
    }
    memset(device_encoded_bits, 0x3C, sizeof(device_encoded_bits));

    // Correctness path: write payload+CRC directly as device bits, mirroring
    // the LDPC encoder path, then copy it back here for test validation.
    cudaError_t cuda_status = cudaMalloc((void **)&device_out_bits,
        output_bits);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device CRC bit output: %s\n",
            cudaGetErrorString(cuda_status));
        return 1;
    }
    if (cuda_crc_encode_bits_to_device(payload, device_out_bits, &device_crc,
            (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
            crctype) != 0) {
        fprintf(stderr,
            "cuda_crc_encode_bits_to_device path failed: type=%d len=%u\n",
            crctype, input_len);
        cudaFree(device_out_bits);
        return 1;
    }
    cuda_status = cudaMemcpy(device_encoded_bits, device_out_bits,
        output_bits, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy device CRC bit output: type=%d len=%u: %s\n",
            crctype, input_len, cudaGetErrorString(cuda_status));
        cudaFree(device_out_bits);
        return 1;
    }
    if (check_device_bit_output(payload, input_len, device_encoded_bits,
            output_bits, device_crc, crctype) != 0) {
        cudaFree(device_out_bits);
        return 1;
    }

    volatile uint32_t host_crc_sink = 0;
    double start;
    double host_us;
    double cuda_device_us;

    // Single-block CPU baseline.
    for (uint32_t i = 0; i < CRC_BENCH_WARMUP; i++) {
        host_crc_sink ^= calculate_crc(payload, (int)input_len, crctype);
    }

    start = now_seconds();
    for (uint32_t i = 0; i < CRC_BENCH_ITERATIONS; i++) {
        host_crc_sink ^= calculate_crc(payload, (int)input_len, crctype);
    }
    host_us = (now_seconds() - start) * 1000000.0 /
        (double)CRC_BENCH_ITERATIONS;

    // Single-block CUDA bit-output timing. This includes host-to-device input
    // copy and one kernel launch, so it is expected to be slower than CPU CRC.
    for (uint32_t i = 0; i < CRC_BENCH_WARMUP; i++) {
        if (cuda_crc_encode_bits_to_device(payload, device_out_bits, NULL,
                (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
                crctype) != 0) {
            fprintf(stderr,
                "CUDA CRC bit-output benchmark warmup failed: type=%d len=%u\n",
                crctype, input_len);
            cudaFree(device_out_bits);
            return 1;
        }
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed before benchmark: %s\n",
            cudaGetErrorString(cuda_status));
        cudaFree(device_out_bits);
        return 1;
    }

    start = now_seconds();
    for (uint32_t i = 0; i < CRC_BENCH_ITERATIONS; i++) {
        if (cuda_crc_encode_bits_to_device(payload, device_out_bits, NULL,
                (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
                crctype) != 0) {
            fprintf(stderr, "CUDA CRC bit-output benchmark failed: type=%d len=%u\n",
                crctype, input_len);
            cudaFree(device_out_bits);
            return 1;
        }
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed after benchmark: %s\n",
            cudaGetErrorString(cuda_status));
        cudaFree(device_out_bits);
        return 1;
    }
    cuda_device_us = (now_seconds() - start) * 1000000.0 /
        (double)CRC_BENCH_ITERATIONS;

    printf("%-6s len=%4u host=%8.3f us cuda_bits=%8.3f us\n",
        crc_type_name(crctype), input_len, host_us, cuda_device_us);

    cudaFree(device_out_bits);
    (void)host_crc_sink;
    return 0;
}

int main() {
    const uint32_t lengths[] = {
        1u, 2u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 31u, 32u, 127u, 128u, 1053u
    };
    const int crc_types[] = {
        CRC16_TYPE, CRC24A_TYPE, CRC24B_TYPE, CRC24C_TYPE
    };

    for (uint32_t t = 0; t < sizeof(crc_types) / sizeof(crc_types[0]); t++) {
        for (uint32_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); l++) {
            if (run_one_case(crc_types[t], lengths[l]) != 0) {
                return 1;
            }
        }
    }

    printf("All CUDA NR CRC tests passed\n");
    return 0;
}
