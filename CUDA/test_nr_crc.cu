#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "nr_crc_internal.h"
#include "nr_ldpc.h"

extern int32_t cuda_crc_encode_bytes(const uint8_t *input_bytes,
    uint8_t *out_bytes, uint8_t **device_out_bytes, uint32_t *crc,
    uint16_t A, uint16_t B, int crctype);
extern int32_t cuda_crc_encode_bytes_batch(const uint8_t *input_bytes,
    uint8_t *out_bytes, uint32_t *crc, uint16_t A, uint16_t B, int crctype,
    uint32_t block_count);

// Keep the benchmark short enough for routine test runs while still doing
// enough repetitions to smooth out timer noise.
#define CRC_BENCH_WARMUP 5u
#define CRC_BENCH_ITERATIONS 200u

// The batched benchmark amortizes CUDA launch overhead across many independent
// CRC blocks. This is the workload shape where GPU CRC can be useful.
#define CRC_BENCH_BATCH_BLOCKS 4096u

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

static int check_host_output(const uint8_t *payload, uint32_t input_len,
    const uint8_t *encoded, uint32_t output_len, uint32_t reported_crc,
    int crctype) {
    uint32_t crc_len_bytes = crc_len_bytes_for_type(crctype);
    uint32_t expected_crc = calculate_crc(payload, (int)input_len, crctype);

    if (reported_crc != expected_crc) {
        fprintf(stderr,
            "CRC value mismatch: type=%d len=%u got=0x%08x expected=0x%08x\n",
            crctype, input_len, reported_crc, expected_crc);
        return 1;
    }

    if (memcmp(encoded, payload, input_len) != 0) {
        fprintf(stderr, "Payload copy mismatch: type=%d len=%u\n",
            crctype, input_len);
        return 1;
    }

    for (uint32_t i = 0; i < crc_len_bytes; i++) {
        uint32_t shift = (crc_len_bytes - 1u - i) * 8u;
        uint8_t expected_byte = (uint8_t)((expected_crc >> shift) & 0xFFu);
        if (encoded[input_len + i] != expected_byte) {
            fprintf(stderr,
                "CRC byte mismatch: type=%d len=%u byte=%u got=0x%02x expected=0x%02x\n",
                crctype, input_len, i, encoded[input_len + i],
                expected_byte);
            return 1;
        }
    }

    if (output_len != input_len + crc_len_bytes) {
        fprintf(stderr, "Internal test length mismatch\n");
        return 1;
    }

    return 0;
}

static int run_one_case(int crctype, uint32_t input_len) {
    uint8_t payload[MAX_ZC * 22u / 8u];
    uint8_t host_encoded[MAX_ZC * 22u / 8u];
    uint8_t device_encoded[MAX_ZC * 22u / 8u];
    static uint8_t batch_payload[CRC_BENCH_BATCH_BLOCKS * (MAX_ZC * 22u / 8u)];
    static uint8_t batch_encoded[CRC_BENCH_BATCH_BLOCKS * (MAX_ZC * 22u / 8u)];
    static uint32_t batch_crc[CRC_BENCH_BATCH_BLOCKS];
    uint8_t *device_out = NULL;
    uint32_t host_crc = 0;
    uint32_t device_crc = 0;
    uint32_t crc_len_bytes = crc_len_bytes_for_type(crctype);
    uint32_t output_len = input_len + crc_len_bytes;

    // Use deterministic pseudo-random payloads so failures are reproducible
    // and every CRC type/length combination exercises different byte values.
    for (uint32_t i = 0; i < input_len; i++) {
        payload[i] = (uint8_t)make_payload_byte(i, crctype, input_len);
    }
    memset(host_encoded, 0xA5, sizeof(host_encoded));
    memset(device_encoded, 0x5A, sizeof(device_encoded));

    // Correctness path 1: request a host copy of payload+CRC plus the integer
    // CRC remainder from the CUDA helper.
    if (cuda_crc_encode_bytes(payload, host_encoded, NULL, &host_crc,
            (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
            crctype) != 0) {
        fprintf(stderr, "cuda_crc_encode_bytes host path failed: type=%d len=%u\n",
            crctype, input_len);
        return 1;
    }
    if (check_host_output(payload, input_len, host_encoded, output_len,
            host_crc, crctype) != 0) {
        return 1;
    }

    // Correctness path 2: request only the device output pointer, mirroring the
    // LDPC encoder path, then copy it back here just for test validation.
    if (cuda_crc_encode_bytes(payload, NULL, &device_out, &device_crc,
            (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
            crctype) != 0) {
        fprintf(stderr,
            "cuda_crc_encode_bytes device path failed: type=%d len=%u\n",
            crctype, input_len);
        return 1;
    }
    if (cudaMemcpy(device_encoded, device_out, output_len,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device CRC output: type=%d len=%u\n",
            crctype, input_len);
        return 1;
    }
    if (check_host_output(payload, input_len, device_encoded, output_len,
            device_crc, crctype) != 0) {
        return 1;
    }

    volatile uint32_t host_crc_sink = 0;
    double start;
    double host_us;
    double cuda_device_us;
    double host_batch_us;
    double cuda_batch_us;

    // Build a contiguous batch of equal-size payloads. The batch CUDA API
    // treats this as block_count independent CRC jobs.
    for (uint32_t block = 0; block < CRC_BENCH_BATCH_BLOCKS; block++) {
        for (uint32_t i = 0; i < input_len; i++) {
            batch_payload[block * input_len + i] =
                (uint8_t)make_payload_byte(i + block, crctype, input_len);
        }
    }

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

    // Single-block CUDA timing. This includes host-to-device input copy and
    // one kernel launch, so it is expected to be slower than CPU CRC.
    for (uint32_t i = 0; i < CRC_BENCH_WARMUP; i++) {
        if (cuda_crc_encode_bytes(payload, NULL, &device_out, NULL,
                (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
                crctype) != 0) {
            fprintf(stderr,
                "CUDA CRC benchmark warmup failed: type=%d len=%u\n",
                crctype, input_len);
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed before benchmark\n");
        return 1;
    }

    start = now_seconds();
    for (uint32_t i = 0; i < CRC_BENCH_ITERATIONS; i++) {
        if (cuda_crc_encode_bytes(payload, NULL, &device_out, NULL,
                (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
                crctype) != 0) {
            fprintf(stderr, "CUDA CRC benchmark failed: type=%d len=%u\n",
                crctype, input_len);
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed after benchmark\n");
        return 1;
    }
    cuda_device_us = (now_seconds() - start) * 1000000.0 /
        (double)CRC_BENCH_ITERATIONS;

    // Batched CPU baseline: same number of independent CRC jobs as CUDA batch,
    // reported per block.
    for (uint32_t i = 0; i < CRC_BENCH_WARMUP; i++) {
        for (uint32_t block = 0; block < CRC_BENCH_BATCH_BLOCKS; block++) {
            host_crc_sink ^= calculate_crc(batch_payload + block * input_len,
                (int)input_len, crctype);
        }
    }

    start = now_seconds();
    for (uint32_t i = 0; i < CRC_BENCH_ITERATIONS; i++) {
        for (uint32_t block = 0; block < CRC_BENCH_BATCH_BLOCKS; block++) {
            host_crc_sink ^= calculate_crc(batch_payload + block * input_len,
                (int)input_len, crctype);
        }
    }
    host_batch_us = (now_seconds() - start) * 1000000.0 /
        ((double)CRC_BENCH_ITERATIONS * (double)CRC_BENCH_BATCH_BLOCKS);

    // Validate the batched CUDA path with a host output copy before timing the
    // device-only benchmark path.
    if (cuda_crc_encode_bytes_batch(batch_payload, batch_encoded, batch_crc,
            (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
            crctype, CRC_BENCH_BATCH_BLOCKS) != 0) {
        fprintf(stderr, "CUDA CRC batch correctness call failed: type=%d len=%u\n",
            crctype, input_len);
        return 1;
    }
    for (uint32_t block = 0; block < CRC_BENCH_BATCH_BLOCKS; block +=
            CRC_BENCH_BATCH_BLOCKS - 1u) {
        if (check_host_output(batch_payload + block * input_len, input_len,
                batch_encoded + block * output_len, output_len,
                batch_crc[block], crctype) != 0) {
            return 1;
        }
    }

    // Batched CUDA timing. This still includes one input copy per iteration,
    // but the kernel launch is amortized across CRC_BENCH_BATCH_BLOCKS blocks.
    for (uint32_t i = 0; i < CRC_BENCH_WARMUP; i++) {
        if (cuda_crc_encode_bytes_batch(batch_payload, NULL, NULL,
                (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
                crctype, CRC_BENCH_BATCH_BLOCKS) != 0) {
            fprintf(stderr,
                "CUDA CRC batch benchmark warmup failed: type=%d len=%u\n",
                crctype, input_len);
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed before batch benchmark\n");
        return 1;
    }

    start = now_seconds();
    for (uint32_t i = 0; i < CRC_BENCH_ITERATIONS; i++) {
        if (cuda_crc_encode_bytes_batch(batch_payload, NULL, NULL,
                (uint16_t)(input_len * 8u), (uint16_t)(output_len * 8u),
                crctype, CRC_BENCH_BATCH_BLOCKS) != 0) {
            fprintf(stderr, "CUDA CRC batch benchmark failed: type=%d len=%u\n",
                crctype, input_len);
            return 1;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed after batch benchmark\n");
        return 1;
    }
    cuda_batch_us = (now_seconds() - start) * 1000000.0 /
        ((double)CRC_BENCH_ITERATIONS * (double)CRC_BENCH_BATCH_BLOCKS);

    printf("%-6s len=%4u host=%8.3f us cuda1=%8.3f us batch_host=%8.3f us batch_cuda=%8.3f us\n",
        crc_type_name(crctype), input_len, host_us, cuda_device_us,
        host_batch_us, cuda_batch_us);

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
