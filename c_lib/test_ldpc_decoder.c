#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "ldpc_decoder.h"

uint8_t info_bits[3840]; // Size should be ceil(K/8) bytes for storing K bits
int8_t LLr[3840*5]; // LLRs for variable nodes, size should match the number of variable nodes in the code
uint8_t info_bytes[3840/8]; // Size should be ceil(K/8) bytes for storing K bits
uint8_t hardbits[3840/8]; // Decoded hardbits, size should be ceil(K/8) bytes for storing K bits

extern int32_t layered_min_sum_decoder(int8_t *LLr, uint8_t *hardbits, int16_t K, int16_t N, int16_t max_iter,int8_t alpha,int8_t beta);

int32_t load_testdata_from_binary_file() {
    // Load LLRs from binary file
    FILE *f = fopen("out/LLR_bits.bin", "rb");
    if (f == NULL) {
        perror("Failed to open LLR_bits.bin");
        return 1;
    }
    fread(LLr, sizeof(int8_t), sizeof(LLr), f); // Read K LLR values
    fclose(f);
    
    //load info bits from binary file for testing
    f = fopen("out/info_bits.bin", "rb");
    if (f == NULL) {
        perror("Failed to open info_bits.bin");
        return 1;
    }
    fread(info_bits, sizeof(uint8_t), sizeof(info_bits), f); // Read ceil(K/8) bytes of info bits
    fclose(f);
    
    //load info bytes from binary file for testing
    f = fopen("out/info_bytes.bin", "rb");
    if (f == NULL) {
        perror("Failed to open info_bytes.bin");
        return 1;
    }
    fread(info_bytes, sizeof(uint8_t), sizeof(info_bytes), f); // Read ceil(K/8) bytes of info bytes
    fclose(f);

    return 0;

}

int main() {
    // read LLRs and info bits from binary files for testing C implementation
    int32_t status = load_testdata_from_binary_file();
    if (status != 0) {
        fprintf(stderr, "Failed to load test data from binary files\n");
        return 1;
    }
    
    uint16_t K = 1280; // Information bit length
    uint16_t E = 1280*5; // Codeword length (N > K) before rate recovery
    uint16_t max_iter = 32; // Maximum number of iterations for the decoder
    
    //test CRC
    uint32_t crc = calculate_crc24a_fast(info_bytes, K/8-3);

    // Call the decoder
    int result = layered_min_sum_decoder(LLr, hardbits, K, E, max_iter, 89, 0);

    if (result == 0) {
        printf("Decoding successful!\n");
        // Process hardbits as needed
    } else {
        printf("Decoding failed to converge within max iterations.\n");
    }
    return 0;
}