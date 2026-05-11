#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>

#include "nr_ldpc.h"

uint8_t info_bits[8448]; // 
int16_t LLr[BG1_NUM_COLS*MAX_ZC]; // LLRs for variable nodes, size should match the number of variable nodes in the code
uint8_t info_bytes[8448/8]; // Size should be ceil(K/8) bytes for storing K bits
uint8_t hardbits[8448/8]; // Decoded hardbits, size should be ceil(K/8) bytes for storing K bits
ldpc_in_config_t inconfig;
ldpc_decoder_cfg_t decoder_config;

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
    fread(LLr, sizeof(int16_t), sizeof(LLr), f); // Read K LLR values
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

    decoder_config.algo = 0; // 0 for layered-BP, 1 for layered-min-sum
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
    
    //call ldpc_5g_decode function from C implementation
    int32_t result = c_ldpc_5g_decode(LLr, hardbits, &inconfig, &decoder_config);

    if (result == 0) {
        printf("Decoding successful!\n");
        // Process hardbits as needed
    } else {
        printf("Decoding failed to converge within max iterations.\n");
    }
    return 0;
}