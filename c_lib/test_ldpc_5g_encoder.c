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
uint8_t info_bytes[8448/8]; // Size should be ceil(K/8) bytes for storing K bits
ldpc_in_config_t inconfig;
uint8_t c_rmbytes[8448*256];

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
    
    //call ldpc_5g_decode function from C implementation
    int32_t result = c_ldpc_5g_encode(info_bytes, c_rmbytes,&inconfig);

    if (result == 0) {
        printf("Decoding successful!\n");
        // Process hardbits as needed
    } else {
        printf("Decoding failed to converge within max iterations.\n");
    }
    return 0;
}