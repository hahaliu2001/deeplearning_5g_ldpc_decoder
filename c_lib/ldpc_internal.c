#include <stdio.h>
#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

// pre-allocate HARQ memory for one code block, the size is BG1_NUM_COLS*MAX_ZC,
// which is the maximum number of bits in one code block after LDPC encoding
int16_t HARQ_memory[BG1_NUM_COLS*MAX_ZC]; 

table_5_3_2_1_t table_5_3_2_1 = {
    .iLS0Zc = {2, 4, 8, 16, 32, 64, 128, 256},
    .iLS1Zc = {3, 6, 12, 24, 48, 96, 192, 384},
    .iLS2Zc = {5, 10, 20, 40, 80, 160, 320},
    .iLS3Zc = {7, 14, 28, 56, 112, 224},
    .iLS4Zc = {9, 18, 36, 72, 144, 288},
    .iLS5Zc = {11, 22, 44, 88, 176, 352},
    .iLS6Zc = {13, 26, 52, 104, 208},
    .iLS7Zc = {15, 30, 60, 120, 240}
};

void search_table_5_3_2_1(uint16_t B, uint16_t bgn, uint16_t *Zc, uint16_t *iLS) {
    // 38.212 5.3.2 table 5.3.2-1 defines the maximum lifting size Zc 
    // and the corresponding lifting size index iLS for a given B

    //38.212 5.2.2 ,find Kb
    uint16_t Kb;
    if(bgn==1){
        Kb = 22;
    } else {
        if(B > 640) Kb = 10;
        else if(B > 560) Kb = 9;
        else if(B > 192) Kb = 8;
        else Kb=6;
    }
    
    //find Zc to satisfy Kb*Zc >= B, and Zc is the minimum lifting size in table 5.3.2-1
    uint16_t *sets[] = { 
        table_5_3_2_1.iLS0Zc, table_5_3_2_1.iLS1Zc, table_5_3_2_1.iLS2Zc, 
        table_5_3_2_1.iLS3Zc, table_5_3_2_1.iLS4Zc, table_5_3_2_1.iLS5Zc, 
        table_5_3_2_1.iLS6Zc, table_5_3_2_1.iLS7Zc 
    };
    int sizes[] = {8, 8, 7, 6, 6, 6, 5, 5};
    uint16_t min_zc = UINT16_MAX;
    int best_set = -1;
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            uint16_t Zc = sets[i][j];
            if (Kb * Zc >= B) {
                if (Zc < min_zc) {
                    min_zc = Zc;
                    best_set = i;
    
                }
                break; // No need to check larger Zc in this set
            }
        }
    }
    *Zc = min_zc;
    *iLS = best_set;
}

int gen_ldpc_parameters(ldpc_config_t *params, ldpc_in_config_t *inconfig ) {
    //read input parameters
    uint16_t bgn = inconfig->bgn;
    uint16_t A = inconfig->A;
    uint16_t crctype = inconfig->crctype;
    uint16_t E = inconfig->E;
    uint16_t rv = inconfig->rv;
    uint16_t Qm = inconfig->Qm;
    uint16_t HARQ_on = inconfig->HARQ_on;
    
    // Set input parameters
    params->A = A;
    params->bgn = bgn;
    params->crctype = crctype;
    params->E = E;
    params->rv = rv;
    params->Qm = Qm;
    params->HARQ_on = HARQ_on;

    // Validate input parameters
    if (bgn != 1 && bgn != 2) {
        fprintf(stderr, "Invalid base graph number: %d. Must be 1 or 2.\n", bgn);
        return -1; // Invalid base graph number
    }
    if (crctype > 3) {
        fprintf(stderr, "Invalid CRC type: %d. Must be in [0,1,2,3].\n", crctype);
        return -1; // Invalid CRC type
    }
    if (Qm != 1 && Qm != 2 && Qm != 4 && Qm != 6 && Qm != 8) {
        fprintf(stderr, "Invalid modulation order: %d. Must be in [1,2,4,6,8].\n", Qm);
        return -1; // Invalid modulation order
    }
    if (HARQ_on != 0 && HARQ_on != 1) {
        fprintf(stderr, "Invalid HARQ_on value: %d. Must be 0 or 1.\n", HARQ_on);
        return -1; // Invalid HARQ_on value
    }
    if(rv !=0 && rv !=1 && rv !=2 && rv !=3){
        fprintf(stderr, "Invalid redundancy version: %d. Must be in [0,1,2,3].\n", rv);
        return -1; // Invalid redundancy version
    }
    if(E % Qm != 0){
        fprintf(stderr, "Invalid E value: %d. Must be a multiple of Qm.\n", E);
        return -1; // Invalid E value
    }

    //check 38.214 5.1.3.2 Transport block size determination, all tbsize must be a multiple of 8 bits
    if(A % 8 != 0){
        fprintf(stderr, "Invalid A value: %d. Must be a multiple of 8.\n", A);
        return -1; // Invalid A value
    }
    if(bgn==1 && A>8448-24){
        fprintf(stderr, "Invalid A value: %d for BG1. Must be less than or equal to 8424.\n", A);
        return -1; // Invalid A value for BG1
    }
    if(bgn==2 && A>3840-24){
        fprintf(stderr, "Invalid A value: %d for BG2. Must be less than or equal to 3816.\n", A);
        return -1; // Invalid A value for BG2
    }

    // Derive parameters based on input
    if(crctype == 0){
        params->B = A + 16; // Add 16 bits for CRC16
    } else {
        params->B = A + 24; // Add 24 bits for CRC
    }

    //find Zc and iLS based on B and bgn
    search_table_5_3_2_1(params->B, bgn, &params->Zc, &params->iLS);

    //find K, F, N based on Zc and bgn
    if(bgn == 1){
        params->K = 22 * params->Zc;
        params->N = 66 * params->Zc;
    } else {
        params->K = 10 * params->Zc;
        params->N = 50 * params->Zc;
    }
    params->F = params->K - params->B;
    
    //find k0 based on rv and E
    //38.212 Table 5.4.2.1-2: Starting position of different redundancy versions
    // to simplify the code, we assume Ncb=N, then Ncb=66Zc for BG1, 50Zc for BG2

    if(bgn == 1){
        uint16_t k0_table[4] = {0, 17*params->Zc, 33*params->Zc, 56*params->Zc}; // k0 = {0, 17Zc, 33Zc, 56Zc} for BG1
        params->k0 = k0_table[rv];
    } else {
        uint16_t k0_table[4] = {0, 13*params->Zc, 25*params->Zc, 43*params->Zc}; // k0 = {0, 12Zc, 25Zc, 37Zc} for BG2
        params->k0 = k0_table[rv];
    }

    // Set pointers to the appropriate arrays based on bgn and iLS
    if (bgn == 1) {
        params->Vij_array = BG1_HBG_Vij_array[params->iLS];
        params->HBG_array = BG1_HBG_array;
        params->HBG_offset_array = BG1_HBG_offset_array;
    } else {
        params->Vij_array = BG2_HBG_Vij_array[params->iLS];
        params->HBG_array = BG2_HBG_array;
        params->HBG_offset_array = BG2_HBG_offset_array;
    }
    return 0; // Success
}

