#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "ldpc_decoder.h"

/*define the arrays used for LDPC decoder*/
//LQ is the array for storing the LLRs for variable nodes, and its max size is 68*384, which is the maximum number of variable nodes in the code. The actual number of variable nodes used in a specific code will be determined by the code parameters (K and N).
//LQ is initialized with 2Zc zero + (B-2Zc) LLr + (K-B) padding + (N-(K-2Zc)) LLr 
static int16_t LQ[BG1_NUM_COLS*MAX_ZC]; // LLRs for variable nodes

//checking nodes to variable nodes, 
static int16_t Lr[BG1_MAX_NON_NULL_ELEMENTS*MAX_ZC]; // Check-to-Variable messagees

static int16_t rel_Lq[MAX_DEGREE]; // Relative LLRs for variable nodes

static int16_t sign_list[MAX_DEGREE]; // Sign list for the variable nodes connected to a check node

static uint16_t column_list[MAX_DEGREE]; // Column indices for the variable nodes connected to a check node

int32_t ldpc_decode(int8_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    // This function performs LDPC decoding based on the input configuration and input LLR.
    // The output bits are stored in hardbits.
    uint16_t non_zero_elements;
    uint16_t num_rows;

    //read parameters from config
    uint16_t bgn = config->bgn;
    uint16_t A = config->A;
    uint16_t crctype = config->crctype;
    uint16_t E = config->E;
    uint16_t rv = config->rv;
    uint16_t Qm = config->Qm;
    uint16_t HARQ_on = config->HARQ_on;
    uint16_t B = config->B;
    uint16_t Zc = config->Zc;

    uint16_t *Vij_array = config->Vij_array;
    uint16_t *HBG_array = config->HBG_array;
    uint16_t *HBG_offset_array = config->HBG_offset_array;

    if(bgn == 1){
        non_zero_elements = BG1_MAX_NON_NULL_ELEMENTS;
        num_rows = BG1_NUM_ROWS;
    } else {
        non_zero_elements = BG2_MAX_NON_NULL_ELEMENTS;
        num_rows = BG2_NUM_ROWS;
    }

    //copy LLR from HARQ memory to LQ, 
    memcpy(LQ, HARQ_memory, sizeof(int16_t)*(config->N+2*Zc)); 

    //clean Lr mem
    memset(Lr, 0, sizeof(int16_t)*non_zero_elements*Zc);

    /* ldpc main loop */
    for(uint16_t iter=0; iter<decoder_cfg->max_iterations; iter++){
        // --- Layered Processing ---
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            //get offset and degree for this HBG row
            uint16_t *HBG_offset_array = config->HBG_offset_array;
            uint16_t *HBG_array = config->HBG_array;
            uint16_t *Vij_array = config->Vij_array;

            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            
            //columns indicate the column index j for the non-null elements in this row, 
            //and Vij indicate the corresponding Vij value for each non-null element
            //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
            //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
            uint8_t *HBGcolumns = HBG_array + offset;
            uint8_t *Vij = Vij_array + offset;

            //each HBG rows index contain Zc row, so we need to process Zc rows for each HBG row
            for (int zx = 0; zx < Zc; zx++) {
                int16_t *sel_lr = Lr + offset*Zc + zx*degree; // Pointer to the start of Lr for this check node (row)
                
                //calculate relative LLRs for the variable nodes connected to this check node
                for (int j = 0; j < degree; j++) {
                    int16_t HBG_col_idx = HBGcolumns[j]; //HBG column index for HBG row i and non-null element j    
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc)) ; // Variable node index for this check node
                    column_list[j] = column; // Store the column index for later use in updating LQ
                    rel_Lq[j] = LQ[column] - sel_lr[j]; // Relative LLR for this variable node
                }

                //update Lr using min-sum algorithm
