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

static uint16_t column_list[MAX_DEGREE]; // Column indices for the variable nodes connected to a check node

void _min_sum_update(uint16_t degree, int16_t *rel_Lq, 
    int16_t *sel_lr, int16_t alpha, int16_t beta,int16_t max_llr, int16_t min_llr, 
    uint16_t *column_list, int16_t *LQ) {
    // Min-Sum Algorithm
    
    //first find the minimum and second minimum absolute values among rel_Lq, 
    //and also calculate the overall sign
    int16_t min1 = MAX_LLR_I16;
    int16_t min2 = MAX_LLR_I16;
    int16_t min1_idx = -1;
    int16_t abs_val;
    int sign = 1;
    for (int j = 0; j < degree; j++) {
        if(rel_Lq[j] >=0) {
            //positicve, no need to change sign 
            abs_val = rel_Lq[j];
        } else {
            //negative, change sign from 1->-1 or -1->1
            sign = -sign; // Update overall sign
            abs_val = -rel_Lq[j];
        }
        
        if (abs_val < min1) {
            min2 = min1;
            min1 = abs_val;
            min1_idx = j;
        } else if (abs_val < min2) {
            min2 = abs_val;
        }
    }
    // now sign is the product of the signs of all the connected variable nodes

    //make sure min1 and min2 is not larger than max_llr, this step seems unnecessary 
    if (min1 > max_llr)  min1 = max_llr;
    if (min2 > max_llr)  min2 = max_llr;
    
    //adjust min1 and min2 for normalized min-sum approximation and offset min-sum approximation
    int32_t tmp1,tmp2;
    int16_t new_min1, new_min2;
    tmp1 = ((int16_t)alpha * (int16_t)((min1 - beta)>0? min1 - beta : 0)); // Apply scaling and offset for min-sum approximation
    tmp2 = ((int16_t)alpha * (int16_t)((min2 - beta)>0? min2 - beta : 0)); // Apply scaling and offset for min-sum approximation
    new_min1 = (int16_t)((tmp1+(1<<14))>>15); // Apply scaling for normalized min-sum approximation, with rounding
    new_min2 = (int16_t)((tmp2+(1<<14))>>15);
    
    //update Lr and LQ for this check node
    for (int j = 0; j < degree; j++) {
        int16_t new_val = (j == min1_idx) ? new_min2 : new_min1; // Min-Sum update
        if(rel_Lq[j] >=0) {
            //positicve, no need to change sign 
            sel_lr[j] = sign * new_val; // Apply overall sign and individual sign
        } else {
            //negative, change sign from 1->-1 or -1->1
            sel_lr[j] = -sign * new_val; // Apply overall sign and individual sign
        }              
        
        // update LQ immediately after updating R for this check node, so that the updated LQ can be used in the next check node in the same layer
        int32_t new_LQ = (int32_t)rel_Lq[j] + (int32_t)sel_lr[j]; // LQ = Relative LLR + new R
        
        // Saturate to int16 range
        if (new_LQ > (int32_t)max_llr) {
            new_LQ = (int32_t)max_llr;
        } else if (new_LQ < (int32_t)min_llr) {
            new_LQ = (int32_t)min_llr;
        }
                                                        
        LQ[column_list[j]] = (int16_t)new_LQ; // Update LQ
    }
}

int32_t ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
    // This function performs LDPC decoding based on the input configuration and input LLR.
    // The output bits are stored in hardbits.
    uint16_t non_zero_elements;
    uint16_t num_rows;

    //read parameters from config
    uint16_t bgn = config->bgn;
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
    memcpy(LQ, LLr, sizeof(int16_t)*(config->N+2*Zc)); 

    //clean Lr mem
    memset(Lr, 0, sizeof(int16_t)*non_zero_elements*Zc);

    /* ldpc main loop */
    for(uint16_t iter=0; iter<decoder_cfg->max_iterations; iter++){
        // --- Layered Processing ---
        for (uint16_t HBG_row = 0; HBG_row < num_rows; HBG_row++) {
            //get offset and degree for this HBG row
            

            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            
            //columns indicate the column index j for the non-null elements in this row, 
            //and Vij indicate the corresponding Vij value for each non-null element
            //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
            //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            //each HBG rows index contain Zc row, so we need to process Zc rows for each HBG row
            for (uint16_t zx = 0; zx < Zc; zx++) {
                int16_t *sel_lr = Lr + offset*Zc + zx*degree; // Pointer to the start of Lr for this check node (row)
                
                //calculate relative LLRs for the variable nodes connected to this check node
                for (uint16_t j = 0; j < degree; j++) {
                    int16_t HBG_col_idx = HBGcolumns[j]; //HBG column index for HBG row i and non-null element j    
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc)) ; // Variable node index for this check node
                    column_list[j] = column; // Store the column index for later use in updating LQ
                    rel_Lq[j] = LQ[column] - sel_lr[j]; // Relative LLR for this variable node
                }

                //update Lr, fixed point only support min-sum algorithm for now, 
                //no need to calculate tanh and atanh, which is computationally expensive
                if(decoder_cfg->algo == 1) {
                    _min_sum_update(degree, rel_Lq, sel_lr, 
                        decoder_cfg->alpha, decoder_cfg->beta, 
                        decoder_cfg->max_llr, decoder_cfg->min_llr, 
                        column_list, LQ);
                }
                else {
                    // Unsupported algorithm
                    return -1;
                }
            } // End of processing for this HBG row
        } // End of layered processing for this iteration
        
        // --- Early Termination (Syndrome Check) at the end of each iteration ---
        //calculate hard bits from LQ, and store in hardbits array, which is ceil(K/8) bytes, MSB first in each byte
        for (int i = 0; i < config->K; i++) {
            hardbits[i/8] = (hardbits[i/8] << 1) | (LQ[i] < 0 ? 1 : 0); // MSB first
        }
        
        //crc check
        if(calculate_crc(hardbits, config->B/8, config->crctype) == 0) {
            return 0; // Converged successfully
        }
        // If crc check fails, continue to the next iteration
    }
    return 1; // Failed to converge within max_iter
}

