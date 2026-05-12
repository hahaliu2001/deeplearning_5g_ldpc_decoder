#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#include "nr_ldpc.h"

/*define the arrays used for LDPC decoder*/
//LQ is the array for storing the LLRs for variable nodes, and its max size is 68*384, which is the maximum number of variable nodes in the code. The actual number of variable nodes used in a specific code will be determined by the code parameters (K and N).
//LQ is initialized with 2Zc zero + (B-2Zc) LLr + (K-B) padding + (N-(K-2Zc)) LLr 
static float LQ[BG1_NUM_COLS*MAX_ZC]; // LLRs for variable nodes

//checking nodes to variable nodes, 
static float Lr[BG1_MAX_NON_NULL_ELEMENTS*MAX_ZC]; // Check-to-Variable messagees

static float rel_Lq[MAX_DEGREE]; // Relative LLRs for variable nodes

static uint16_t column_list[MAX_DEGREE]; // Column indices for the variable nodes connected to a check node

void _BP_update(uint16_t degree, float *rel_Lq, 
    float *sel_lr, uint16_t *column_list, float *LQ) {
    // Min-Sum Algorithm
    
    //first find the minimum and second minimum absolute values among rel_Lq, 
    //and also calculate the overall sign
    int16_t zero_idx = -1;
    int16_t num_zeros = 0;
    float tanh_prod = 1.0f;
    for (int j = 0; j < degree; j++) {
        if(0.0f == rel_Lq[j]) {
            num_zeros++;
            zero_idx = j;
        } else {
            tanh_prod *= tanhf(rel_Lq[j]/2);
        }
    }
    // now sign is the product of the signs of all the connected variable nodes   
        
    //update Lr and LQ for this check node
    if(num_zeros == 0) {
    //
        for (int j = 0; j < degree; j++) {
            
            //float new_val = tanh_prod / tanhf(rel_Lq[j]/2);
            //above divide tanhf(rel_Lq[j]/2) may cause issue if it is close to 0
            float new_val = 1.0f;
            if(j==0) {
                for(int m=1; m<degree; m++)
                    new_val *= tanhf(rel_Lq[m]/2);
            } else if(j==degree-1) {
                for(int m=0; m<degree-1; m++)
                    new_val *= tanhf(rel_Lq[m]/2);
            } else {
                for(int m =0; m<j; m++)
                    new_val *= tanhf(rel_Lq[m]/2);
                for(int m =j+1; m<degree; m++)
                    new_val *= tanhf(rel_Lq[m]/2);
            }

            if(new_val >= 1) {
                //positicve, no need to change sign 
                sel_lr[j] = 2 * 19.07; //atanh(1) = infinite, set 19.07 as maximum value
                //printf("new_val %f >=1  ", new_val);
            } else if(new_val <= -1){
                sel_lr[j] = -2 * 19.07; //atanh(-1) = negative infinite, set - 19.07 as min value
                //printf("new_val %f <=-1  ", new_val);
            } else {
                sel_lr[j] = 2 * atanhf(new_val);
            }
        }
    } else if(num_zeros == 1) {
        // only one zero value, all sel_lr except zero_idx are zero
        memset(sel_lr, 0.0f, degree*sizeof(float));

        if(tanh_prod >= 1) {
            sel_lr[zero_idx] = 2 * 19.07;
            //printf("tanh_prod %f >=1  ", tanh_prod);
        } else if(tanh_prod <= -1) {
            sel_lr[zero_idx] = -2 * 19.07;
            //printf("tanh_prod %f <=-1  ", tanh_prod);
        } else {
            sel_lr[zero_idx] = 2 * atanhf(tanh_prod);
        }
    } else {
        //for num of zeros >=2, all sel_lr==0
        memset(sel_lr, 0.0f, degree*sizeof(float));
    }


    // update LQ immediately after updating R for this check node, so that the updated LQ can be used in the next check node in the same layer
    for (int j = 0; j < degree; j++) {
        LQ[column_list[j]] = rel_Lq[j] + sel_lr[j];
    }
    
}

int32_t layered_BP_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg) {
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

    //memcpy(LQ, LLr, sizeof(int16_t)*(config->N+2*Zc)); 
    //copy from LLr to LQ, and convert from int16 to float 
    //scaling by decoder_cfg->max_llr, scaling value
    //will greatly affect LDPC performance from almost all error to no error
    for(int i=0;i<config->N+2*Zc; i++) {
        LQ[i] = (float)LLr[i];
        LQ[i] = LQ[i] / (float)decoder_cfg->max_llr;
    }
       

    //clean Lr mem
    //memset(Lr, 0, sizeof(float)*non_zero_elements*Zc);
    for(int i=0; i< non_zero_elements*Zc; i++)
        Lr[i] = 0.0f;
    
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
                float *sel_lr = Lr + offset*Zc + zx*degree; // Pointer to the start of Lr for this check node (row)
                
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
                _BP_update(degree, rel_Lq, sel_lr, column_list, LQ);
                
            } // End of processing for this HBG row
        } // End of layered processing for this iteration
        
        // --- Early Termination (Syndrome Check) at the end of each iteration ---
        //calculate hard bits from LQ, and store in hardbits array, which is ceil(K/8) bytes, MSB first in each byte
        for (int i = 0; i < config->B; i++) {
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

