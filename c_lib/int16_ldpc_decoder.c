#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "ldpc_decoder.h"

/*define the arrays used for LDPC decoder*/
//LQ is the array for storing the LLRs for variable nodes, and its size is 52*384, which is the maximum number of variable nodes in the code. The actual number of variable nodes used in a specific code will be determined by the code parameters (K and N).
//LQ is initialized with 2Zc zero + (K-2Zc) LLr + (10Zc-K) padding + (N-(K-2Zc)) LLr + (52Zc-N) zero
static int16_t LQ[52*MAX_ZC]; // LLRs for variable nodes

//checking nodes to variable nodes, 
static int16_t Lr[BG2_MAX_NON_NULL_ELEMENTS*MAX_ZC]; // Check-to-Variable messagees

static int16_t rel_Lq[MAX_DEGREE]; // Relative LLRs for variable nodes

static int16_t sign_list[MAX_DEGREE]; // Sign list for the variable nodes connected to a check node

static uint16_t column_list[MAX_DEGREE]; // Column indices for the variable nodes connected to a check node

static void int16_fill_LLr_to_LQ(const int8_t *LLr, int16_t K,int16_t E, int16_t N, int16_t Zc, int16_t filling_LLr);
static int16_t _do_parity_check(int16_t *LQ, uint16_t Zc,uint8_t *Vij_array);
/*
 * Layered Min-Sum Decoder (Fixed-point int16)
 * LLr: Posterior LLRs (size N, int16_t)
 * hardbits" deoded hardbits of size ceil(K/8) bytes (uint8_t), MSB first in each byte
 * K: Number of information bits
 * N: Codeword length (N > K) before rate recovery
 * max_iter: Maximum number of iterations to perform
 
 */
int32_t int16_layered_min_sum_decoder(int8_t *LLr, uint8_t *hardbits, int16_t K, int16_t E, 
    int16_t max_iter,int16_t alpha,int16_t beta,int16_t filling_LLr, int16_t en_crc_check) 
{
    uint16_t Zc;
    uint8_t *Vij_array; 
    uint32_t offset, degree;
    uint8_t *HBGcolumns, *Vij;
    int16_t column;
    int16_t HBG_col_idx;
    int16_t shift;
    int16_t *sel_lr;
    int16_t min1, min2,new_min1, new_min2;
    int16_t min1_idx;
    int16_t abs_val;
    
    uint16_t N; // Codeword length after rate recovery, which is the actual number of variable nodes in the code

    //int16_t alpha = 0.7 * 32767; // Scaling factor for normalized min-sum approximation, scaled by 128 for fixed-point representation
    //int8_t beta = 0.3 * 128; // Offset for offset min-sum approximation, scaled by 128 for fixed-point representation
    //int16_t beta=0;

    /* get parameters, return Zc,and Vij_array*/
    get_LDPC_parameters(K, &Zc, &Vij_array);

    N = 52*Zc; // Codeword length after rate recovery, which is the actual number of variable nodes in the code, determined by Zc and the structure of the parity-check matrix

    /*step 1: rate recovery and initialization LQ*/
    int16_fill_LLr_to_LQ(LLr, K, E, N, Zc, filling_LLr);
            
    memset(Lr, 0, sizeof(Lr)); // Initialize Lr to zero

    /*step 2: layered LDPC decoder*/
    for (int iter = 0; iter < max_iter; iter++) {
        // --- Layered Processing ---
        for (int HBG_row = 0; HBG_row < BG2_NUM_ROWS; HBG_row++) {
            //get offset and degree for this HBG row
            offset = BG2_HBG_offset_array[HBG_row];
            degree = BG2_HBG_offset_array[HBG_row + 1] - offset;
            
            //columns indicate the column index j for the non-null elements in this row, 
            //and Vij indicate the corresponding Vij value for each non-null element
            //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
            //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
            HBGcolumns = BG2_HBG_array + offset;
            Vij = Vij_array + offset;

            //each HBG rows index contain Zc row, so we need to process Zc rows for each HBG row
            for (int zx = 0; zx < Zc; zx++) {
                sel_lr = Lr + offset*Zc + zx*degree; // Pointer to the start of Lr for this check node (row)
                
                //calculate relative LLRs for the variable nodes connected to this check node
                for (int j = 0; j < degree; j++) {
                    HBG_col_idx = HBGcolumns[j]; //HBG column index for HBG row i and non-null element j    
                    shift = Vij[j];
                    column = (HBG_col_idx * Zc + ((shift + zx) % Zc)) ; // Variable node index for this check node
                    column_list[j] = column; // Store the column index for later use in updating LQ
                    rel_Lq[j] = LQ[column] - sel_lr[j]; // Relative LLR for this variable node
                }

                //update Lr using min-sum algorithm

                //first find the minimum and second minimum absolute values among rel_Lq, 
                //and also calculate the overall sign
                min1 = 32767;
                min2 = 32767;
                min1_idx = -1;
                int sign = 1;
                for (int j = 0; j < degree; j++) {
                    if(rel_Lq[j] >=0) {
                        sign_list[j] = 1;
                        abs_val = rel_Lq[j];
                    } else {
                        sign_list[j] = -1;
                        abs_val = -rel_Lq[j];
                    }
                    sign *= sign_list[j]; // Overall sign is the product of the signs of the connected variable nodes

                    if (abs_val < min1) {
                        min2 = min1;
                        min1 = abs_val;
                        min1_idx = j;
                    } else if (abs_val < min2) {
                        min2 = abs_val;
                        
                    }
                }

                //adjust min1 and min2 for normalized min-sum approximation and offset min-sum approximation
                int32_t tmp1,tmp2;
                //tmp1 = ((int16_t)alpha * (int16_t)((min1-beta)>0? min1-beta : 0)); // Apply scaling and offset for min-sum approximation
                //tmp2 = ((int16_t)alpha * (int16_t)((min2-beta)>0? min2-beta : 0)); // Apply scaling and offset for min-sum approximation
                tmp1 = (int32_t)alpha *(int32_t)(min1);
                tmp2 = (int32_t)alpha *(int32_t)(min2);
                new_min1 = (int16_t)((tmp1+(1<<14))>>15);
                new_min2 = (int16_t)((tmp2+(1<<14))>>15);
                //new_min1 = min1; // For standard min-sum, no adjustment
                //new_min2 = min2; // For standard min-sum, no adjustment

                //update Lr for this check node
                for (int j = 0; j < degree; j++) {
                    int16_t new_val = (j == min1_idx) ? new_min2 : new_min1; // Min-Sum update
                    new_val = sign *sign_list[j] * new_val; // Apply overall sign
                    sel_lr[j] = new_val; // Update R
                
                    // update LQ immediately after updating R for this check node, so that the updated LQ can be used in the next check node in the same layer
                    int16_t new_LQ = (int16_t)rel_Lq[j] + (int16_t)sel_lr[j]; // LQ = Relative LLR + new R
                    // Saturate to int8
                    int16_t sat_LQ;
                    sat_LQ = (int16_t)new_LQ;
                                                            
                    LQ[column_list[j]] = sat_LQ;
                } // End of processing for this check node (row)
            } // End of processing for this HBG row
        } // End of layered processing for this iteration

        // --- Early Termination (Syndrome Check) at the end of each iteration ---
        if(en_crc_check) {
            //calculate hard bits from LQ, and store in hardbits array, which is ceil(K/8) bytes, MSB first in each byte
            for (int i = 0; i < K; i++) {
                hardbits[i/8] = (hardbits[i/8] << 1) | (LQ[i] < 0 ? 1 : 0); // MSB first
            }

            //crc check
            if(calculate_crc24a_fast(hardbits, K/8) == 0) {
                return 0; // Converged successfully
            }
        }
        else {
            if(_do_parity_check(LQ, Zc, Vij_array)== 0) { // If parity check fails, continue to the next iteration, otherwise return success
                return 0; // Converged successfully
            }
            
        }

    }

    return 1; // Failed to converge within max_iter
}

static int16_t _do_parity_check(int16_t *LQ, uint16_t Zc,uint8_t *Vij_array) {
    uint8_t *HBGcolumns, *Vij;

    //perform parity check using hard bits calculated from LQ and the parity-check matrix H
    //return 1 if parity check pass, otherwise return 0
    for (int HBG_row = 0; HBG_row < BG2_NUM_ROWS; HBG_row++) {
        uint16_t offset = BG2_HBG_offset_array[HBG_row];
        uint16_t degree = BG2_HBG_offset_array[HBG_row + 1] - offset;
        HBGcolumns = BG2_HBG_array + offset;
        Vij = Vij_array + offset;

        for (int zx = 0; zx < Zc; zx++) {
            int parity_sum = 0;
            for (int j = 0; j < degree; j++) {
                uint8_t shift = Vij[j];
                uint16_t column = HBGcolumns[j] * Zc + ((shift + zx) % Zc);
                parity_sum ^= (LQ[column] < 0) ? 1 : 0; // XOR for parity check
            }
            if (parity_sum != 0) {
                return 1; // Parity check failed
            }
        }
    }
    
    return 0; // All parity checks passed
}

static void int16_fill_LLr_to_LQ(const int8_t *LLr, int16_t K,int16_t E, int16_t N, int16_t Zc, int16_t filling_LLr) {
    /*normalize LLr and then fill to LQ memory
    input:
        LLr: soft bits
        K: information bits size including crc size
        E: bits size after rate matching which is the size of input LLr data
        N: 52*Zc
        Zc: lifting size, maxium = 3840 for BG2
        filling_LLr: this is filling bit LLr. filling bits value are known to be 0, so filling_LLr should be larger than max(LLr)
    */
   int8_t max_v, abs_val, shift;

    /*step 0: normalize input LLr data, in my test, if the LQ value satuate to MAX_LLR, then LDPC decoder prformance is poor
    LLr data need normalize to make peak value = 128/8
    */

    for(int16_t i=0;i<E;i++) {
        abs_val = abs(LLr[i]);
        max_v = (max_v > abs_val) ? max_v : abs_val;
    }
    /*
    if max_v >= 01xxxxxx, LLr right shift by 3
    else if max_v >= 001xxxxx, LLr right shift by 2
    else if max_v >= 0001xxxx, LLr right shift by 1
    else do nothing
    */
    if((max_v >> 6) & 1)
        shift = 3;
    else if((max_v >> 5) & 1)
        shift = 2;
    else if((max_v >> 4) & 1)
        shift = 1;
    else if((max_v >> 3) & 1)
        shift = 0;
    else if((max_v >> 2) & 1)
        shift = -1;
    else if((max_v >> 1) & 1)
        shift = -2;
    else 
        shift = -3;
    
    shift = 0;
    //first 2Zc information bits is not transmitted, fill with Zero
    //memset(LQ , 0, 2*Zc); 
    for(int m=0; m<2*Zc; m++)
        LQ[m] = 0;
    
    //fill the information LLr
    if(shift >= 0) {
        for(int m=2*Zc; m<K; m++)
            LQ[m] = (int16_t)(LLr[m - 2*Zc] >> shift);
    }
    else {
        for(int m=2*Zc; m<K; m++)
            LQ[m] = (int16_t)(LLr[m - 2*Zc] << -shift);
    }
    
    // fill filling bits
    if(10*Zc > K)
        for(int m=K; m<10*Zc; m++)
            LQ[m] = (int16_t)(filling_LLr);
    
    //filling parity LLr
    if(shift >= 0) {
        for(int m=10*Zc; m<10*Zc+(E - (K - 2*Zc)); m++)
            LQ[m] = (int16_t)(LLr[(m - 10*Zc) + (K - 2*Zc)] >> shift);
    }
    else {
        for(int m=10*Zc; m<10*Zc+(E - (K - 2*Zc)); m++)
            LQ[m] = (int16_t)(LLr[(m - 10*Zc) + (K - 2*Zc)] << -shift);
    }
    
    //filling zero for not transmistted parity bits
    if(N > 10*Zc+(E - (K - 2*Zc)))
        for(int m=10*Zc+(E - (K - 2*Zc)); m<N; m++)
            LQ[m] = 0;

}

