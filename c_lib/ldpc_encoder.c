#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

static uint8_t pc[4][MAX_ZC]; //parity bits
static uint8_t L[4][MAX_ZC];
static uint8_t Lsum[MAX_ZC];

/*LDPC encoder
in_bit: input bit stream of size B
out_bits: B input bits + F filling bits + w sequence, 
*/
int32_t nr_ldpc_encode(uint8_t *in_bits, uint8_t *out_bits, ldpc_config_t *ldpc_config){

    uint16_t num_rows;

    //read parameters from config
    uint16_t bgn = ldpc_config->bgn;
    uint16_t Zc = ldpc_config->Zc;
    uint16_t K = ldpc_config->K;
    uint16_t B = ldpc_config->B;
    uint16_t F = ldpc_config->F; //number of filling bit
    uint16_t Kb = K / Zc; //22 for bg1, 10 for bg2
    uint16_t *Vij_array = ldpc_config->Vij_array;
    uint16_t *HBG_array = ldpc_config->HBG_array;
    uint16_t *HBG_offset_array = ldpc_config->HBG_offset_array;
    
    if(bgn == 1){
        num_rows = BG1_NUM_ROWS;
    } else {
        num_rows = BG2_NUM_ROWS;
    }

    //first copy in_bit to out_bits, padding 0 for filling bit
    memcpy(out_bits, in_bits, B);
    memset(out_bits+B, 0, F);
    uint8_t *cseq = out_bits; //point to c sequence
    uint8_t *wseq = out_bits+K; //point to w sequence

    //ldpc encoding, refer to LDPC_encoder_optimization.pdf
    // calculate L = A @ c
    for(int HBG_row =0; HBG_row < 4; HBG_row++) {
        uint32_t offset = HBG_offset_array[HBG_row];
        uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
        //columns indicate the column index j for the non-null elements in this row, 
        //and Vij indicate the corresponding Vij value for each non-null element
        //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
        //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
        uint16_t *HBGcolumns = HBG_array + offset;
        uint16_t *Vij = Vij_array + offset;
        for (uint16_t zx = 0; zx < Zc; zx++){
            L[HBG_row][zx] = 0;
            for(uint32_t j=0; j < degree; j++) {
                int16_t HBG_col_idx = HBGcolumns[j];                
                if(HBG_col_idx < Kb) {
                    //HBG_col_idx < Kb indicate information bits
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    L[HBG_row][zx] += cseq[column];
                }
                
            }
            L[HBG_row][zx] = L[HBG_row][zx] % 2;
        }
    }

    //add four L together and save to Lsum
    for (uint16_t zx = 0; zx < Zc; zx++){
        Lsum[zx] = (L[0][zx]+L[1][zx]+L[2][zx]+L[3][zx]) % 2;
    }
    
    /* calculate pc[0]-pc[3] values. These are first four Zc of parity bits value
    corresponding to bg1 HBGcolumn 22- 25 and bg2 HBGcolumn 10-13. 
    in below matrix, I1,I2,I3,I4 are Zc by Zc circular permutation matrix I(Pi,j)
    I is Zc by Zc identity matrix, 0 is zero matrix */
    /*from 38.212 Table 5.3.2-2:LDPC base graph 1 we get below B matrix ( row index 0-3 column index 22-25)*/
    /*BG1 B matrix format = [[I1,I,0,0]
                             [I2,I,I,0]
                             [0, 0,I,I]
                             [I1,0,0,I]]*/
    /*from 38.212 Table 5.3.2-3: LDPC base graph 2 we get below B matrix ( row index 0-3 column index 22-25)*/
    /*BG2 B matrix format = [[I3,I,0,0]
                             [0 ,I,I,0]
                             [I4,0,I,I]
                             [I3,0,0,I]]*/         
    
    /* we can see that: let S = B @ pc[0:3], then add four value together, the result = I2@pc[0] or I4@pc[0]*/
    //pc[0] is the first Zc parity bits and = Lsum right shift by Pij

    //let's get Pij value
    uint16_t shift0, shift1; 
    if(bgn==1) {
        shift0 = *(Vij_array+HBG_offset_array[0]+17); //HBGrow 0 offset 17 is I1 Pij
        shift1 = *(Vij_array+HBG_offset_array[1]+16); //HBGrow 1 offset 16 is I2 Pij
    }else {
        shift0 = *(Vij_array+HBG_offset_array[0]+6); //HBGrow 0 offset 6 is I3 Pij
        shift1 = *(Vij_array+HBG_offset_array[2]+5); //HBGrow 2 offset 5 is I4 Pij
    }

    //cal pc[0]
    for (uint16_t zx = 0; zx < Zc; zx++) {
        pc[0][(shift1 + zx) % Zc] = Lsum[zx];
    }

    //pc[1] = L[0] + I1@pc[0]
    //pc[3] = L[3] + I1@pc[0]
    for (uint16_t zx = 0; zx < Zc; zx++) {
        pc[1][zx] = (L[0][zx] + pc[0][(shift0 + zx) % Zc]) % 2;
        pc[3][zx] = (L[3][zx] + pc[0][(shift0 + zx) % Zc]) % 2;
    }

    if(bgn == 1){
        //pc[2] = L[2]+pc[3]
        for (uint16_t zx = 0; zx < Zc; zx++) {
            pc[2][zx] = (L[2][zx] + pc[3][zx]) % 2;
        }
    } else {
        //pc[2] = L[2]+pc[3]
        for (uint16_t zx = 0; zx < Zc; zx++) {
            pc[2][zx] = (L[1][zx] + pc[1][zx]) % 2;
        }
    }

    //write pc to wseq, first four group of parity bits with each group size Zc
    for (uint16_t zx = 0; zx < Zc; zx++) {
        wseq[zx] = pc[0][zx];
        wseq[1*Zc + zx] = pc[1][zx];
        wseq[2*Zc + zx] = pc[2][zx];
        wseq[3*Zc + zx] = pc[3][zx];
    }

    /*additionaL PARITY bits = C @[c,pc]*/
    
    for(int HBG_row = 4; HBG_row < num_rows; HBG_row++) {
        uint32_t offset = HBG_offset_array[HBG_row];
        uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
        //columns indicate the column index j for the non-null elements in this row, 
        //and Vij indicate the corresponding Vij value for each non-null element
        //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
        //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
        uint16_t *HBGcolumns = HBG_array + offset;
        uint16_t *Vij = Vij_array + offset;
        
        for (uint16_t zx = 0; zx < Zc; zx++){
            uint16_t ps = 0;
            for(uint32_t j=0; j < degree; j++) {
                int16_t HBG_col_idx = HBGcolumns[j];
                
                if(HBG_col_idx < Kb+4) {
                    //HBG_col_idx < Kb+4 indicate information bits and first 4 parity bits
                    uint16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    ps += cseq[column];
                }
                
            }
            //write ps to wseq
            wseq[zx + Zc*HBG_row] = ps % 2;
        }
    }

    
    return 0;
}

int32_t bak_nr_ldpc_encode(uint8_t *in_bits, uint8_t *out_bits, ldpc_config_t *ldpc_config){

    uint16_t num_rows;

    //read parameters from config
    uint16_t bgn = ldpc_config->bgn;
    uint16_t Zc = ldpc_config->Zc;
    uint16_t K = ldpc_config->K;
    uint16_t B = ldpc_config->B;
    uint16_t F = ldpc_config->F; //number of filling bit
    uint16_t Kb = K / Zc; //22 for bg1, 10 for bg2
    uint16_t *Vij_array = ldpc_config->Vij_array;
    uint16_t *HBG_array = ldpc_config->HBG_array;
    uint16_t *HBG_offset_array = ldpc_config->HBG_offset_array;
    
    if(bgn == 1){
        num_rows = BG1_NUM_ROWS;
    } else {
        num_rows = BG2_NUM_ROWS;
    }

    //first copy in_bit to out_bits, padding 0 for filling bit
    memcpy(out_bits, in_bits, B);
    memset(out_bits+B, 0, F);
    uint8_t *cseq = out_bits; //point to c sequence
    uint8_t *wseq = out_bits+K; //point to w sequence

    //ldpc encoding, refer to LDPC_encoder_optimization.pdf
    uint8_t L[4];
    uint8_t pc[4];
    uint8_t ps[BG1_NUM_ROWS - 4];
    for (uint16_t zx = 0; zx < Zc; zx++){
        //first 4 row is to calculate L
        for(int HBG_row =0; HBG_row < 4; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            //columns indicate the column index j for the non-null elements in this row, 
            //and Vij indicate the corresponding Vij value for each non-null element
            //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
            //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            L[HBG_row] = 0;
            //calculate A @ c
            for(uint32_t j=0; j < degree; j++) {
                int16_t HBG_col_idx = HBGcolumns[j];
                if(HBG_col_idx < Kb) {
                    //this is c bits
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    L[HBG_row] = (L[HBG_row]+cseq[column]) % 2;
                }
                
            }
        }

        //second calculate pc1-pc4
        pc[0] = (L[0] + L[1] + L[2] + L[3]) % 2;
        pc[1] = (L[1] + L[2] + L[3]) % 2;
        pc[3] = (L[0] + L[1] + L[2]) % 2;   
        if(bgn==1) {
            pc[2] = (L[0] + L[1]) % 2;
        } else {
            pc[2] = (L[2] + L[3]) % 2;
        }

        //write pc to wseq
        wseq[zx + Zc * 0] = pc[0];
        wseq[zx + Zc * 1] = pc[0];
        wseq[zx + Zc * 2] = pc[0];
        wseq[zx + Zc * 3] = pc[0];

        //additionaL PARITY bits pa
        for(int HBG_row =4; HBG_row < num_rows; HBG_row++) {
            uint32_t offset = HBG_offset_array[HBG_row];
            uint32_t degree = HBG_offset_array[HBG_row + 1] - offset;
            //columns indicate the column index j for the non-null elements in this row, 
            //and Vij indicate the corresponding Vij value for each non-null element
            //[HBGcolumns:HBGcolumns+degree-1] are the HBG column indices for the non-null elements in this row, 
            //and [Vij:Vij+degree-1] are the corresponding Vij values for these non-null elements
            uint16_t *HBGcolumns = HBG_array + offset;
            uint16_t *Vij = Vij_array + offset;

            ps[HBG_row-4] = 0;
            for(uint32_t j=0; j < degree; j++) {
                int16_t HBG_col_idx = HBGcolumns[j];
                //C@c + D@pc
                if(HBG_col_idx < Kb + 4) {
                    int16_t shift = Vij[j];
                    int16_t column = (HBG_col_idx * Zc + ((shift + zx) % Zc));
                    ps[HBG_row-4] = (ps[HBG_row-4]+cseq[column]) % 2;
                }
                
            }        

            //write ps to wseq
            wseq[zx + Zc*HBG_row] = ps[HBG_row-4];
        }

    }
    return 0;
}