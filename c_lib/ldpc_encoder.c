#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "nr_ldpc.h"

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