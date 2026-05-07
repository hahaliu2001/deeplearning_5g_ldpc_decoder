#ifndef _LDPC_DECODER_H_
#define _LDPC_DECODER_H_

//max_degree,bg1_total_non_null_elements, bg1_num_rows, bg1_num_cols, bg2_total_non_null_elements, bg2_num_rows, bg2_num_cols 
// = ldpc_c_parameters.get_ldpc_parameters()
#define BG1_NUM_ROWS 46
#define BG1_NUM_COLS 68
#define BG1_MAX_NON_NULL_ELEMENTS 316

#define BG2_NUM_ROWS 42
#define BG2_NUM_COLS 52
#define BG2_MAX_NON_NULL_ELEMENTS 197 

//BG1 has maximum degree 19, BG2 has maximum degree 10, we use 19 as the max degree for both to simplify the code
#define MAX_DEGREE 19 

//38.212 5.3.2 table 5.3.2-1 defines the maximum lifting size Zc 
#define MAX_ZC 384  

#define MAX_LLR_I8 127
#define MIN_LLR_I8 -128

#define MAX_LLR_I16 32767
#define MIN_LLR_I16 -32768

extern uint16_t BG1_HBG_offset_array[BG2_NUM_ROWS+1];
extern uint16_t BG1_HBG_array[BG2_MAX_NON_NULL_ELEMENTS];

extern uint16_t BG2_HBG_offset_array[BG2_NUM_ROWS+1];
extern uint16_t BG2_HBG_array[BG2_MAX_NON_NULL_ELEMENTS];

extern uint32_t calculate_crc(const uint8_t *data, int len, int crctype);

extern void get_LDPC_parameters(int16_t K, uint16_t *Zc, uint8_t **Vij_array);

#endif /* _LDPC_DECODER_H_ */