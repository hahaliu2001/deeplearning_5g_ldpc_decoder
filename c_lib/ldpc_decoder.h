#ifndef _LDPC_DECODER_H_
#define _LDPC_DECODER_H_


#define BG2_NUM_ROWS 42
#define BG2_MAX_NON_NULL_ELEMENTS 197 

#define MAX_DEGREE 10

#define MAX_ZC 384

#define MAX_LLR 127
#define MIN_LLR -128

extern uint8_t BG2_HBG_offset_array[BG2_NUM_ROWS+1];
extern uint8_t BG2_HBG_array[BG2_MAX_NON_NULL_ELEMENTS];

extern uint32_t calculate_crc24a_fast(const uint8_t *data, size_t len);

extern void get_LDPC_parameters(int16_t K, uint16_t *Zc, uint8_t **Vij_array);

#endif /* _LDPC_DECODER_H_ */