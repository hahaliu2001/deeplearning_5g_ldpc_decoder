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

/*ldpc parameters used for ldpc encoder/decoder and rate matching/rate recovery*/
typedef struct 
{
    /* input parameters for ldpc */ 
    uint16_t bgn; // base graph number, 1 or 2
    uint16_t A;   // transport block size in bits without CRC
    uint16_t crctype; // crc type, [0,1,2,3] for [crc16, crc24a, crc24b, crc24c]
    uint16_t E;   // number of bits to be transmitted after rate matching
    uint16_t rv;  // redundancy version, [0,1,2,3]
    uint16_t Qm;    // modulation order, [1,2,4,6,8] for [QPSK, 16QAM, 64QAM, 256QAM]
    uint16_t HARQ_on; // HARQ on/off, 1 for on, 0 for off
    
    /* parameters derived from input */
    uint16_t B; // number of bits in the code block after CRC 
    uint16_t Zc; // lifting size, table 5.3.2-1
    uint16_t iLS; // lifting size index, table 5.3.2-1
    uint16_t K; // LDPC code block size in bits including filling bits and CRC bits,
                // K = 22Zc for BG1, 10Zc for BG2
    uint16_t F; //number of filling bits, F = K - B
    uint16_t N; //number of bits after LDPC encoding, N = 66Zc for BG1, 50Zc for BG2
    uint16_t k0; // rate matching starting point, k0 = (A + 24) / Qm for HARQ on, k0 = 0 for HARQ off

    uint16_t *Vij_array; //point to BG2_Vij_array or BG1_Vij_array based on bgn, iLS
    uint16_t *HBG_array; //point to BG2_HBG_array or BG1_HBG_array based on bgn
    uint16_t *HBG_offset_array; // point to BG2_HBG_offset_array or BG1_HBG_offset_array based on bgn

} ldpc_config_t;

/*ldpc decoder parameters*/
typedef struct 
{
    uint16_t algo; //decoding algorithm, 0 for sum-product algorithm, 1 for min-sum algorithm
    uint16_t max_iterations; //maximum number of iterations for the decoding algorithm
    uint16_t alpha; //scaling factor for min-sum algorithm, in fixed point with 8 fractional bits, e.g., alpha = 0.75 -> 192
    uint16_t beta; //offset for offset min-sum algorithm, in fixed point with 8 fractional bits, e.g., beta = 0.25 -> 64
    int16_t filling_bit_value; //LLR value of the filling bits, expected to be a large positive value
    int16_t max_llr; //maximum LLR value for clipping, expected to be a large positive value
    int16_t min_llr; //minimum LLR value for clipping, expected to be a large negative value
    uint8_t *decoded_bits_array; //output decoded bits array, length = K-F
} ldpc_decoder_cfg_t;

extern uint16_t BG1_HBG_offset_array[BG2_NUM_ROWS+1];
extern uint16_t BG1_HBG_array[BG2_MAX_NON_NULL_ELEMENTS];
extern uint16_t BG1_HBG_Vij_array[8][BG2_MAX_NON_NULL_ELEMENTS];

extern uint16_t BG2_HBG_offset_array[BG2_NUM_ROWS+1];
extern uint16_t BG2_HBG_array[BG2_MAX_NON_NULL_ELEMENTS];
extern uint16_t BG2_HBG_Vij_array[8][BG2_MAX_NON_NULL_ELEMENTS];

extern uint32_t calculate_crc(const uint8_t *data, int len, int crctype);

extern void get_LDPC_parameters(int16_t K, uint16_t *Zc, uint8_t **Vij_array);

#endif /* _LDPC_DECODER_H_ */