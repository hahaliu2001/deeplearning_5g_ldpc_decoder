#ifndef _LDPC_DECODER_H_
#define _LDPC_DECODER_H_

#include <stddef.h>
#include <stdint.h>

// Base-graph dimensions and non-null entry counts from 3GPP TS 38.212 5.3.2.
#define BG1_NUM_ROWS 46
#define BG1_NUM_COLS 68
#define BG1_MAX_NON_NULL_ELEMENTS 316

#define BG2_NUM_ROWS 42
#define BG2_NUM_COLS 52
#define BG2_MAX_NON_NULL_ELEMENTS 197 

// BG1 has maximum row degree 19 and BG2 has maximum row degree 10. Use the
// larger value for shared scratch buffers.
#define MAX_DEGREE 19 

// Maximum lifting size Zc from 3GPP TS 38.212 table 5.3.2-1.
#define MAX_ZC 384  

#define MAX_LLR_I8 127
#define MIN_LLR_I8 -128

#define MAX_LLR_I16 32767
#define MIN_LLR_I16 -32768

// 3GPP TS 38.212 table 5.3.2-1: lifting-size sets indexed by iLS.
typedef struct {
    uint16_t iLS0Zc[8];
    uint16_t iLS1Zc[8];
    uint16_t iLS2Zc[7];
    uint16_t iLS3Zc[6];
    uint16_t iLS4Zc[6];
    uint16_t iLS5Zc[6];
    uint16_t iLS6Zc[5];
    uint16_t iLS7Zc[5];
} table_5_3_2_1_t;

// External LDPC configuration supplied by the caller.
typedef struct {
    uint16_t bgn;     // Base graph number: 1 or 2.
    uint16_t A;       // Payload size in bits before CRC attachment.
    uint16_t crctype; // 0: CRC16, 1: CRC24A, 2: CRC24B, 3: CRC24C.
    uint16_t E;       // Number of rate-matched bits to transmit.
    uint16_t rv;      // Redundancy version: 0..3.
    uint16_t Qm;      // Modulation order: 1, 2, 4, 6, or 8.
    uint16_t HARQ_on; // 1 enables HARQ combining, 0 starts a fresh buffer.
} ldpc_in_config_t;

// Derived LDPC parameters used by encoding, decoding, rate matching, and
// rate recovery. gen_ldpc_parameters() fills this from ldpc_in_config_t.
typedef struct {
    // Caller-provided parameters copied from ldpc_in_config_t.
    uint16_t bgn;
    uint16_t A;
    uint16_t crctype;
    uint16_t E;
    uint16_t rv;
    uint16_t Qm;
    uint16_t HARQ_on;
    
    // Parameters derived from the input configuration.
    uint16_t B;   // Code-block size after CRC attachment.
    uint16_t Zc;  // Lifting size from table 5.3.2-1.
    uint16_t iLS; // Lifting-size set index.
    uint16_t K;   // Systematic code-block size including filler bits.
                  // K = 22*Zc for BG1, 10*Zc for BG2.
    uint16_t F;   // Number of filler bits: K - B.
    uint16_t N;   // Encoded length before puncturing: 66*Zc for BG1, 50*Zc for BG2.
    uint16_t k0;  // Circular-buffer start position for rate matching/recovery.

    // Base-graph table slices selected by bgn and iLS.
    uint16_t *Vij_array;
    uint16_t *HBG_array;
    uint16_t *HBG_offset_array;

} ldpc_config_t;

// LDPC decoder configuration.
typedef struct {
    uint16_t algo;           // 0: BP, 1: min-sum, 2: trained min-sum, 3: degree min-sum, 4: float min-sum.
    uint16_t max_iterations; // Maximum decoder iterations.
    uint16_t alpha;          // Q15 min-sum scale, e.g. 0.75 -> round(0.75*32767).
    uint16_t beta;           // Offset min-sum value in the same LLR scale as inputs.
    int16_t filling_bit_value; // LLR assigned to filler-bit positions.
    int16_t max_llr;         // Upper LLR clipping bound; also normalizes BP inputs.
    int16_t min_llr;         // Lower LLR clipping bound.
} ldpc_decoder_cfg_t;

// Compressed base-graph tables.
extern uint16_t BG1_HBG_offset_array[BG1_NUM_ROWS+1];
extern uint16_t BG1_HBG_array[BG1_MAX_NON_NULL_ELEMENTS];
extern uint16_t BG1_HBG_Vij_array[8][BG1_MAX_NON_NULL_ELEMENTS];

extern uint16_t BG2_HBG_offset_array[BG2_NUM_ROWS+1];
extern uint16_t BG2_HBG_array[BG2_MAX_NON_NULL_ELEMENTS];
extern uint16_t BG2_HBG_Vij_array[8][BG2_MAX_NON_NULL_ELEMENTS];

// CRC helper.
extern uint32_t calculate_crc(const uint8_t *data, int len, int crctype);

// Shared HARQ/rate-recovery buffer consumed by decoder kernels.
extern int16_t HARQ_memory[BG1_NUM_COLS*MAX_ZC]; 

// Parameter generation.
extern int gen_ldpc_parameters(ldpc_config_t *params, ldpc_in_config_t *inconfig );

// Decoder kernels.
extern int32_t layered_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg);
extern int32_t layered_degree_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg);
extern int32_t layered_degree_minsum_ldpc_decode_with_alpha(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg, const int16_t *degree_alpha_values);
extern int32_t layered_neural_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg);
extern int32_t layered_BP_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg);
extern int32_t layered_float_minsum_ldpc_decode(int16_t *LLr, uint8_t *hardbits, ldpc_config_t *config, ldpc_decoder_cfg_t *decoder_cfg);

// Rate recovery/matching.
extern void nr_ldpc_rate_recover_harq_combine(ldpc_config_t *config, int16_t filling_bit_value, int16_t *input_LLr);
extern void nr_ldpc_rate_match(uint8_t *inbits, uint8_t *outbytes,ldpc_config_t *config);

// Test-channel helper: unpack packed bits, add AWGN, and output int16_t LLRs.
extern int tx_rx_process(const uint8_t *inbytes, size_t num_bits, uint16_t Qm, float noise_power_db, int16_t *llr_out);
extern float dequantize_int16_to_llr(int16_t quantized_llr);

// Top-level codec entry points.
extern int32_t c_ldpc_5g_decode(int16_t *LLr, uint8_t *hardbits, ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config);
extern int32_t nr_ldpc_encode(uint8_t *in_bits, uint8_t *out_bits, ldpc_config_t *ldpc_config);
extern int32_t c_ldpc_5g_encode(uint8_t *infobytes, uint8_t *rm_bytes, ldpc_in_config_t *inconfig);
extern float dequantize_int16_to_llr(int16_t quantized_llr);
#endif /* _LDPC_DECODER_H_ */
