#ifndef CUDA_NR_CRC_INTERNAL_H
#define CUDA_NR_CRC_INTERNAL_H

#include <stdint.h>

// Shared CRC constants for the CUDA CRC implementation.
//
// The numeric type IDs match the C/Python encoder paths:
//   0: CRC16, 1: CRC24A, 2: CRC24B, 3: CRC24C.
#define CRC_TABLE_SIZE 256

#define CRC16_TYPE 0
#define CRC24A_TYPE 1
#define CRC24B_TYPE 2
#define CRC24C_TYPE 3

#define CRC16_WIDTH 16
#define CRC24_WIDTH 24

#define CRC16_MASK 0xFFFF
#define CRC24_MASK 0xFFFFFF

#define CRC16_TOP_BIT 0x8000
#define CRC24_TOP_BIT 0x800000

#define CRC16_POLY 0x1021
#define CRC24A_POLY 0x864CFB
#define CRC24B_POLY 0x800063
#define CRC24C_POLY 0xB2B117

// Runtime view of one CRC variant.
//
// table is the normal byte-wise lookup table. word_contribution_table stores
// the same polynomial transformed for the CUDA 32-bit update path.
typedef struct {
    uint16_t width;
    uint32_t polynomial;
    uint32_t top_bit;
    uint32_t mask;
    const uint32_t *table;
    const uint32_t *word_contribution_table;
} crc_spec_t;

// Resolve the public crctype ID into static tables and polynomial parameters.
int cuda_crc_get_spec(int crctype, crc_spec_t *spec);

#endif
