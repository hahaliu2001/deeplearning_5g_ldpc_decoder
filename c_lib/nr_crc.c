#include <stdio.h>
#include <stdint.h>

static uint32_t crc24a_table[256];
static uint32_t crc24b_table[256];
static uint32_t crc24c_table[256];
static uint32_t crc16_table[256];

static int crc24a_table_initialized = 0;
static int crc24b_table_initialized = 0;
static int crc24c_table_initialized = 0;
static int crc16_table_initialized = 0;

//generate the 256-entry lookup table for CRC24A, CRC24B, CRC24C, and CRC16 based on the specified polynomial
void generate_crc24_table(int crctype) {
    uint32_t crcpoly;

    //get crc polynomial based on type, 3gpp 38.212 5.1
    switch (crctype) {
        case 1: crcpoly = 0x864CFB; break; // CRC24A
        case 2: crcpoly = 0x800063; break; // CRC24B
        case 3: crcpoly = 0xB2B117; break; // CRC24C
        default: return; // Invalid type
    }

    //get crc table based on type
    uint32_t *crc_table;
    switch (crctype) {
        case 1: crc_table = crc24a_table; break;
        case 2: crc_table = crc24b_table; break;
        case 3: crc_table = crc24c_table; break;
        default: return; // Invalid type
    }

    // Check if the table is already initialized
    if(crctype==1 && crc24a_table_initialized) return;
    if(crctype==2 && crc24b_table_initialized) return;
    if(crctype==3 && crc24c_table_initialized) return;

    // Generate crc24 table
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i << 16;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x800000) {
                crc = (crc << 1) ^ crcpoly;
            } else {
                crc <<= 1;
            }
        }
        crc_table[i] = crc & 0xFFFFFF;
    }

    //set table initialized flag
    if(crctype==1) crc24a_table_initialized = 1;
    if(crctype==2) crc24b_table_initialized = 1;
    if(crctype==3) crc24c_table_initialized = 1;
    
    return;
}

//generate crc16 table based on polynomial 0x1021
void generate_crc16_table() {
    if(crc16_table_initialized) return; // Table already generated
    uint32_t poly = 0x1021;
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
        }
        crc16_table[i] = crc & 0xFFFF;
    }
    crc16_table_initialized = 1;
}

void generate_crc_table(int crctype) {
    if (crctype == 0) {
        generate_crc16_table();
    } else if (crctype >= 1 && crctype <= 3) {
        generate_crc24_table(crctype);
    }
}

// Generate all CRC tables at once
void generate_all_crc_tables() {
    generate_crc16_table(); // CRC16
    generate_crc24_table(1); // CRC24A
    generate_crc24_table(2); // CRC24B
    generate_crc24_table(3); // CRC24C
}

uint32_t calculate_crc(const uint8_t *data, int len, int crctype) {
    
    generate_crc_table(crctype); // Ensure the appropriate table is generated

    //get crc table
    uint32_t *crc_table;
    switch (crctype) {
        case 0: crc_table = crc16_table; break;
        case 1: crc_table = crc24a_table; break;
        case 2: crc_table = crc24b_table; break;
        case 3: crc_table = crc24c_table; break;
        default: return 0; // Invalid type
    }

    //crc16 encoding
    if (crctype==0) {
        uint32_t crc = 0;
        for (size_t i = 0; i < len; i++) {
            // Index = (Current MSB of CRC) XOR (Next Data Byte)
            uint8_t index = ((crc >> 8) ^ data[i]) & 0xFF;
            // Shift CRC left by 8 bits and XOR with table value
            crc = ((crc << 8) ^ crc_table[index]) & 0xFFFF;
        }
        return crc & 0xFFFF;
    }

    //crc24 encoding
    uint32_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        // Index = (Current MSB of CRC) XOR (Next Data Byte)
        uint8_t index = ((crc >> 16) ^ data[i]) & 0xFF;
        // Shift CRC left by 8 bits and XOR with table value
        crc = ((crc << 8) ^ crc_table[index]) & 0xFFFFFF;
    }

    return crc & 0xFFFFFF;
}
