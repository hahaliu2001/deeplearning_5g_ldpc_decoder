#include <stdio.h>
#include <stdint.h>

static uint32_t crc24a_table[256];
static uint32_t crc24b_table[256];
static uint32_t crc24c_table[256];
static uint16_t crc16_table[256];

static int crc24a_table_initialized = 0;
static int crc24b_table_initialized = 0;
static int crc24c_table_initialized = 0;
static int crc16_table_initialized = 0;

void generate_crc_table(int crcpoly) {
    if (crcpoly == 0x864CFB && !crc24a_table_initialized) {
        // Generate CRC24A table
        for (int i = 0; i < 256; i++) {
            uint32_t crc = (uint32_t)i << 16;
            for (int j = 0; j < 8; j++) {
                if (crc & 0x800000) {
                    crc = (crc << 1) ^ crcpoly;
                } else {
                    crc <<= 1;
                }
            }
            crc24a_table[i] = crc & 0xFFFFFF;
        }
        crc24a_table_initialized = 1;
    } else if (crcpoly == 0x5D6DCB && !crc24b_table_initialized) {
        // Generate CRC24B table
        for (int i = 0; i < 256; i++) {
            uint32_t crc = (uint32_t)i << 16;
            for (int j = 0; j < 8; j++) {
                if (crc & 0x800000) {
                    crc = (crc << 1) ^ crcpoly;
                } else {
                    crc <<= 1;
                }
            }
            crc24b_table[i] = crc & 0xFFFFFF;
        }
        crc24b_table_initialized = 1;
    } else if (crcpoly == 0x328FA6 && !crc24c_table_initialized) {
        // Generate CRC24C table
        for (int i = 0; i < 256; i++) {
            uint32_t crc = (uint32_t)i << 16;
            for (int j = 0; j < 8; j++) {
                if (crc & 0x800000) {
                    crc = (crc << 1) ^ crcpoly;
                } else {
                    crc <<= 1;
                }
            }
            crc24c_table[i] = crc & 0xFFFFFF;
        }
        crc24c_table_initialized = 1;
    } else if (crcpoly == 0x1021 && !crc16_table_initialized) {
        // Generate CRC16 table
        for (int i = 0; i < 256; i++) {
            uint16_t crc = (