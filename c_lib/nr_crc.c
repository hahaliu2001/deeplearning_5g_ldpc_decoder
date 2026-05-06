#include <stdio.h>
#include <stdint.h>

static uint32_t crc24a_table[256];
static uint32_t crc24b_table[256];
static uint32_t crc24c_table[256];
static uint16_t crc16_table[256];