#include <stdint.h>

#include "nr_crc_internal.h"

typedef struct {
    uint32_t value[CRC_TABLE_SIZE];
} crc_lookup_table_t;

typedef struct {
    uint32_t value[4u * CRC_TABLE_SIZE];
} crc_word_contribution_table_t;

static constexpr crc_lookup_table_t make_crc_lookup_table(uint16_t width,
    uint32_t polynomial, uint32_t top_bit, uint32_t mask) {
    crc_lookup_table_t table = {};
    uint16_t shift = width - 8;

    for (int i = 0; i < CRC_TABLE_SIZE; i++) {
        uint32_t crc = (uint32_t)i << shift;

        for (int bit = 0; bit < 8; bit++) {
            if (crc & top_bit) {
                crc = (crc << 1) ^ polynomial;
            } else {
                crc <<= 1;
            }
        }

        table.value[i] = crc & mask;
    }

    return table;
}

static constexpr crc_word_contribution_table_t
make_crc_word_contribution_table(const crc_lookup_table_t &crc_table,
    uint16_t width, uint32_t mask) {
    crc_word_contribution_table_t table = {};

    for (int value = 0; value < CRC_TABLE_SIZE; value++) {
        table.value[value] = crc_table.value[value] & mask;
    }

    for (int remaining = 1; remaining < 4; remaining++) {
        for (int value = 0; value < CRC_TABLE_SIZE; value++) {
            uint32_t crc =
                table.value[(remaining - 1) * CRC_TABLE_SIZE + value];
            uint8_t index = (uint8_t)((crc >> (width - 8u)) & 0xFFu);
            table.value[remaining * CRC_TABLE_SIZE + value] =
                ((crc << 8) ^ crc_table.value[index]) & mask;
        }
    }

    return table;
}

static constexpr crc_lookup_table_t crc16_table =
    make_crc_lookup_table(CRC16_WIDTH, CRC16_POLY, CRC16_TOP_BIT, CRC16_MASK);
static constexpr crc_lookup_table_t crc24a_table =
    make_crc_lookup_table(CRC24_WIDTH, CRC24A_POLY, CRC24_TOP_BIT, CRC24_MASK);
static constexpr crc_lookup_table_t crc24b_table =
    make_crc_lookup_table(CRC24_WIDTH, CRC24B_POLY, CRC24_TOP_BIT, CRC24_MASK);
static constexpr crc_lookup_table_t crc24c_table =
    make_crc_lookup_table(CRC24_WIDTH, CRC24C_POLY, CRC24_TOP_BIT, CRC24_MASK);

static constexpr crc_word_contribution_table_t crc16_word_contribution_table =
    make_crc_word_contribution_table(crc16_table, CRC16_WIDTH, CRC16_MASK);
static constexpr crc_word_contribution_table_t crc24a_word_contribution_table =
    make_crc_word_contribution_table(crc24a_table, CRC24_WIDTH, CRC24_MASK);
static constexpr crc_word_contribution_table_t crc24b_word_contribution_table =
    make_crc_word_contribution_table(crc24b_table, CRC24_WIDTH, CRC24_MASK);
static constexpr crc_word_contribution_table_t crc24c_word_contribution_table =
    make_crc_word_contribution_table(crc24c_table, CRC24_WIDTH, CRC24_MASK);

int cuda_crc_get_spec(int crctype, crc_spec_t *spec) {
    switch (crctype) {
        case CRC16_TYPE:
            spec->width = CRC16_WIDTH;
            spec->polynomial = CRC16_POLY;
            spec->top_bit = CRC16_TOP_BIT;
            spec->mask = CRC16_MASK;
            spec->table = crc16_table.value;
            spec->word_contribution_table = crc16_word_contribution_table.value;
            return 1;
        case CRC24A_TYPE:
            spec->width = CRC24_WIDTH;
            spec->polynomial = CRC24A_POLY;
            spec->top_bit = CRC24_TOP_BIT;
            spec->mask = CRC24_MASK;
            spec->table = crc24a_table.value;
            spec->word_contribution_table = crc24a_word_contribution_table.value;
            return 1;
        case CRC24B_TYPE:
            spec->width = CRC24_WIDTH;
            spec->polynomial = CRC24B_POLY;
            spec->top_bit = CRC24_TOP_BIT;
            spec->mask = CRC24_MASK;
            spec->table = crc24b_table.value;
            spec->word_contribution_table = crc24b_word_contribution_table.value;
            return 1;
        case CRC24C_TYPE:
            spec->width = CRC24_WIDTH;
            spec->polynomial = CRC24C_POLY;
            spec->top_bit = CRC24_TOP_BIT;
            spec->mask = CRC24_MASK;
            spec->table = crc24c_table.value;
            spec->word_contribution_table = crc24c_word_contribution_table.value;
            return 1;
        default:
            return 0;
    }
}

static uint32_t calculate_crc_from_spec(const uint8_t *data, int len,
    const crc_spec_t *spec) {
    uint32_t crc = 0;
    uint16_t shift = spec->width - 8;

    for (int i = 0; i < len; i++) {
        uint8_t index = (uint8_t)(((crc >> shift) ^ data[i]) & 0xFF);
        crc = ((crc << 8) ^ spec->table[index]) & spec->mask;
    }

    return crc & spec->mask;
}

void generate_all_crc_tables() {
    // Tables are generated at compile time and statically initialized.
}

uint32_t calculate_crc(const uint8_t *data, int len, int crctype) {
    crc_spec_t spec;

    if (!cuda_crc_get_spec(crctype, &spec)) {
        return 0;
    }

    return calculate_crc_from_spec(data, len, &spec);
}
