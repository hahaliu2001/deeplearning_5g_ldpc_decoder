#include <stdint.h>

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

// One 256-entry lookup table for each supported NR CRC polynomial.
static uint32_t crc16_table[CRC_TABLE_SIZE];
static uint32_t crc24a_table[CRC_TABLE_SIZE];
static uint32_t crc24b_table[CRC_TABLE_SIZE];
static uint32_t crc24c_table[CRC_TABLE_SIZE];

// Tables are generated lazily on first use.
static int crc16_table_initialized = 0;
static int crc24a_table_initialized = 0;
static int crc24b_table_initialized = 0;
static int crc24c_table_initialized = 0;

// Runtime description of a CRC variant.
typedef struct {
    uint16_t width;      // Number of CRC bits: 16 or 24.
    uint32_t polynomial; // Generator polynomial without the implicit top bit.
    uint32_t top_bit;    // MSB of the active CRC register.
    uint32_t mask;       // Keeps the CRC register inside its width.
    uint32_t *table;     // Lookup table for this variant.
    int *initialized;    // Lazy-initialization flag for the table.
} crc_spec_t;

// Translate the public crctype value into the parameters used by the CRC engine.
static int get_crc_spec(int crctype, crc_spec_t *spec) {
    switch (crctype) {
        case CRC16_TYPE:
            spec->width = CRC16_WIDTH;
            spec->polynomial = CRC16_POLY;
            spec->top_bit = CRC16_TOP_BIT;
            spec->mask = CRC16_MASK;
            spec->table = crc16_table;
            spec->initialized = &crc16_table_initialized;
            return 1;
        case CRC24A_TYPE:
            spec->width = CRC24_WIDTH;
            spec->polynomial = CRC24A_POLY;
            spec->top_bit = CRC24_TOP_BIT;
            spec->mask = CRC24_MASK;
            spec->table = crc24a_table;
            spec->initialized = &crc24a_table_initialized;
            return 1;
        case CRC24B_TYPE:
            spec->width = CRC24_WIDTH;
            spec->polynomial = CRC24B_POLY;
            spec->top_bit = CRC24_TOP_BIT;
            spec->mask = CRC24_MASK;
            spec->table = crc24b_table;
            spec->initialized = &crc24b_table_initialized;
            return 1;
        case CRC24C_TYPE:
            spec->width = CRC24_WIDTH;
            spec->polynomial = CRC24C_POLY;
            spec->top_bit = CRC24_TOP_BIT;
            spec->mask = CRC24_MASK;
            spec->table = crc24c_table;
            spec->initialized = &crc24c_table_initialized;
            return 1;
        default:
            return 0;
    }
}

/*
 * Generate the byte-wise lookup table for one CRC variant.
 *
 * Each entry represents feeding one byte into the top of an otherwise empty CRC
 * register and shifting it through the polynomial. With the table in place, the
 * runtime update can process one byte at a time instead of one bit at a time.
 */
static void generate_crc_lookup_table(const crc_spec_t *spec) {
    uint16_t shift = spec->width - 8;

    for (int i = 0; i < CRC_TABLE_SIZE; i++) {
        uint32_t crc = (uint32_t)i << shift;

        for (int bit = 0; bit < 8; bit++) {
            if (crc & spec->top_bit) {
                crc = (crc << 1) ^ spec->polynomial;
            } else {
                crc <<= 1;
            }
        }

        spec->table[i] = crc & spec->mask;
    }
}

// Generate the lookup table for a CRC variant the first time it is needed.
static void ensure_crc_table(const crc_spec_t *spec) {
    if (*spec->initialized) {
        return;
    }

    generate_crc_lookup_table(spec);
    *spec->initialized = 1;
}

/*
 * Compute the CRC remainder with a generated lookup table.
 *
 * The update is MSB-first:
 *   1. XOR the next data byte with the current CRC top byte.
 *   2. Shift the CRC register left by one byte.
 *   3. XOR in the table entry selected by step 1.
 */
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

/*
 * Generate all CRC lookup tables supported by this module.
 *
 * This is optional; calculate_crc() also initializes tables lazily.
 */
void generate_all_crc_tables() {
    crc_spec_t spec;

    for (int crctype = CRC16_TYPE; crctype <= CRC24C_TYPE; crctype++) {
        if (get_crc_spec(crctype, &spec)) {
            ensure_crc_table(&spec);
        }
    }
}

/*
 * Calculate a 3GPP NR CRC over byte-aligned input data.
 *
 * See 3GPP TS 38.212 section 5.1 for the CRC definitions.
 *
 * crctype:
 *   0: CRC16  polynomial 0x1021
 *   1: CRC24A polynomial 0x864CFB
 *   2: CRC24B polynomial 0x800063
 *   3: CRC24C polynomial 0xB2B117
 *
 * Returns the CRC remainder, or 0 for an unsupported crctype.
 */
uint32_t calculate_crc(const uint8_t *data, int len, int crctype) {
    crc_spec_t spec;

    if (!get_crc_spec(crctype, &spec)) {
        return 0;
    }

    ensure_crc_table(&spec);
    return calculate_crc_from_spec(data, len, &spec);
}
