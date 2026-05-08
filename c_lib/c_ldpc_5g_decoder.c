#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "ldpc_decoder.h"

/********* LDPC Decoder Main Function *********/
int ldpc_5g_decode(int16_t *LLr, uint8_t *hardbits, ldpc_in_config_t *inconfig, ldpc_decoder_cfg_t *decoder_config) {