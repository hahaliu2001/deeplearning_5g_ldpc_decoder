#include <stdio.h>
#include <stdlib.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nr_ldpc.h"

//maximum rate matching outbut bits size = 100PRB*12sc*14symbol*8(256QAM)*4(layers)
#define MAX_E (100*12*14*8*4)
static uint8_t ek[MAX_E];
/*5g ldpc rate matching
dn: input sequence with size = N
ek: rat matching output with size = E/8 bytes
*/
void nr_ldpc_rate_match(uint8_t *inbits, uint8_t *outbytes,ldpc_config_t *config) {
    //read parameters from config
    uint16_t E = config->E;
    uint16_t N = config->N;
    uint16_t k0 = config->k0;
    uint16_t Zc = config->Zc;
    uint16_t B = config->B;
    uint16_t Qm = config->Qm;
    uint16_t K = config->K;

    uint8_t *outptr = outbytes;
    //rate match input need bypass first 2*Zc data
    uint8_t *rm_in = inbits + 2*Zc;

    // bit selection 38.212 5.4.2.1
    int k=0;
    int j = 0;
    while(k < E) {
        int pos = (k0+j) % N;
        if(pos < B-2*Zc || pos >= K-2*Zc) {
            //not filling bit
            ek[k++] = rm_in[pos];
        }
        j++;
    }
    // bit interleaving 38.212 5.4.2.2
    int bit_pos = 8;
    uint8_t dbyte = 0;
    for(j=0; j<E/Qm; j++) {
        for(int i=0; i< Qm; i++) {
            dbyte += ek[i*E/Qm+j] << (bit_pos-1);
            bit_pos--;
            if(bit_pos == 0) {
                //write byte to out bufer, reset dbyte and bit_pos
                *outptr++ = dbyte;
                dbyte = 0;
                bit_pos = 8;
            }
        }
    }

    //if E % 8 !=0. residul byte
    if(E % 8) {
        *outptr = dbyte;
    }

}