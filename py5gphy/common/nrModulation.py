# -*- coding:utf-8 -*-
import numpy as np
import math

def nrModulate(inbits, modtype):
    """modulation refer to 38.211 5.1 Modulation mapper
    mod_data = nrModulate(inbits, modtype)
    """
    modtype = modtype.lower()
    assert modtype in ["pi/2-bpsk", "bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam"], "modulation type is incorrect"

    b = inbits.astype('f') #convert to float32
    N = b.size
    
    if modtype == "bpsk":
        mod_data = ((1-2*b) + 1j*(1-2*b))/math.sqrt(2)         
    elif modtype == "pi/2-bpsk":
        d1 = ((1-2*b) + 1j*(1-2*b))/math.sqrt(2)
        d2 = ((2*b-1) + 1j*(1-2*b))/math.sqrt(2)
        mod_data = d1
        mod_data[1::2] = d2[1::2]        
    elif modtype == "qpsk":
        assert N % 2 == 0, "length of databits must be multiple of 2"
        mod_data = ((1-2*b[0::2]) + 1j*(1-2*b[1::2]))/math.sqrt(2)        
    elif modtype == "16qam":
        assert N % 4 == 0, "length of databits must be multiple of 4"
        mod_data = ((1-2*b[0::4]) * (2 - (1-2*b[2::4])) 
               + 1j*(1-2*b[1::4]) * (2 - (1-2*b[3::4]))
                    )/math.sqrt(10)                
    elif modtype == "64qam":
        assert N % 6 == 0, "length of databits must be multiple of 6"
        mod_data = ((1-2*b[0::6]) * (4 - (1-2*b[2::6])*(2-(1-2*b[4::6]))) 
               + 1j*(1-2*b[1::6]) * (4 - (1-2*b[3::6])*(2-(1-2*b[5::6])))
                    )/math.sqrt(42)        
    elif modtype == "256qam":
        assert N % 8 == 0, "length of databits must be multiple of 8"
        mod_data = ((1-2*b[0::8]) * (8 - (1-2*b[2::8])*(4 - (1-2*b[4::8])*(2-(1-2*b[6::8]))))
               + 1j*(1-2*b[1::8]) * (8 - (1-2*b[3::8])*(4 - (1-2*b[5::8])*(2-(1-2*b[7::8]))))
                    )/math.sqrt(170)
    elif modtype == "1024qam":
        assert N % 10 == 0, "length of databits must be multiple of 10"
        mod_data = ((1-2*b[0::10]) * (16 - (1-2*b[2::10])*(8 - (1-2*b[4::10])*(4 - (1-2*b[6::10])*(2-(1-2*b[8::10])))))
               + 1j*(1-2*b[1::10]) * (16 - (1-2*b[3::10])*(8 - (1-2*b[5::10])*(4 - (1-2*b[7::10])*(2-(1-2*b[9::10])))))
                    )/math.sqrt(682)
    return mod_data

def get_mod_list(modtype):
    """return all modulation and bit sequence for given modtype
    """
    #get all modulation values for modtype
    Qm_list = {"pi/2-bpsk":1, "bpsk":1, "qpsk":2, "16qam":4, "64qam":6, "256qam":8, "1024qam":10}
    Qm = Qm_list[modtype]
    mod_array = np.zeros(2**Qm, 'c8')
    inbits_array = np.zeros((2**Qm,Qm),'i8')
    for m in range(2**Qm):
        if Qm == 1:
            inbits = [int(x) for x in list('{0:01b}'.format(m))]
        elif Qm == 2:
            inbits = [int(x) for x in list('{0:02b}'.format(m))]
        elif Qm == 4:
            inbits = [int(x) for x in list('{0:04b}'.format(m))]
        elif Qm == 6:
            inbits = [int(x) for x in list('{0:06b}'.format(m))]
        elif Qm == 8:
            inbits = [int(x) for x in list('{0:08b}'.format(m))]
        elif Qm == 10:
            inbits = [int(x) for x in list('{0:010b}'.format(m))]
        else:
            inbits = 0
        inbits_array[m,:] = np.array(inbits)
        mod_data = nrModulate(np.array(inbits), modtype)
        mod_array[m] = mod_data
    
    return mod_array,inbits_array

def get_oppisite_syms(inbits,modtype):
    #find Symbo list with opposite bit value for each bit,
    # for example, QAM16 symbol A with bits "1111" 
    #the symbol with first bit=0 and closest to symbol A is symbol with bits 101
    #the symbol with second bit=0 and closest to symbol A is symbol with its 1010
    #the symbol with third bit=0 and closest to symbol A is symbol with bits 101
    #the symbol with forth bit=0 and closest to symbol A is symbol with bits 110
    #the slection is related to QAM constellation
#["pi/2-bpsk", "bpsk", "qpsk", "16qam", "64qam", "256qam", "1024qam"]
    if modtype == "pi/2-bpsk":
        opposite_syms = nrModulate(1-inbits, modtype)
    elif modtype == "bpsk":
        opposite_syms = nrModulate(1-inbits, modtype)
    elif modtype == "qpsk":
        newbits = np.array([1-inbits[0],inbits[1],inbits[0],1-inbits[1]])
        opposite_syms = nrModulate(newbits, modtype)
    elif modtype == "16qam":
        newbits = np.array([
            1-inbits[0],inbits[1],0,inbits[3],
            inbits[0],1-inbits[1],inbits[2],0,
            inbits[0],inbits[1],1-inbits[2],inbits[3],
            inbits[0],inbits[1],inbits[2],1-inbits[3]
            ])
        opposite_syms = nrModulate(newbits, modtype)
    elif modtype == "64qam":
        newbits = np.array([
            1-inbits[0],inbits[1],0,inbits[3],1,inbits[5],
            inbits[0], 1-inbits[1],inbits[2],0, inbits[4],1,
            inbits[0],inbits[1],1-inbits[2],inbits[3],0,inbits[5],
            inbits[0], inbits[1],inbits[2],1-inbits[3], inbits[4],0,
            inbits[0],inbits[1],inbits[2],inbits[3],1-inbits[4],inbits[5],
            inbits[0], inbits[1],inbits[2],inbits[3], inbits[4],1-inbits[5]
            ])
        opposite_syms = nrModulate(newbits, modtype)
    elif modtype == "256qam":
        newbits = np.array([
            1-inbits[0],inbits[1],0,inbits[3],1,inbits[5],1,inbits[7],
            inbits[0], 1-inbits[1],inbits[2],0, inbits[4],1,inbits[6],1,
            inbits[0],inbits[1],1-inbits[2],inbits[3],0,inbits[5],1,inbits[7],
            inbits[0], inbits[1],inbits[2],1-inbits[3], inbits[4],0,inbits[6],1,
            inbits[0],inbits[1],inbits[2],inbits[3],1-inbits[4],inbits[5],0,inbits[7],
            inbits[0], inbits[1],inbits[2],inbits[3], inbits[4],1-inbits[5],inbits[6],0,
            inbits[0],inbits[1],inbits[2],inbits[3],inbits[4],inbits[5],1-inbits[6],inbits[7],
            inbits[0], inbits[1],inbits[2],inbits[3], inbits[4],inbits[5],inbits[6],1-inbits[7],
            ])
        opposite_syms = nrModulate(newbits, modtype)
    elif modtype == "1024qam":
        newbits = np.array([
            1-inbits[0],inbits[1],0,inbits[3],1,inbits[5],1,inbits[7],1,inbits[9],
            inbits[0],1-inbits[1],inbits[2],0,inbits[4],1,inbits[6],1,inbits[8],1,
            inbits[0],inbits[1],1-inbits[2],inbits[3],0,inbits[5],1,inbits[7],1,inbits[9],
            inbits[0],inbits[1],inbits[2],1-inbits[3],inbits[4],0,inbits[6],1,inbits[8],1,
            inbits[0],inbits[1],inbits[2],inbits[3],1-inbits[4],inbits[5],0,inbits[7],1,inbits[9],
            inbits[0],inbits[1],inbits[2],inbits[3],inbits[4],1-inbits[5],inbits[6],0,inbits[8],1,
            inbits[0],inbits[1],inbits[2],inbits[3],inbits[4],inbits[5],1-inbits[6],inbits[7],0,inbits[9],
            inbits[0],inbits[1],inbits[2],inbits[3],inbits[4],inbits[5],inbits[6],1-inbits[7],inbits[8],0,
            inbits[0],inbits[1],inbits[2],inbits[3],inbits[4],inbits[5],inbits[6],inbits[7],1-inbits[8],inbits[9],
            inbits[0],inbits[1],inbits[2],inbits[3],inbits[4],inbits[5],inbits[6],inbits[7],inbits[8],1-inbits[9],
            ])
        opposite_syms = nrModulate(newbits, modtype)
    
    return opposite_syms

if __name__ == "__main__":
    from scipy import io
    import os
    
    print("test nr modulation")
    from tests.common import test_nrModulation
    file_lists = test_nrModulation.get_testvectors()
    count = 1
    for filename in file_lists:
        print("count= {}, filename= {}".format(count, filename))
        count += 1
        test_nrModulation.test_nr_modulation(filename)
