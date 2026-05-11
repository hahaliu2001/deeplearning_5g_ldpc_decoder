# -*- coding:utf-8 -*-
from scipy import io

import numpy as np
import pandas as pd

from py5gphy.ldpc import ldpc_info

def generate_GEO_LDPC_parameter():
    """
    the code is to generate all LDPC parameters that used in GEO LDPC spec
    """

    # supported information bit length and code rate list in GEO LDPC spec
    info_bit_length_list = [320,400,480,640,800,960,1280,3840,256,384,512,768]
    code_rate_list = [1/4, 1/5]

    GEO_LDPC = pd.DataFrame(columns=['Info Bits', 'code_rate', 'Base Graph','Out size', 'iLS', 'Kb','Zc', 'num_filling_bits', \
                                    'info punctured bits', 'Out info bits size', 'out parity bits size'])
    for B in info_bit_length_list:
        for code_rate in code_rate_list:
            _, LDPC_parameter = gen_LDPC_parameter(B, code_rate)
            # append a list to dataframe
            GEO_LDPC.loc[len(GEO_LDPC)] = LDPC_parameter
    
    return GEO_LDPC

def gen_LDPC_parameter(B, code_rate):
    """
    the code is used to generate LDPC parameters defined in 3GPP TS38.212 section 5.3.2
    it support only base graph 2, and code rate 1/4 or 1/5
    and it support rv(redundancy version) 0 only, which means no starting point shift for parity bits
    input:
        B: information bit length
        code rate: 1/4 or 1/5
    output:
        B: input bit size 
        code_rate: "1/4" or "1/5"),  int(BG), int(E), int(iLS), int(Kb), int(Zc), int(num_filter_bits), int(information_punctured_bits), int(information_bits_size), int(parity_bits_size)]    
        BG: base graph, for GEO LDPC spec, only support BG2
        E: input bit size / code rate
        iLS: lifting size
        Kb: refer to 3GPP TS38.212 section 5.2.2, the Kb is determined by the information bit length B
            this value is related to BG and B, and it is used to determine the base graph size Zc
        Zc: base graph size
        num_filling_bits: the filling bits size
        information_punctured_bits: the information punctured bits size, for BG2
        parity_punctured_bits: the parity punctured bits size, for BG2
    """
    
    assert code_rate in [1/4, 1/5], "code rate should be 1/4 or 1/5"
    assert B <= 3840, "information bit length should be less than or equal to 3840 for BG2"

    BG = 2 # only support base graph 2 for GEO LDPC spec

    # step 1: find the base graph size Zc and the lifting size iLS
    if B > 640:
        Kb = 10
    elif B > 560:
        Kb = 9
    elif B > 192:
        Kb = 8
    else:
        Kb = 6

    Zlist = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, \
            26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, \
            112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384]
    
    Zc = 0
    for v in Zlist:
       if v * Kb >= B:
        Zc = v
        break
    assert Zc != 0, "cannot find suitable Zc for the given information bit length B"

    # get iLS
    iLS = ldpc_info.find_iLS(Zc)

    H = ldpc_info.getH(Zc, bgn=2, iLS=iLS) #get H

    # get filling bits size
    K = 10*Zc # for BG2, the information bit length is always 10*Zc
    num_filling_bits = K - B

    # LDPC encoder output code length
    N = 50*Zc # for BG2, the code length is always 50*Zc
    
    #get output sequence information bits size and parity bits size
    information_punctured_bits = 2*Zc
    information_bits_size = B - information_punctured_bits
    
    # output bit size
    E = int(B/code_rate)
    parity_bits_size = E - information_bits_size
    
    if code_rate == 1/4:
        code_rate_str = "1/4"
    else:
        code_rate_str = "1/5"

    return [E, N, Zc, num_filling_bits, H],[int(B), code_rate_str, int(BG), int(E), int(iLS), int(Kb), int(Zc), int(num_filling_bits), int(information_punctured_bits), int(information_bits_size), int(parity_bits_size)]

def gen_BG1_HBG_idx_array():
  """provide array with size and index of HBG matrix for BG1, which is used for C implementation
  """

  curpath = "py5gphy/ldpc/tables/"
  filename = "BG1S0.mat1"
  matfile = io.loadmat(curpath + filename)
  BG = matfile["BG"]

  datalist = []
  offsetlist = []
  offset = 0
  
  for i in range(BG.shape[0]):
    data = BG[i, :]
    nonzero_idx = np.nonzero(data > -1)[0]
    datalist.extend(nonzero_idx.tolist() if hasattr(nonzero_idx, 'tolist') else list(nonzero_idx))
    
    offsetlist.append(offset)
    offset += len(nonzero_idx)

  with open("tables/BG1_HBG_table.txt", "w") as f:
    str1 = ", ".join(f"{x:3d}" for x in datalist)
    str1 = "{"+  str1 +"}"  
    f.write(str1 + "\n")
  
  with open("tables/BG1_HBG_offset_table.txt", "w") as f:
    str1 = ", ".join(f"{x:3d}" for x in offsetlist)
    str1 = "{"+  str1 +"}"  
    f.write(str1 + "\n")

def gen_BG2_HBG_idx_array():
  """provide array with size and index of HBG matrix for BG2, which is used for C implementation
  """

  curpath = "py5gphy/ldpc/tables/"
  filename = "BG2S0.mat1"
  matfile = io.loadmat(curpath + filename)
  BG = matfile["BG"]

  datalist = []
  offsetlist = []
  offset = 0
  
  for i in range(BG.shape[0]):
    data = BG[i, :]
    nonzero_idx = np.nonzero(data > -1)[0]
    datalist.extend(nonzero_idx.tolist() if hasattr(nonzero_idx, 'tolist') else list(nonzero_idx))
    
    offsetlist.append(offset)
    offset += len(nonzero_idx)

  with open("tables/BG2_HBG_table.txt", "w") as f:
    str1 = ", ".join(f"{x:3d}" for x in datalist)
    str1 = "{"+  str1 +"}"  
    f.write(str1 + "\n")
  
  with open("tables/BG2_HBG_offset_table.txt", "w") as f:
    str1 = ", ".join(f"{x:3d}" for x in offsetlist)
    str1 = "{"+  str1 +"}"  
    f.write(str1 + "\n")

def gen_BG2_HBG_Vij_array():
  """get HBG Vij array for each iLS
  """

  curpath = "py5gphy/ldpc/tables/"
  
  dataarray=[]
  for iLS in range(8):
    filename = f"BG2S{iLS}.mat1"
    matfile = io.loadmat(curpath + filename)
    BG = matfile["BG"]

    
    datalist = []
    for i in range(BG.shape[0]):
      data = BG[i, :]
      sel_data = data[data > -1]
      datalist.extend(sel_data.tolist() if hasattr(sel_data, 'tolist') else list(sel_data))

    dataarray.append(datalist)    
      
  with open(f"tables/BG2_HBG_Vij_array.txt", "w") as f:
    for datalist in dataarray:
      str1 = ", ".join(f"{x:3d}" for x in datalist)
      str1 = "{"+  str1 +"}"  
      f.write(str1 + "\n")


if __name__ == "__main__":
    
    GEO_LDPC = generate_GEO_LDPC_parameter()

    GEO_LDPC.to_excel("out/GEO_LDPC_parameters.xlsx", index=False)

    gen_BG2_HBG_idx_array()
    gen_BG2_HBG_Vij_array()
    pass
