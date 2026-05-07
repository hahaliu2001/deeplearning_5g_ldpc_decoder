# -*- coding:utf-8 -*-
import os
import sys
from scipy import io

import numpy as np
import pandas as pd
sys.path.append(os.getcwd())    

from py5gphy.ldpc import ldpc_info

# the file is to generate LDPC tables and parameters that are used for 
# c-code 5G LDPC encoding and decoding, which are defined in 3GPP TS38.212 section 5.3.2

def get_ldpc_parameters():
    """
    the code is to get bg1 and bg2 ldpc parameters used for c-code 5G LDPC encoding and decoding, which are defined in 3GPP TS38.212 section 5.3.2
    the parameters include:
    - the maximum degree of variable node and check node for base graph 1 and base graph 2, which is used to determine the size of the messages passed between variable nodes and check nodes
    input:
        bgn: base graph number, should be 1 or 2
    output:
        max_degree: the maximum degree of variable node and check node for the given base graph number
    """
    
    # read BG1 and BG2 matrix from .mat file, which are generated from 3GPP TS38.212 section 5.3.2 table 5.3.2-1
    curpath = "py5gphy/ldpc/tables/"
    filename = f"BG1S0.mat1"
    matfile = io.loadmat(curpath + filename)
    BG1 = matfile["BG"]

    filename = f"BG2S0.mat1"
    matfile = io.loadmat(curpath + filename)
    BG2 = matfile["BG"]

    #get the number of non-null elements in BG for each line
    non_null_elements_per_line = np.sum(BG1 > -1, axis=1)
    #get the maximum degree of variable node and check node for base graph 1
    max_degree_bg1 = np.max(non_null_elements_per_line)
    bg1_total_non_null_elements = np.sum(BG1 > -1)
    bg1_num_rows, bg1_num_cols = BG1.shape
    
    #get the number of non-null elements in BG for each line
    non_null_elements_per_line = np.sum(BG2 > -1, axis=1)
    #get the maximum degree of variable node and check node for base graph 2
    max_degree_bg2 = np.max(non_null_elements_per_line)
    bg2_total_non_null_elements = np.sum(BG2 > -1)
    bg2_num_rows, bg2_num_cols = BG2.shape

    max_degree = max(max_degree_bg1, max_degree_bg2)
    return max_degree,bg1_total_non_null_elements, bg1_num_rows, bg1_num_cols, bg2_total_non_null_elements, bg2_num_rows, bg2_num_cols

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

def gen_BG1_HBG_Vij_array():
  """get HBG Vij array for each iLS
  """

  curpath = "py5gphy/ldpc/tables/"
  
  dataarray=[]
  for iLS in range(8):
    filename = f"BG1S{iLS}.mat1"
    matfile = io.loadmat(curpath + filename)
    BG = matfile["BG"]

    
    datalist = []
    for i in range(BG.shape[0]):
      data = BG[i, :]
      sel_data = data[data > -1]
      datalist.extend(sel_data.tolist() if hasattr(sel_data, 'tolist') else list(sel_data))

    dataarray.append(datalist)    
      
  with open(f"tables/BG1_HBG_Vij_array.txt", "w") as f:
    for datalist in dataarray:
      str1 = ", ".join(f"{x:3d}" for x in datalist)
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
    max_degree,bg1_total_non_null_elements, bg1_num_rows, bg1_num_cols, bg2_total_non_null_elements, bg2_num_rows, bg2_num_cols = get_ldpc_parameters()
    print(max_degree,bg1_total_non_null_elements, bg1_num_rows, bg1_num_cols, bg2_total_non_null_elements, bg2_num_rows, bg2_num_cols)

    gen_BG2_HBG_idx_array()
    gen_BG1_HBG_idx_array()
    gen_BG1_HBG_Vij_array()
    gen_BG2_HBG_Vij_array()

    pass