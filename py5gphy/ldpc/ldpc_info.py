# -*- coding:utf-8 -*-
import numpy as np
from scipy import io

def get_cbs_info(B, bgn):
    """ get code block information, such as K, C,...
    
    input:
      B: number of code blck segment input bits
      
    output:
      C: num of CB, 
      cbz: Number of bits in each code block (excluding CB-CRC bits and filler bits)
      L: Number of parity bits in each code block
      F: Number of filler bits in each code block
      K: Number of bits in each code block (including CB-CRC bits and filler bits)
      Zc: minimum value of Z in all sets of lifting sizes in Table 5.3.2-1
    """
    #38.212 5.2.2
    # Get the maximum code block size
    if bgn == 1:
      Kcb = 8448
    else:
      Kcb = 3840
    
    #Get number of code blocks and length of CB-CRC coded block
    if B <= Kcb:
       L = 0
       C = 1
       Bd = B
    else:
       L = 24 # add '24B' CRC
       C = int(np.ceil(B/(Kcb - L)))
       Bd = B + C*L
    
    # Obtain the number of bits per code block (excluding CB-CRC bits)
    cbz = B // C
    #the spec doesn't clearly say B is dividable by C, but my understanding is that it should be
    #so I add below assert to check if it is true
    #result: has run around 20K ULSCH testcases and it is still true
    assert (B % C) == 0 

    #Get number of bits in each code block (excluding filler bits)
    Kd = Bd // C
    #same as above, looks like it should be true
    assert (Bd % C) == 0

    #Find the minimum value of Z in all sets of lifting sizes in 38.212
    # Table 5.3.2-1, denoted as Zc, such that Kb*Zc>=Kd
    if bgn == 1:
      Kb = 22
    else:
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

    for v in Zlist:
       if v * Kb >= Kd:
          Zc = v
          break
    
    #Get number of bits (including <NULL> filler bits) to be input to the LDPC
    # encoder
    if bgn == 1:
      K = 22*Zc
    else:
      K = 10*Zc
      
    return C, cbz, L, K-Kd, K, Zc


def find_iLS(Zc):
    """Find the set with index iLS in Table 5.3.2-1 which contains Zc
      """
    ZSets = [[2, 4, 8, 16, 32, 64, 128, 256],     #Set 0
        [3, 6, 12, 24, 48, 96, 192, 384],         #Set 1
        [5, 10, 20, 40, 80, 160, 320],            #Set 2
        [7, 14, 28, 56, 112, 224],                #Set 3
        [9,18, 36, 72, 144, 288],                 #Set 4
        [11, 22, 44, 88, 176, 352],               #Set 5
        [13, 26, 52, 104, 208],                   #Set 6
        [15, 30, 60, 120, 240]]                   #Set 7                
    
    for setid in range(8):
       if Zc in ZSets[setid]:
          return setid
    
    return 255 #error

def getH(Zc, bgn, iLS):
    """ refer to TS38.212 5.3.2, generate H matrix 
     for base graph 1:generate 46Zc X 68Zc matrix 
     for base graph 2: generate 42Zc X 52Zc matrix
    """

    # Get the matrix with base graph number 'bgn' and set number 'setIdx'.
    # The element of matrix V in the following is H_BG(i,j)*V(i,j), where
    # H_BG(i,j) and V(i,j) are defined in TS 38.212 5.3.2; if V(i,j) is not
    # defined in Table 5.3.2-2 or Table 5.3.2-3, the elements are -1.
    # V is 46 X 68 matrix for bgn1, 42 X 52 matrix for bgn2
    curpath = "py5gphy/ldpc/tables/"
    filename = f"BG{bgn}S{iLS}.mat1"
    matfile = io.loadmat(curpath + filename)
    BG = matfile["BG"]

    if bgn == 1:
       H = np.zeros((46*Zc,68*Zc),'i1')
       rowsize = 46
       colsize = 68
    else:
       H = np.zeros((42*Zc,52*Zc), 'i1')
       rowsize = 42
       colsize = 52

    for i in range(rowsize):
        for j in range(colsize):
           if BG[i,j] > -1: # not NULL bit
                Pij = BG[i,j] % Zc

                #circularly shift identity Zc X Zc matrix I to the right Pij times
                shiftI = np.zeros((Zc,Zc),'i1')
                for m in range(Zc):
                   #m is 1 in I matrix on both row and col, and need right shift by Pij to get new col location
                   shift_col_loc = (Pij+m) % Zc
                   shiftI[m, shift_col_loc] = 1
                
                #update H
                H[i*Zc:(i+1)*Zc,j*Zc:(j+1)*Zc] = shiftI
                   
    return H

def gen_ldpc_para(N, bgn):
    if bgn == 1:
        Zc = N // 66
        K = 22 * Zc
    else:
        Zc = N // 50
        K = 10 * Zc
    
    #step 1: Find the set with index iLS in Table 5.3.2-1 which contains Zc
    iLS = find_iLS(Zc)
    assert iLS < 8

    #step 2 : get H
    H = getH(Zc, bgn, iLS)

    return H, K, Zc

if __name__ == "__main__":
    #find_iLS(2)
    H = getH(4, 1, 0)
    pass
  
  