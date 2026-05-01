# -*- coding:utf-8 -*-
import numpy as np
import math

def demod(insymbols,noise_var):
    """1024QAM demodulation
    """
    A = 1/math.sqrt(682)
    LLR = np.zeros(10*insymbols.size,dtype='f')
    LLR[0::10] = LLR_1024qam_bit_0_1(insymbols.real,noise_var,A)
    LLR[1::10] = LLR_1024qam_bit_0_1(insymbols.imag,noise_var,A)
    
    LLR[2::10] = LLR_1024qam_bit_2_3(insymbols.real,noise_var,A)
    LLR[3::10] = LLR_1024qam_bit_2_3(insymbols.imag,noise_var,A)

    LLR[4::10] = LLR_1024qam_bit_4_5(insymbols.real,noise_var,A)
    LLR[5::10] = LLR_1024qam_bit_4_5(insymbols.imag,noise_var,A)

    LLR[6::10] = LLR_1024qam_bit_6_7(insymbols.real,noise_var,A)
    LLR[7::10] = LLR_1024qam_bit_6_7(insymbols.imag,noise_var,A)

    LLR[8::10] = LLR_1024qam_bit_8_9(insymbols.real,noise_var,A)
    LLR[9::10] = LLR_1024qam_bit_8_9(insymbols.imag,noise_var,A)

    hardbits = np.array([0 if a>0 else 1 for a in LLR])
    return hardbits,LLR

def LLR_1024qam_bit_0_1(insym, noise_var,A):
    """ 1024QAM b0 and b1 LLR"""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -30*A:
            LLR[m] = 64*A*(r+15*A)/noise_var[m]
        elif r < -28*A:
            LLR[m] = 60*A*(r+14*A)/noise_var[m]
        elif r < -26*A:
            LLR[m] = 56*A*(r+13*A)/noise_var[m]
        elif r < -24*A:
            LLR[m] = 52*A*(r+12*A)/noise_var[m]
        elif r < -22*A:
            LLR[m] = 48*A*(r+11*A)/noise_var[m]
        elif r < -20*A:
            LLR[m] = 44*A*(r+10*A)/noise_var[m]
        elif r < -18*A:
            LLR[m] = 40*A*(r+9*A)/noise_var[m]
        elif r < -16*A:
            LLR[m] = 36*A*(r+8*A)/noise_var[m]
        elif r < -14*A:
            LLR[m] = 32*A*(r+7*A)/noise_var[m]
        elif r < -12*A:
            LLR[m] = 28*A*(r+6*A)/noise_var[m]
        elif r < -10*A:
            LLR[m] = 24*A*(r+5*A)/noise_var[m]
        elif r < -8*A:
            LLR[m] = 20*A*(r+4*A)/noise_var[m]
        elif r < -6*A:
            LLR[m] = 16*A*(r+3*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 12*A*(r+2*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 8*A*(r+1*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 4*A*r/noise_var[m]
        elif r < 4*A:
            LLR[m] = 8*A*(r-1*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 12*A*(r-2*A)/noise_var[m]
        elif r < 8*A:
            LLR[m] = 16*A*(r-3*A)/noise_var[m]
        elif r < 10*A:
            LLR[m] = 20*A*(r-4*A)/noise_var[m]
        elif r < 12*A:
            LLR[m] = 24*A*(r-5*A)/noise_var[m]
        elif r < 14*A:
            LLR[m] = 28*A*(r-6*A)/noise_var[m]
        elif r < 16*A:
            LLR[m] = 32*A*(r-7*A)/noise_var[m]
        elif r < 18*A:
            LLR[m] = 36*A*(r-8*A)/noise_var[m]
        elif r < 20*A:
            LLR[m] = 40*A*(r-9*A)/noise_var[m]
        elif r < 22*A:
            LLR[m] = 44*A*(r-10*A)/noise_var[m]
        elif r < 24*A:
            LLR[m] = 48*A*(r-11*A)/noise_var[m]
        elif r < 26*A:
            LLR[m] = 52*A*(r-12*A)/noise_var[m]
        elif r < 28*A:
            LLR[m] = 56*A*(r-13*A)/noise_var[m]
        elif r < 30*A:
            LLR[m] = 60*A*(r-14*A)/noise_var[m]
        else:
            LLR[m] = 64*A*(r-15*A)/noise_var[m]
        
    return LLR

def LLR_1024qam_bit_2_3(insym, noise_var,A):
    """ 1024QAM b2 and b3 LLR"""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -30*A:
            LLR[m] = 32*A*(r+23*A)/noise_var[m]
        elif r < -28*A:
            LLR[m] = 28*A*(r+22*A)/noise_var[m]
        elif r < -26*A:
            LLR[m] = 24*A*(r+21*A)/noise_var[m]
        elif r < -24*A:
            LLR[m] = 20*A*(r+20*A)/noise_var[m]
        elif r < -22*A:
            LLR[m] = 16*A*(r+19*A)/noise_var[m]
        elif r < -20*A:
            LLR[m] = 12*A*(r+18*A)/noise_var[m]
        elif r < -18*A:
            LLR[m] = 8*A*(r+17*A)/noise_var[m]
        elif r < -14*A:
            LLR[m] = 4*A*(r+16*A)/noise_var[m]
        elif r < -12*A:
            LLR[m] = 8*A*(r+15*A)/noise_var[m]
        elif r < -10*A:
            LLR[m] = 12*A*(r+14*A)/noise_var[m]
        elif r < -8*A:
            LLR[m] = 16*A*(r+13*A)/noise_var[m]
        elif r < -6*A:
            LLR[m] = 20*A*(r+12*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 24*A*(r+11*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 28*A*(r+10*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 32*A*(r+9*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 32*A*(-r+9*A)/noise_var[m]
        elif r < 4*A:
            LLR[m] = 28*A*(-r+10*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 24*A*(-r+11*A)/noise_var[m]
        elif r < 8*A:
            LLR[m] = 20*A*(-r+12*A)/noise_var[m]
        elif r < 10*A:
            LLR[m] = 16*A*(-r+13*A)/noise_var[m]
        elif r < 12*A:
            LLR[m] = 12*A*(-r+14*A)/noise_var[m]
        elif r < 14*A:
            LLR[m] = 8*A*(-r+15*A)/noise_var[m]
        elif r < 18*A:
            LLR[m] = 4*A*(-r+16*A)/noise_var[m]
        elif r < 20*A:
            LLR[m] = 8*A*(-r+17*A)/noise_var[m]
        elif r < 22*A:
            LLR[m] = 12*A*(-r+18*A)/noise_var[m]
        elif r < 24*A:
            LLR[m] = 16*A*(-r+19*A)/noise_var[m]
        elif r < 26*A:
            LLR[m] = 20*A*(-r+20*A)/noise_var[m]
        elif r < 28*A:
            LLR[m] = 24*A*(-r+21*A)/noise_var[m]
        elif r < 30*A:
            LLR[m] = 28*A*(-r+22*A)/noise_var[m]
        else:
            LLR[m] = 32*A*(-r+23*A)/noise_var[m]
    
    return LLR

def LLR_1024qam_bit_4_5(insym, noise_var,A):
    """ 1024QAM b4 and b5 LLR"""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -30*A:
            LLR[m] = 16*A*(r+27*A)/noise_var[m]
        elif r < -28*A:
            LLR[m] = 12*A*(r+26*A)/noise_var[m]
        elif r < -26*A:
            LLR[m] = 8*A*(r+25*A)/noise_var[m]
        elif r < -22*A:
            LLR[m] = 4*A*(r+24*A)/noise_var[m]
        elif r < -20*A:
            LLR[m] = 8*A*(r+23*A)/noise_var[m]
        elif r < -18*A:
            LLR[m] = 12*A*(r+22*A)/noise_var[m]
        elif r < -16*A:
            LLR[m] = 16*A*(r+21*A)/noise_var[m]
        elif r < -14*A:
            LLR[m] = 16*A*(-r-11*A)/noise_var[m]
        elif r < -12*A:
            LLR[m] = 12*A*(-r-10*A)/noise_var[m]
        elif r < -10*A:
            LLR[m] = 8*A*(-r-9*A)/noise_var[m]
        elif r < -6*A:
            LLR[m] = 4*A*(-r-8*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 8*A*(-r-7*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 12*A*(-r-6*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 16*A*(-r-5*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 16*A*(r-5*A)/noise_var[m]
        elif r < 4*A:
            LLR[m] = 12*A*(r-6*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 8*A*(r-7*A)/noise_var[m]
        elif r < 10*A:
            LLR[m] = 4*A*(r-8*A)/noise_var[m]
        elif r < 12*A:
            LLR[m] = 8*A*(r-9*A)/noise_var[m]
        elif r < 14*A:
            LLR[m] = 12*A*(r-10*A)/noise_var[m]
        elif r < 16*A:
            LLR[m] = 16*A*(r-11*A)/noise_var[m]
        elif r < 18*A:
            LLR[m] = 16*A*(-r+21*A)/noise_var[m]
        elif r < 20*A:
            LLR[m] = 12*A*(-r+22*A)/noise_var[m]
        elif r < 22*A:
            LLR[m] = 8*A*(-r+23*A)/noise_var[m]
        elif r < 26*A:
            LLR[m] = 4*A*(-r+24*A)/noise_var[m]
        elif r < 28*A:
            LLR[m] = 8*A*(-r+25*A)/noise_var[m]
        elif r < 30*A:
            LLR[m] = 12*A*(-r+26*A)/noise_var[m]
        else:
            LLR[m] = 16*A*(-r+27*A)/noise_var[m]
    
    return LLR

def LLR_1024qam_bit_6_7(insym, noise_var,A):
    """ 1024QAM b6 and b7 LLR"""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -30*A:
            LLR[m] = 8*A*(r+29*A)/noise_var[m]
        elif r < -26*A:
            LLR[m] = 4*A*(r+28*A)/noise_var[m]
        elif r < -24*A:
            LLR[m] = 8*A*(r+27*A)/noise_var[m]
        elif r < -22*A:
            LLR[m] = 8*A*(-r-21*A)/noise_var[m]
        elif r < -18*A:
            LLR[m] = 4*A*(-r-20*A)/noise_var[m]
        elif r < -16*A:
            LLR[m] = 8*A*(-r-19*A)/noise_var[m]
        elif r < -14*A:
            LLR[m] = 8*A*(r+13*A)/noise_var[m]
        elif r < -10*A:
            LLR[m] = 4*A*(r+12*A)/noise_var[m]
        elif r < -8*A:
            LLR[m] = 8*A*(r+11*A)/noise_var[m]
        elif r < -6*A:
            LLR[m] = 8*A*(-r-5*A)/noise_var[m]
        elif r < -2*A:
            LLR[m] = 4*A*(-r-4*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 8*A*(-r-3*A)/noise_var[m]
        elif r < 2*A:
            LLR[m] = 8*A*(r-3*A)/noise_var[m]
        elif r < 6*A:
            LLR[m] = 4*A*(r-4*A)/noise_var[m]
        elif r < 8*A:
            LLR[m] = 8*A*(r-5*A)/noise_var[m]
        elif r < 10*A:
            LLR[m] = 8*A*(-r+11*A)/noise_var[m]
        elif r < 14*A:
            LLR[m] = 4*A*(-r+12*A)/noise_var[m]
        elif r < 16*A:
            LLR[m] = 8*A*(-r+13*A)/noise_var[m]
        elif r < 18*A:
            LLR[m] = 8*A*(r-19*A)/noise_var[m]
        elif r < 22*A:
            LLR[m] = 4*A*(r-20*A)/noise_var[m]
        elif r < 24*A:
            LLR[m] = 8*A*(r-21*A)/noise_var[m]
        elif r < 26*A:
            LLR[m] = 8*A*(-r+27*A)/noise_var[m]
        elif r < 30*A:
            LLR[m] = 4*A*(-r+28*A)/noise_var[m]
        else:
            LLR[m] = 8*A*(-r+29*A)/noise_var[m]
    
    return LLR

def LLR_1024qam_bit_8_9(insym, noise_var,A):
    """ 1024QAM b8 and b9 LLR"""
    LLR = np.zeros(insym.size,dtype='f')
    for m in range(insym.size):
        r = insym[m]
        if r < -28*A:
            LLR[m] = 4*A*(r+30*A)/noise_var[m]
        elif r < -24*A:
            LLR[m] = 4*A*(-r-26*A)/noise_var[m]
        elif r < -20*A:
            LLR[m] = 4*A*(r+22*A)/noise_var[m]
        elif r < -16*A:
            LLR[m] = 4*A*(-r-18*A)/noise_var[m]
        elif r < -12*A:
            LLR[m] = 4*A*(r+14*A)/noise_var[m]
        elif r < -8*A:
            LLR[m] = 4*A*(-r-10*A)/noise_var[m]
        elif r < -4*A:
            LLR[m] = 4*A*(r+6*A)/noise_var[m]
        elif r < 0:
            LLR[m] = 4*A*(-r-2*A)/noise_var[m]
        elif r < 4*A:
            LLR[m] = 4*A*(r-2*A)/noise_var[m]
        elif r < 8*A:
            LLR[m] = 4*A*(-r+6*A)/noise_var[m]
        elif r < 12*A:
            LLR[m] = 4*A*(r-10*A)/noise_var[m]
        elif r < 16*A:
            LLR[m] = 4*A*(-r+14*A)/noise_var[m]
        elif r < 20*A:
            LLR[m] = 4*A*(r-18*A)/noise_var[m]
        elif r < 24*A:
            LLR[m] = 4*A*(-r+22*A)/noise_var[m]
        elif r < 28*A:
            LLR[m] = 4*A*(r-26*A)/noise_var[m]
        else:
            LLR[m] = 4*A*(-r+30*A)/noise_var[m]
    
    return LLR


if __name__ == "__main__":
    
    aaa=1