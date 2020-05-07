# -*- coding: utf-8 -*-
import numpy as np
import librosa

def zero_pad(x, winlen, shift):
    lf  = len(x)
    T   = int(np.ceil(lf-winlen)/shift) + 1
    lf2 = winlen + T*shift
    x   = np.concatenate((x,np.zeros(lf2+shift-lf,)), axis = 0)
    return x


class STFT():
    
    
    def __init__(self, winlen, shift, center=True):
        self.winlen = winlen
        self.shift = shift
        self.center = center
        
    def __call__(self, x):
        return librosa.core.stft(x, self.winlen, hop_length=self.shift, center=self.center)
        
        
        
class iSTFT():
    
    
    def __init__(self, winlen, shift, center=True):
        self.winlen = winlen
        self.shift = shift
        self.center = center
        
    def __call__(self, x):
        return librosa.core.istft(x, self.shift, self.winlen, center=self.center)