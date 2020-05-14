# -*- coding: utf-8 -*-
import torch
import torchaudio
from scipy.signal import hann


class STFT():
    
    def __init__(self, hop_length=None, win_length=None, device='cpu'):
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(hann(win_length)).to(torch.float32).to(device)
        
    def __call__(self, x):
        return torch.stft(x, self.win_length, self.hop_length, self.win_length,
                          window=self.window)
    
class iSTFT():
    
    def __init__(self, hop_length=None, win_length=None, siglen=None,
                 device='cpu'):
        self.hop_length = hop_length
        self.win_length = win_length
        self.siglen = siglen
        self.window = torch.from_numpy(hann(win_length)).to(torch.float32).to(device)
        
    def __call__(self, x):
        return torchaudio.functional.istft(x, self.win_length, self.hop_length,
                                           self.win_length, length=self.siglen,
                                           window=self.window)


if __name__ == '__main__':
    siglen = 512+128*(400-1)
    s = torch.randn(2,siglen)
    stft = STFT(128, 512)
    istft = iSTFT(128, 512, siglen)
    r = s-istft(stft(s))
    error = r.abs().median().item()
    print('reconstruction error: ' + str(error))
    
    