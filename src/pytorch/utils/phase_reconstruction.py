# -*- coding: utf-8 -*-
import torch
from torchaudio.functional import complex_norm


def mysign(x):
    return x.div_(complex_norm(x).add_(1e-16).unsqueeze(-1).expand_as(x))

def projection_consistencysum(x1, x2, mixture, stft, istft):

    f1 = istft(x1)
    f2 = istft(x2)
    d  = mixture-(f1+f2)
    y1 = stft(f1+d/2)
    y2 = stft(f2+d/2)
    return y1, y2

def misi(spec1, spec2, mixture, stft, istft, maxiter=50):
    
    x1 = spec1
    x2 = spec2
    amp1 = complex_norm(spec1)
    amp2 = complex_norm(spec2)
    
    for i in range(maxiter):
        y1, y2 = projection_consistencysum(x1, x2, mixture, stft, istft)
        x1 = amp1*mysign(y1)
        x2 = amp2*mysign(y2)
    
    f1est = istft(x1);
    f2est = istft(x2);
    return f1est, f2est

