# -*- coding: utf-8 -*-
import numpy as np

def misign(x):
    return np.exp(1j*np.angle(x))


def projection_consistencysum(x1, x2, mixture, stft, istft)

    f1 = istft(x1)
    f2 = istft(x2)
    d  = mixture-(f1+f2)
    y1 = stft(f1+d/2)
    y2 = stft(f2+d/2)
    return y1, y2



def misi(spec1, spec2, mixture, stft, istft, maxiter=50):
    
    x1 = spec1;
    x2 = spec2;
    amp1 = np.abs(spec1);
    amp2 = np.abs(spec2);
    
    for i in range(maxiter):
        [y1, y2] = projection_consistencysum(x1, x2, mixture, STFT, iSTFT)
        x1 = abs1*mysign(y1)
        x2 = abs2*mysign(y2)
    
    f1est = istft(x1);
    f2est = istft(x2);
    return f1est, f2est
    