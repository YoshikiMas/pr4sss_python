# -*- coding: utf-8 -*-
import numpy as np

def mysign(x):
    return np.exp(1j*np.angle(x))

def projection_consistencysum(x1, x2, mixture, stft, istft):

    f1 = istft(x1)
    f2 = istft(x2)
    d  = mixture-(f1+f2)
    y1 = stft(f1+d/2)
    y2 = stft(f2+d/2)
    return y1, y2

def prox_dis(x, amp, gamma):
    y = mysign(x)*(np.abs(x)-gamma/(amp)+np.sqrt((np.abs(x)-gamma/(amp))**2+4*gamma))/2;
    return y



def misi(spec1, spec2, mixture, stft, istft, maxiter=50):
    
    x1 = spec1;
    x2 = spec2;
    amp1 = np.abs(spec1);
    amp2 = np.abs(spec2);
    
    for i in range(maxiter):
        y1, y2 = projection_consistencysum(x1, x2, mixture, stft, istft)
        x1 = amp1*mysign(y1)
        x2 = amp2*mysign(y2)
    
    f1est = istft(x1);
    f2est = istft(x2);
    return f1est, f2est
    
    
def divmisi_ver1(spec1, spec2, mixture, stft, istft, maxiter=50, gamma=0.1, lm=1e4, epsilon=1e-15):

    x1 = spec1;
    x2 = spec2;
    amp1 = np.abs(spec1)+epsilon;
    amp2 = np.abs(spec2)+epsilon;
    
    z1 = spec1;
    z2 = spec2;
    u1 = np.zeros_like(z1);
    u2 = np.zeros_like(z2);
    
    for i in range(maxiter):
        
        # x-update
        x1 = prox_dis(z1-u1, amp1, gamma)
        x2 = prox_dis(z2-u2, amp2, gamma)
        
        # z-update
        v1 = x1+u1;
        v2 = x2+u2;
        [y1, y2] = projection_consistencysum(v1, v2, mixture, stft, istft);
        z1 = (v1+gamma*lm*y1)/(1+gamma*lm);
        z2 = (v2+gamma*lm*y2)/(1+gamma*lm);
        
        # u-update
        u1 = u1+x1-z1;
        u2 = u2+x2-z2;

    
    f1est = istft(prox_dis(z1-u1, amp1, gamma))
    f2est = istft(prox_dis(z2-u2, amp2, gamma))
    return f1est, f2est