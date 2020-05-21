# -*- coding: utf-8 -*-
import torch
from torchaudio.functional import complex_norm


def mysign(x):
    return x.div_(complex_norm(x).add_(1e-15).unsqueeze(-1).expand_as(x))

def projection_consistencysum(x1, x2, mixture, stft, istft):

    f1 = istft(x1)
    f2 = istft(x2)
    d  = mixture-(f1+f2)
    y1 = stft(f1+d/2)
    y2 = stft(f2+d/2)
    return y1, y2

def projection_sum(f1, f2, mixture):
    
    d  = mixture-(f1+f2)
    y1 = f1+d/2
    y2 = f2+d/2
    return y1, y2

def misi(spec1, spec2, mixture, stft, istft, maxiter=5):
    
    x1 = spec1
    x2 = spec2
    amp1 = complex_norm(spec1)[...,None]
    amp2 = complex_norm(spec2)[...,None]
    
    for i in range(maxiter):
        y1, y2 = projection_consistencysum(x1, x2, mixture, stft, istft)
        x1 = amp1*mysign(y1)
        x2 = amp2*mysign(y2)
    
    f1est = istft(x1);
    f2est = istft(x2);
    return f1est, f2est

def prox_dis(x, amp, gamma):
    xabs = complex_norm(x).add_(1e-15)[...,None]
    y = mysign(x)*(xabs-gamma/(amp)+torch.sqrt((xabs-gamma/(amp))**2+4*gamma))/2
    return y

def prox_kl(x, amp, gamma):
    xabs = complex_norm(x).add_(1e-15)[...,None]
    y = mysign(x)*(xabs-gamma+torch.sqrt((xabs-gamma)**2+4*gamma*amp))/2
    return y

def divmisi_ver1(spec1, spec2, mixture, stft, istft, maxiter=5, gamma=1., lm=1e4):

    x1 = spec1
    x2 = spec2
    amp1 = complex_norm(spec1).add_(1e-15)[...,None]
    amp2 = complex_norm(spec2).add_(1e-15)[...,None]
    
    z1 = spec1;
    z2 = spec2;
    u1 = torch.zeros_like(z1);
    u2 = torch.zeros_like(z2);

    
    for i in range(maxiter):
        
        # x-update
        x1 = prox_dis(z1-u1, amp1, gamma)
        x2 = prox_dis(z2-u2, amp2, gamma)
        
        # z-update
        v1 = x1+u1
        v2 = x2+u2
        y1, y2 = projection_consistencysum(v1, v2, mixture, stft, istft)
        z1 = (v1+gamma*lm*y1)/(1+gamma*lm)
        z2 = (v2+gamma*lm*y2)/(1+gamma*lm)
        
        # u-update
        u1 = u1+x1-z1
        u2 = u2+x2-z2

    
    f1est = istft(prox_dis(z1, amp1, gamma))
    f2est = istft(prox_dis(z2, amp2, gamma))
    return f1est, f2est
    
def divmisi_ver2(spec1, spec2, mixture, stft, istft, maxiter=5, gamma=1., lm=1e4):

    x1 = spec1
    x2 = spec2
    amp1 = complex_norm(spec1).add_(1e-15)[...,None]
    amp2 = complex_norm(spec2).add_(1e-15)[...,None]
    
    z1, z2 = projection_sum(istft(spec1),
                                       istft(spec2),
                                       mixture
                                       )
    u1 = torch.zeros_like(spec1);
    u2 = torch.zeros_like(spec2);
    
    for i in range(maxiter):
        
        # x-update
        x1 = prox_dis(stft(z1)-u1, amp1, gamma)
        x2 = prox_dis(stft(z2)-u2, amp2, gamma)
        
        # z-update
        v1 = istft(x1+u1)
        v2 = istft(x2+u2)
        y1, y2 = projection_sum(v1, v2, mixture)
        z1 = (v1+2.*gamma*lm*y1)/(1+2.*gamma*lm)
        z2 = (v2+2.*gamma*lm*y2)/(1+2.*gamma*lm)
        
        # u-update
        u1 = u1+x1-stft(z1)
        u2 = u2+x2-stft(z2)

    
    f1est = istft(prox_dis(stft(z1)-u1, amp1, gamma))
    f2est = istft(prox_dis(stft(z2)-u2, amp2, gamma))
    return f1est, f2est