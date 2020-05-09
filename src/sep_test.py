# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import glob
import numpy as np
import soundfile as sf

from utils.evals import sisdr
from utils.others import zero_pad, STFT, iSTFT
from utils.phase_reconstruction import misi, divmisi_ver1


## Parameters 
winlen = 1024
shift = 256
fs = 16000
epsilon = 1e-15

stft = STFT(winlen, shift)
istft = iSTFT(winlen, shift)


## Init
file_names = glob.glob('D:/wsj0-2mix/wav16k/min/tr/mix/*.wav')
mixture_base = 'D:/wsj0-2mix/wav16k/min/tr/mix/'
s1_base = 'D:/wsj0-2mix/wav16k/min/tr/s1/'
s2_base = 'D:/wsj0-2mix/wav16k/min/tr/s2/'

sisdr_tiam = []
sisdr_tpsm = []
sisdr_misi = []
sisdr_dmisi = []


def seprate_mixture(i):
    
    # load
    wav_name = file_names[i].replace('\\','/').split('/')[-1]
    fm, _ = sf.read(mixture_base+wav_name)
    f1, _ = sf.read(s1_base+wav_name)
    f2, _ = sf.read(s2_base+wav_name)
    
    
    # pre-process
    f1 = zero_pad(f1, winlen, shift)
    f2 = zero_pad(f2, winlen, shift)
    fm = zero_pad(fm, winlen, shift)
    
    # stft
    c1 = stft(f1)
    c2 = stft(f2)
    cm = stft(fm)
    
    # mask
    tiam1 = np.minimum(np.abs(c1)/(np.abs(cm)+epsilon), np.ones_like(np.abs(cm)))
    tiam2 = np.minimum(np.abs(c2)/(np.abs(cm)+epsilon), np.ones_like(np.abs(cm)))
    tpsm1 = np.minimum(np.maximum(np.cos(np.angle(cm)-np.angle(c1))*np.abs(c1)/(np.abs(cm)+epsilon), 0.),1.)
    tpsm2 = np.minimum(np.maximum(np.cos(np.angle(cm)-np.angle(c2))*np.abs(c2)/(np.abs(cm)+epsilon), 0.),1.)
                       
    # sepration
    masked1 = tiam1*cm
    masked2 = tiam2*cm
    f1_tiam = istft(masked1)
    f2_tiam = istft(masked2)
    f1_tpsm = istft(tpsm1*cm)
    f2_tpsm = istft(tpsm2*cm)
                       
    f1_misi, f2_misi = misi(masked1, masked2, fm, stft, istft)
    f1_dmisi, f2_dmisi = divmisi_ver1(masked1, masked2, fm, stft, istft, maxiter=500)
                       
    # evaluation
    sisdr_tiam.append((sisdr(f1, f1_tiam)+sisdr(f2, f2_tiam))/2)
    sisdr_tpsm.append((sisdr(f1, f1_tpsm)+sisdr(f2, f2_tpsm))/2)                    
    sisdr_misi.append((sisdr(f1, f1_misi)+sisdr(f2, f2_misi))/2)                   
    sisdr_dmisi.append((sisdr(f1, f1_dmisi)+sisdr(f2, f2_dmisi))/2)
      
## Main
for i in map(seprate_mixture, range(8)):
    pass


print('tIAM: '+str(np.median(sisdr_tiam)))
print('tPSM: '+str(np.median(sisdr_tpsm)))
print('MISI: '+str(np.median(sisdr_misi)))
print('div-MISI: '+str(np.median(sisdr_dmisi)))                        