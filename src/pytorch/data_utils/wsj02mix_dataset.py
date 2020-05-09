# -*- coding: utf-8 -*-
import glob
import numpy as np
import torch
import torchaudio


class WSJ02MixDataset(torch.utils.data.Dataset):
    
    
    def __init__(self, dataset_base, siglen=None, dtype=torch.float32):
        file_names = glob.glob(dataset_base+'/mix/*.wav')
        wav_names = [fn.replace('\\','/').split('/')[-1] for fn in file_names]
        self.wav_names = wav_names
        self.mix_base = dataset_base + '/mix/'
        self.s1_base = dataset_base + '/s1/'
        self.s2_base = dataset_base + '/s2/'
        self.siglen = siglen
        self.dtype = dtype
          
       
    def __len__(self):
        return len(self.wav_names)
        
    def __getitem__(self, idx):
        wav_name = self.wav_names[idx]
        mix, _ = torchaudio.load(self.mix_base+wav_name)
        s1, _ = torchaudio.load(self.s1_base+wav_name)
        s2, _ = torchaudio.load(self.s2_base+wav_name)
        
        mix = mix[0,].to(self.dtype)
        s1 = s1[0,].to(self.dtype)
        s2 = s2[0,].to(self.dtype)
        
        if type(self.siglen) == int:
            mixlen = len(mix)
            
            if mixlen > self.siglen:
                start_idx = np.random.randint(len(mix)-self.siglen)
                mix = mix[start_idx:start_idx+self.siglen]
                s1 = s1[start_idx:start_idx+self.siglen]
                s2 = s2[start_idx:start_idx+self.siglen]
                
            else:
                mix = torch.cat([mix,
                                 torch.zeros(self.siglen-mixlen).to(self.dtype)
                                 ])
                s1 = torch.cat([s1,
                                torch.zeros(self.siglen-mixlen).to(self.dtype)
                                ])
                s2 = torch.cat([s2,
                                torch.zeros(self.siglen-mixlen).to(self.dtype)
                                ])                
        return mix, s1, s2