# -*- coding: utf-8 -*-
import glob
import numpy as np
import torch
import torchaudio


class VoicebankDemandDataset(torch.utils.data.Dataset):
    
    
    def __init__(self, dataset_base, train=True, siglen=None, dtype=torch.float32):
        if train:
            file_names = np.sort(glob.glob(dataset_base+'/clean_trainset_wav/*.wav'))
            wav_names = [fn.replace('\\','/').split('/')[-1] for fn in file_names]
            self.wav_names = wav_names
            self.clean_base = dataset_base + '/clean_trainset_wav/'
            self.noisy_base = dataset_base + '/noisy_trainset_wav/'
            
        else:
            file_names = np.sort(glob.glob(dataset_base+'/clean_testset_wav/*.wav'))
            wav_names = [fn.replace('\\','/').split('/')[-1] for fn in file_names]
            self.wav_names = wav_names
            self.clean_base = dataset_base + '/clean_testset_wav2/'
            self.noisy_base = dataset_base + '/noisy_testset_wav2/'
        self.siglen = siglen
        self.dtype = dtype
          
       
    def __len__(self):
        return len(self.wav_names)
        
    def __getitem__(self, idx):
        wav_name = self.wav_names[idx]
        s, _ = torchaudio.load(self.clean_base+wav_name)
        x, _ = torchaudio.load(self.noisy_base+wav_name)
        
        s = s[0,].to(self.dtype)
        x = x[0,].to(self.dtype)
        
        if type(self.siglen) == int:
            xlen = len(x)
            
            if xlen > self.siglen:
                start_idx = np.random.randint(len(s)-self.siglen)
                s = s[start_idx:start_idx+self.siglen]
                x = x[start_idx:start_idx+self.siglen]
                
            else:
                s = torch.cat([s, torch.zeros([self.siglen-xlen], dtype=self.dtype)])
                x = torch.cat([x, torch.zeros([self.siglen-xlen], dtype=self.dtype)])
        return s, x-s, x
    
if __name__ == '__main__':
    
    path = 'D:/Voicebank_DEMAND'  # Please chnage to ~~~/Voicebank_DEMAND
    
    # Train
    dataset = VoicebankDemandDataset(path, train=True, siglen=16000)
    s, n, x = dataset[0]
    print(len(dataset) == 11572)
    print(len(s) == 16000 & len(x) == 16000)
    
    # Test
    dataset = VoicebankDemandDataset(path, train=False, siglen=None)
    s, n, x = dataset[0]
    print(len(dataset) == 824)
    print(len(s) == len(x))
    print(type(len(s)) == int)
    
    


