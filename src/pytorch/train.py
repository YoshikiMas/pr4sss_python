# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from data_utils.wsj02mix_dataset import WSJ02MixDataset
from utils.stft_related import STFT, iSTFT


# Init
random.seed(777)  
np.random.seed(777)  
torch.manual_seed(777)  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Params
dataset_base_tr = 'D:/wsj0-2mix/wav16k/min/tr'
dataset_base_cv = 'D:/wsj0-2mix/wav16k/min/cv'

batch_size = 512
num_epoch = 1

winlen = 512
shift = 128
siglen = 512+128*(400-1)
stft = STFT(shift,winlen)
istft = iSTFT(shift,winlen)


## Dataset
tr_dataset = WSJ02MixDataset(dataset_base_tr, siglen = siglen)
cv_dataset = WSJ02MixDataset(dataset_base_cv)
tr_data_loader = torch.utils.data.DataLoader(tr_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
cv_data_loader = torch.utils.data.DataLoader(cv_dataset,
                                             batch_size=1,
                                             shuffle=False)


# ToDO
# model
# criterion
# optimizer
# scheduler

for epoch in range(1):
    for i, (mix, s1, s2) in enumerate(tr_data_loader):
        
        cm, c1, c2 = [stft(x.to(device)) for x in [mix, s1, s2]]
        print((mix-istft(cm).to('cpu')).abs().flatten().mean().item())

print('Finish')