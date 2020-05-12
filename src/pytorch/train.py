# -*- coding: utf-8 -*-
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchaudio.functional import complex_norm

from data_utils.wsj02mix_dataset import WSJ02MixDataset
from utils.stft_related import STFT, iSTFT
from utils.pre_process import to_normlized_log
from models.bigru_separator import GRU2SPK
from utils.evals import msa_pit
from utils.visualization import result_show


# Init
random.seed(777)  
np.random.seed(777)  
torch.manual_seed(777)  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark=True


## Params
# dataset
dataset_base_tr = 'D:/wsj0-2mix/wav8k/min/tr'
dataset_base_cv = 'D:/wsj0-2mix/wav8k/min/cv'

# training
batch_size = 64
num_epoch = 300
lr = 4e-4

# STFTs
winlen = 256
shift = 64
siglen = winlen+shift*(400-5)

# model
input_dim = 129
output_dim = 129
hidden_dim = 600
num_layers = 3

# save
dir_name = '../../results/model/'
model_name = 'bigru_' + str(hidden_dim) + '_' + str(num_layers)
save_name =  dir_name + model_name + '_' + '{:0=3}.ckpt'

## STFT/iSTFT
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


## Model
model = GRU2SPK(input_dim, output_dim, hidden_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

## Training
tr_loss = []
cv_loss = []
start = time.time()


for epoch in range(num_epoch):
    
    # Training
    running_loss = []
    model.train()
    for i, (mix, s1, s2) in enumerate(tr_data_loader):
        
        cm, c1, c2 = [stft(x.to(device)) for x in [mix, s1, s2]]
        am, a1, a2 = [complex_norm(x) for x in [cm, c1, c2]]
        mask1, mask2 = model(to_normlized_log(am))
        loss = msa_pit(a1, a2, mask1*am, mask2*am)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())

    tr_loss.append(np.mean(running_loss))    
        
    
    # Validation
    running_loss = []
    model.eval()
    with torch.no_grad():
        for i, (mix, s1, s2) in enumerate(cv_data_loader):
            
            cm, c1, c2 = [stft(x.to(device)) for x in [mix, s1, s2]]
            am, a1, a2 = [complex_norm(x) for x in [cm, c1, c2]]
            mask1, mask2 = model(to_normlized_log(am))
            loss = msa_pit(a1, a2, mask1*am, mask2*am)
            running_loss.append(loss.item())
            
    cv_loss.append(np.mean(running_loss))
    
    
    # Post-processing
    print('epoch: {:0=3}'.format(epoch))
    print('computational time: {0}'.format(time.time()-start))
    print('tr loss: {0}'.format(tr_loss[-1]))
    print('cv loss: {0}'.format(cv_loss[-1]))
    
    result_show(dir_name + model_name + '_separated.png',
                a1[0, ...].detach().clone().to("cpu").numpy(),
                a2[0, ...].detach().clone().to("cpu").numpy(),
                (mask1*am)[0, ...].detach().clone().to("cpu").numpy(),
                (mask2*am)[0, ...].detach().clone().to("cpu").numpy(),
                )
    
    scheduler.step()
    if (epoch+1)%10 == 0:
        torch.save(model.state_dict(), save_name.format(epoch))
    

print('Finish')