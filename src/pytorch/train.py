# -*- coding: utf-8 -*-
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
import re
import yaml

import torch
import torch.optim as optim
from torchaudio.functional import complex_norm
from torch.autograd import detect_anomaly

from utils.dict_struct import Struct
from data_utils.wsj02mix_dataset import WSJ02MixDataset
from utils.stft_related import STFT, iSTFT
from utils.pre_process import to_normlized_log
from models import separator
from utils.evals import msa_pit
from utils.visualization import result_show


# Init
random.seed(777)  
np.random.seed(777)  
torch.manual_seed(777)  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


## Params
parser = ArgumentParser(description='Training script for Deep speech sep.')
parser.add_argument('dir_name', type=str)
args = parser.parse_args()

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

with open(args.dir_name+'/config.yml') as f:
    config = Struct(vars(args), **yaml.load(f, Loader=loader))

config.save_name =  config.dir_name + '/model_{:0=3}.ckpt'


## STFT/iSTFT
stft = STFT(config.shift, config.winlen, device=device)
istft = iSTFT(config.shift, config.winlen, device=device)


## Dataset
tr_dataset = WSJ02MixDataset(config.dataset_base_tr, siglen=config.siglen)
cv_dataset = WSJ02MixDataset(config.dataset_base_cv)
tr_data_loader = torch.utils.data.DataLoader(tr_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True)
cv_data_loader = torch.utils.data.DataLoader(cv_dataset,
                                             batch_size=1,
                                             shuffle=False)


## Model
model = getattr(separator,config.model)(config.input_dim,
                config.output_dim,
                config.hidden_dim,
                config.num_layers
                ).to(device)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

## Training
tr_loss = []
cv_loss = []
start = time.time()

print('Start training...')
with detect_anomaly():
    
    for epoch in range(config.num_epoch):
        
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
            
            # if i > 1:
            #     break
            
            
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
                
                # if i > 1:
                #     break                
                
        cv_loss.append(np.mean(running_loss))
        
        
        # Post-processing
        print('epoch: {:0=3}'.format(epoch))
        print('computational time: {0}'.format(time.time()-start))
        print('tr loss: {0}'.format(tr_loss[-1]))
        print('cv loss: {0}'.format(cv_loss[-1]))
        
        result_show(config.dir_name+'/separated.png',
                    a1[0, ...].detach().clone().to("cpu").numpy(),
                    a2[0, ...].detach().clone().to("cpu").numpy(),
                    (mask1*am)[0, ...].detach().clone().to("cpu").numpy(),
                    (mask2*am)[0, ...].detach().clone().to("cpu").numpy(),
                    )
        
        scheduler.step()
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), config.save_name.format(epoch))
    

print('Finish')