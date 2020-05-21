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
from data_utils.voicebank_demand_dataset import VoicebankDemandDataset
from utils.stft_related import STFT, iSTFT
from utils.pre_process import to_normlized_log
from models import separator
from utils.evals import msa
from utils.visualization import result_show

from utils.phase_reconstruction import misi, divmisi_ver1

## Init
random.seed(777)  
np.random.seed(777)  
torch.manual_seed(777)  
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


## Train
def train(config):
    device = config.device
    
    ## STFT/iSTFT
    stft = STFT(config.shift, config.winlen, device=device)
    istft = iSTFT(config.shift, config.winlen, device=device)


    ## Dataset
    tr_dataset = VoicebankDemandDataset(config.dataset_base,
                                        train=config.train,
                                        siglen=config.siglen)
    tr_data_loader = torch.utils.data.DataLoader(tr_dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=True)


    ## Model
    model = getattr(separator,config.model)(config.input_dim,
                    config.output_dim,
                    config.hidden_dim,
                    config.num_layers
                    ).to(device)

    optimizer = getattr(optim,config.optimizer)(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    ## Training
    tr_loss = []
    cv_loss = []
    start = time.time()

    print('Start training...' + str(device))
    with detect_anomaly():

        for epoch in range(config.num_epoch):

            # Training
            running_loss = []
            model.train()
            for i, (s, n, sn) in enumerate(tr_data_loader):

                s, n, sn = [stft(x.to(device)) for x in [s, n, sn]]
                amps, ampn, ampsn = [complex_norm(x) for x in [s, n, sn]]
                masks, maskn = model(to_normlized_log(ampsn))
                loss = torch.mean(config.ratio*msa(amps, masks*ampsn) + \
                                  (1.-config.ratio)*msa(ampn, maskn*ampsn))
                model.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())

                
            tr_loss.append(np.mean(running_loss))


            # Post-processing
            print('epoch: {:0=3}'.format(epoch))
            print('computational time: {0}'.format(time.time()-start))
            print('tr loss: {0}'.format(tr_loss[-1]))

            # mask1, mask2 = model.separate(to_normlized_log(am))
            amps = amps[0, ...].detach().clone().to("cpu").numpy()
            ampn = ampn[0, ...].detach().clone().to("cpu").numpy()
            ampsest = (masks*ampsn)[0, ...].detach().clone().to("cpu").numpy()
            ampnest = (maskn*ampsn)[0, ...].detach().clone().to("cpu").numpy()
            
            result_show(config.dir_name+'/separated.png', amps, ampn,
                        ampsest, ampnest, aspect=0.5)

            scheduler.step()
            if (epoch+1)%10 == 0:
                torch.save(model.state_dict(), config.save_name.format(epoch))
                pass            

    print('Finish')
    

if __name__ == '__main__':
    
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
    config.device = torch.device(config.device)
    if hasattr(config, 'optimizer'):
        pass
    else:
        config.optimizer = 'Adam'
    if hasattr(config, 'weight_decay'):
        pass
    else:
        config.weight_decay = 0
        
    ## Train
    train(config)