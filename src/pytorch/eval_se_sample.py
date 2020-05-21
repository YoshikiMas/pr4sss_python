# -*- coding: utf-8 -*-
import time
import glob
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
from utils.evals import msa, sisdr
from utils.visualization import result_show

from utils.phase_reconstruction import misi, divmisi_ver1, divmisi_ver2

## Init
random.seed(777)  
np.random.seed(777)  
torch.manual_seed(777)  
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


## Train
def eval(config):
    device = config.device
    
    ## STFT/iSTFT
    stft = STFT(config.shift, config.winlen, device=device)
    istft = iSTFT(config.shift, config.winlen, device=device)


    ## Dataset
    tt_dataset = VoicebankDemandDataset(config.dataset_base,
                                        train=False)
    tt_data_loader = torch.utils.data.DataLoader(tt_dataset,
                                                 batch_size=1,
                                                 shuffle=False)


    ## Model
    model = getattr(separator,config.model)(config.input_dim,
                    config.output_dim,
                    config.hidden_dim,
                    config.num_layers
                    ).to(device)
    model.load_state_dict(torch.load(config.eval_path, map_location='cuda'))


    print('Start evaluation...' + str(device))
    with detect_anomaly():

        with torch.no_grad():

            # Evaluation
            running_sisdr_obs = []
            running_sisdr_misi = []
            running_sisdr_prop = []
            model.eval()
            for i, (s, n, sn) in enumerate(tt_data_loader):
                
                s, n, sn = [x.to(device) for x in [s, n, sn]]
                cs, cn, csn = [stft(x) for x in [s, n, sn]]
                amps, ampn, ampsn = [complex_norm(x) for x in [cs, cn, csn]]
                masks, maskn = model(to_normlized_log(ampsn))
                
                siglen_ = (sn.shape)[1]
                istft = iSTFT(config.shift, config.winlen, siglen_, device=device)
                
                csest = (masks[...,None]*csn)
                cnest = (maskn[...,None]*csn)     
                sest_obs =  istft(csest)
                sest_misi, _ = misi(csest, cnest, sn, stft, istft)
                sest_prop, _ = divmisi_ver2(csest, cnest, sn, stft, istft, maxiter=100)

                print("=="*32)
                running_sisdr_obs.append(sisdr(s, sest_obs).mean().item())
                print(running_sisdr_obs[-1])
                running_sisdr_misi.append(sisdr(s, sest_misi).mean().item())
                print(running_sisdr_misi[-1])
                running_sisdr_prop.append(sisdr(s, sest_prop).mean().item())
                print(running_sisdr_prop[-1])                                
                

                if i > 10:
                    break

    print('Finish')
    return running_sisdr_obs, running_sisdr_misi, running_sisdr_prop
    

if __name__ == '__main__':
    
    ## Params
    example_dir = '../../results/0515.2020.bilstm_3_se'
    parser = ArgumentParser(description='Training script for Deep speech sep.')
    parser.add_argument('--dir_name', default=example_dir, type=str)
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
    ckpts = np.sort(glob.glob(config.dir_name + '/*.ckpt'))
    config.eval_path = ckpts[-1]
    if hasattr(config, 'optimizer'):
        pass
    else:
        config.optimizer = 'Adam'
    if hasattr(config, 'weight_decay'):
        pass
    else:
        config.weight_decay = 0
        
    ## Eval
    running_sisdr_obs, running_sisdr_misi, running_sisdr_prop = eval(config)
    print("=="*32)
    print(np.mean(running_sisdr_obs))
    print(np.mean(running_sisdr_misi))
    print(np.mean(running_sisdr_prop))


