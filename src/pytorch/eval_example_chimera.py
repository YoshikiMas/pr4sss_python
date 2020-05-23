# -*- coding: utf-8 -*-
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
from data_utils.wsj02mix_dataset import WSJ02MixDataset
from utils.stft_related import STFT, iSTFT
from utils.pre_process import to_normlized_log
from models import separator
from utils.evals import msa_pit, dc_loss, sisdr_pi
from utils.visualization import result_show

from utils.phase_reconstruction import misi, divmisi_ver1

## Init
random.seed(777)  
np.random.seed(777)  
torch.manual_seed(777)  
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


## Train
def eval(config):
    device = config.device
    print(device)
    ## STFT/iSTFT
    stft = STFT(config.shift, config.winlen, device=device)

    ## Dataset
    tt_dataset = WSJ02MixDataset(config.dataset_base_tt)
    tt_data_loader = torch.utils.data.DataLoader(tt_dataset,
                                                 batch_size=1,
                                                 shuffle=False)


    ## Model
    model = getattr(separator,config.model)(config.input_dim,
                    config.output_dim,
                    config.embed_dim,
                    config.hidden_dim,
                    config.num_layers
                    ).to(device)
    model.load_state_dict(torch.load(config.eval_path, map_location='cuda'))
    ## Evaluation
    start = time.time()

    print('Start training...' + str(device))
    # with detect_anomaly():
    
    # Evaluation
    running_loss = []
    running_sisdr_obs = []
    running_sisdr_misi = []
    running_sisdr_prop = []
    model.eval()
    with torch.no_grad():
        for i, (mix, s1, s2) in enumerate(tt_data_loader):
            
            mix, s1, s2 = [x.to(device) for x in [mix, s1, s2]]
            cm, c1, c2 = [stft(x) for x in [mix, s1, s2]]
            am, a1, a2 = [complex_norm(x) for x in [cm, c1, c2]]
            mask1, mask2, _ = model(to_normlized_log(am))
            loss = msa_pit(a1, a2, mask1*am, mask2*am)
            running_loss.append(loss.item())

            # eval sisdr
            siglen_ = (mix.shape)[1]
            istft = iSTFT(config.shift, config.winlen, siglen_, device=device)
            
            c1est = (mask1[...,None]*cm)
            c2est = (mask2[...,None]*cm)     
            s1est_obs, s2est_obs =  [istft(x) for x in [c1est, c2est]]
            s1est_misi, s2est_misi = misi(c1est, c2est, mix, stft, istft)
            s1est_prop, s2est_prop = divmisi_ver1(c1est, c2est, mix, stft,
                                                  istft, maxiter=10)
            
            print("=="*32)
            running_sisdr_obs.append(sisdr_pi(s1, s2, s1est_obs, s2est_obs).mean().item())
            print(running_sisdr_obs[-1])
            running_sisdr_misi.append(sisdr_pi(s1, s2, s1est_misi, s2est_misi).mean().item())
            print(running_sisdr_misi[-1])
            running_sisdr_prop.append(sisdr_pi(s1, s2, s1est_prop, s2est_prop).mean().item())
            print(running_sisdr_prop[-1])
        

    print('computational time: {0}'.format(time.time()-start))
    print('cv loss: {0}'.format(np.mean(running_loss)))
    print('Finish')
    return running_sisdr_obs, running_sisdr_misi, running_sisdr_prop

if __name__ == '__main__':
    
    ## Params
    example_dir = '../../results/0515.2020.bigru_3_chimera'
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
    ## Eval
    running_sisdr_obs, running_sisdr_misi, running_sisdr_prop = eval(config)
    print("=="*32)
    print(np.mean(running_sisdr_obs))
    print(np.mean(running_sisdr_misi))
    print(np.mean(running_sisdr_prop))

