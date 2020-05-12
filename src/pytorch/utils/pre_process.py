# -*- coding: utf-8 -*-
import torch

def to_normlized_log(x, flooring=1e-4, eps=1e-8):
    bs = x.shape[0]
    maxval, _ = x.reshape(-1).max(-1)
    logx = torch.log10(torch.clamp(x, min=flooring*maxval))
    norm_logx = (logx - logx.mean(-1, keepdim=True))/(logx.std(-1, keepdim=True)+eps)
    return norm_logx

