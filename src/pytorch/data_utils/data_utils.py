# -*- coding: utf-8 -*-
import numpy as np
import torch

def zero_pad(x, shift, winlen, dtype=torch.float32):
    b, siglen = x.shape
    padlen = int(np.ceil((siglen-winlen)/shift)+1)*shift + winlen - siglen
    x = torch.cat([
        torch.zeros((b,winlen), dtype=dtype), x,
        torch.zeros((b,padlen+winlen), dtype=dtype)
        ], dim=-1)
    return x


