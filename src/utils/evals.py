# -*- coding: utf-8 -*-
import numpy as np

def sisdr(ori, est):
    """
    ori: time-domain signal (Ls)
    est: time-domain signal (Ls)

    """
    alpha = np.dot(ori, est)/np.dot(ori, ori)
    sisdr = 20*np.log10(np.linalg.norm(alpha*ori)/np.linalg.norm(alpha*ori-est))
    return sisdr
