# -*- coding: utf-8 -*-
import torch

def msa(s, x):
    return torch.sum(torch.pow(s-x, 2), [-2,-1])

def msa_pit(s1, s2, x1, x2):
    
    msa1 = msa(s1, x1) + msa(s2, x2) 
    msa2 = msa(s1, x2) + msa(s2, x1)
    
    loss = torch.min(msa1, msa2).mean()
    return loss
    
    
    
    
    


