# -*- coding: utf-8 -*-
import torch

def msa(s, x, comp_ratio=1., eps=1e-5):
    s = (s+eps)**comp_ratio
    x = (x+eps)**comp_ratio
    return torch.sum(torch.pow(s-x, 2), [-2,-1])

def psa(s, x):
    """
    Parameters
    ----------
    s : batch x freq x time x real x imag
    x : batch x freq x time x real x imag
    Returns
    -------
    loss: batch 
    """
    return torch.sum(torch.pow(s-x, 2), [-3,-2,-1])

def msa_pit(s1, s2, x1, x2):
    
    msa1 = msa(s1, x1) + msa(s2, x2) 
    msa2 = msa(s1, x2) + msa(s2, x1)
    
    loss = torch.min(msa1, msa2).mean()
    return loss

def msal1(s, x):
    return torch.sum(torch.abs(s-x), [-2,-1])

def msal1_pit(s1, s2, x1, x2):
    
    msa1 = msal1(s1, x1) + msal1(s2, x2) 
    msa2 = msal1(s1, x2) + msal1(s2, x1)
    
    loss = torch.min(msa1, msa2).mean()
    return loss



def dc_loss(a1, a2, net_embed, binary_non_slient, n_spks=2, device='cpu'):
    """
    Arguments:
        net_embed B x F x T x D
        tgt_index B x F x T x 
        binary_mask B x F x T
    """
    B, F, T, D = net_embed.shape
    net_embed = net_embed.reshape(B, T*F, D)
    probs = torch.cat((a1[...,None], a2[...,None]), -1)                            
    tgt_idx = torch.argmax(probs, -1, keepdim=False)
    tgt_embed = torch.nn.functional.one_hot(tgt_idx, num_classes=2)
    tgt_embed = tgt_embed.reshape(B, T*F, n_spks).to(torch.float32)
    binary_non_slient = binary_non_slient.reshape(B, T*F, 1)

    # encode one-hot
    net_embed = net_embed * binary_non_slient
    tgt_embed = tgt_embed * binary_non_slient
    loss = torch.norm( torch.bmm(net_embed.permute(0, 2, 1), net_embed),
                      dim=(-2, -1) ) + \
        torch.norm( torch.bmm(tgt_embed.permute(0, 2, 1), tgt_embed),
                   dim=(-2, -1) ) - \
        torch.norm( torch.bmm(net_embed.permute(0, 2, 1), tgt_embed),
                   dim=(-2, -1) ) * 2
    return loss.mean()

def sisdr(s, sest):
    alpha = (s*sest).sum(1, keepdim=True)/(s*s).sum(1, keepdim=True)
    s_ = alpha*s
    r = s_ - sest
    val = 10*torch.log10((s_*s_).sum(dim=1)/(r*r).sum(dim=1))
    return val

def sisdr_pi(s1, s2, s1est, s2est):
    
    val1 = (sisdr(s1, s1est) + sisdr(s2, s2est))/2
    val2 = (sisdr(s1, s2est) + sisdr(s2, s1est))/2
    val = torch.max(val1, val2).mean()
    return val

if __name__ == '__main__':

    device = torch.device("cpu")
    
    B = 20
    T = 30
    F = 50
    D = 10
    N = 2
 
    a1 = torch.randn(B, T, F).to(torch.float32).to(device)
    a2 = torch.randn(B, T, F).to(torch.float32).to(device)
    net_embed = torch.randn(B, T, F, D).to(torch.float32).to(device)
    binary_non_slient = torch.randn(B, T, F).sign().to(device)
    loss = dc_loss(a1,
                   a2,
                   net_embed,
                   binary_non_slient,
                   n_spks=N,
                   device=device)
    print(loss.item())
        
         
        
    
    
    