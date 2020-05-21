# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def result_show(fig_name, s1, s2, x1, x2, vmin=-60, eps=1e-10, aspect=4.):
    
    vmax = 20*np.log10(np.max([np.max(s1), np.max(s2)]))-10
    vmin += vmax

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].imshow(20*np.log10(np.flipud(s1)+eps), vmax=vmax, vmin=vmin)
    axes[0,1].imshow(20*np.log10(np.flipud(s2)+eps), vmax=vmax, vmin=vmin)
    axes[1,0].imshow(20*np.log10(np.flipud(x1)+eps), vmax=vmax, vmin=vmin)
    axes[1,1].imshow(20*np.log10(np.flipud(x2)+eps), vmax=vmax, vmin=vmin)    
    
    for axis in axes.flatten():
        axis.set_aspect(aspect)
        
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


