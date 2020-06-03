# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=None,
                 padding_mode="reflect",
                 leakiness=0.2):
        super().__init__()
        if padding is None:
             padding = [(i - 1) // 2 for i in kernel_size]  # 'same' padding
             
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(leakiness)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):
    
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=None,
                 leakiness=0.2):
        super().__init__() 
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'same' padding
            
        self.tconv = nn.ConvTranspose2d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(leakiness)
        
    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
        


class UNet(nn.Module):
    def __init__(self,
                 kernel_size=(5,3),
                 stride=(1,2),
                 leakiness=0.2,
                 encoder_channels=[(1,2), (2,4), (4,8), (8, 16), (16,32)],
                 decoder_channels=[(32,16), (32,8), (16, 4), (8,2), (4,2)]
                 ):
    
        super(UNet, self).__init__()
        self.encoders = []
        self.decoders = []
        self.stride = stride
        for idx, channel in enumerate(encoder_channels):
            enc = Encoder(in_channels=channel[0],
                          out_channels=channel[1],
                          kernel_size=kernel_size,
                          stride=stride)
            self.add_module("encoder{}".format(idx), enc)
            self.encoders.append(enc)
            
        for idx, channel in enumerate(decoder_channels):
            dec = Decoder(in_channels=channel[0],
                          out_channels=channel[1],
                          kernel_size=kernel_size,
                          stride=stride)
            self.add_module("decoder{}".format(idx), dec)
            self.decoders.append(dec)
            
        self.last_conv=nn.Conv2d(decoder_channels[-1][-1], 1, kernel_size=1) 
        
    def pre_pad(self, x):
        bn, fn, tn = x.shape
        base_fn = (self.stride[0])**len(self.encoders)
        base_tn = (self.stride[1])**len(self.encoders)
        pad_fn = int(np.ceil((fn-1)/base_fn)*base_fn) + 1 - fn
        pad_tn = int(np.ceil((tn-1)/base_tn)*base_tn) + 1 - tn
        x = torch.cat((x, torch.zeros((bn, fn, pad_tn), dtype=x.dtype, device=x.device)),
                      axis=2)
        x = torch.cat((x, torch.zeros((bn, pad_fn, tn+pad_tn), dtype=x.dtype, device=x.device)),
                      axis=1)
        return x, fn, tn
         
    def forward(self, xin):      
        outputs = []
        xpad, freqs, frames = self.pre_pad(xin)
        x = xpad[:, None, :, :]
        # encoder
        for i in range(len(self.encoders)):
            outputs.append(x)
            x = self.encoders[i](outputs[-1])
        
        # decoder
        x = self.decoders[0](x)
        for i in range(1,len(self.decoders)):
            x = self.decoders[i](torch.cat((x, outputs[-i]), dim=1))
        est_mask = torch.sigmoid(self.last_conv(x))
        return est_mask[:, 0, :freqs, :frames]
    
if __name__ == '__main__':

    device = torch.device("cpu")
    print(device)
    model = UNet().to(device)
    
    # Case 1
    B = 1
    K = 1  # Arbitrary integer
    T = 33  # Arbitrary integer
    mask1 = model(torch.randn((B, K, T), dtype=torch.float32, device=device))
    print(list(mask1.shape) == [B, K, T])
    
    device = torch.device("cuda:0")
    print(device)
    
    # Case 2
    model = UNet().to(device)
    B = 4
    K = 257
    T = 1 + 32*10
    mask2 = model(torch.randn((B, K, T), dtype=torch.float32, device=device))
    print(list(mask2.shape) == [B, K, T])

    # Case 3
    model = UNet(encoder_channels=[(1,32), (32,64), (64,64), (64, 64), (64,64)],
                  decoder_channels=[(64,64), (128,64), (128, 64), (128,32), (64,16)]
                  ).to(device)
    B = 4
    K = 257
    T = 2 + 32*10
    mask3 = model(torch.randn((B, K, T), dtype=torch.float32, device=device))
    print(list(mask3.shape) == [B, K, T])
    
    # Case 3
    model = UNet(encoder_channels=[(1,32), (32,64), (64,64), (64, 64), (64,64)],
                  decoder_channels=[(64,64), (128,64), (128, 64), (128,32), (64,16)],
                  stride=(2,2)
                  ).to(device)
    mask4 = model(torch.randn((B, K, T), dtype=torch.float32, device=device))
    print(list(mask4.shape) == [B, K, T])