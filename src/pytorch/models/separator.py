# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRU2SPK(nn.Module):
    
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=300,
                 num_layers=4,
                 dropout=0.3
                 ):
        
        super(BiGRU2SPK, self).__init__()
        self.output_dim = output_dim
        self.rnn = nn.GRU(input_dim,
                          hidden_dim,
                          num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True
                          )
        self.fc = nn.Linear(hidden_dim*2,
                            output_dim*2)
        self.init_rnn(self.rnn)
        
    def init_rnn(self, m):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    torch.nn.init.xavier_uniform_(ih)
                    
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)
                    
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, frame_length, _ = x.size()
        rnn_output, _ = self.rnn(x)
        masks = self.fc(rnn_output)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(
            batch_size,
            frame_length,
            self.output_dim,
            2
            )
        masks = masks.permute(0, 2, 1, 3)
        return masks[..., 0], masks[..., 1]
    
class BiLSTM2SPK(nn.Module):
    
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=300,
                 num_layers=4,
                 dropout=0.3
                 ):
        
        super(BiLSTM2SPK, self).__init__()
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim,
                          hidden_dim,
                          num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True
                          )
        self.fc = nn.Linear(hidden_dim*2,
                            output_dim*2)
        self.init_rnn(self.rnn)
        
    def init_rnn(self, m):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    torch.nn.init.xavier_uniform_(ih)
                    
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)
                    
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, frame_length, _ = x.size()
        rnn_output, _ = self.rnn(x)
        masks = self.fc(rnn_output)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(
            batch_size,
            frame_length,
            self.output_dim,
            2
            )
        masks = masks.permute(0, 2, 1, 3)
        return masks[..., 0], masks[..., 1]


class BiGRU2SPKChimera(nn.Module):
    
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 embed_dim,
                 hidden_dim=300,
                 num_layers=4,
                 dropout=0.3
                 ):
        
        super(BiGRU2SPKChimera, self).__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.rnn = nn.GRU(input_dim,
                          hidden_dim,
                          num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True
                          )
        self.fc_mi = nn.Linear(hidden_dim*2,
                            output_dim*2)
        self.fc_dc = nn.Linear(hidden_dim*2,
                            output_dim*embed_dim)
        self.init_rnn(self.rnn)
        
    def init_rnn(self, m):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    torch.nn.init.xavier_uniform_(ih)
                    
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)
                    
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        batch_size, frame_length, _ = x.size()
        rnn_output, _ = self.rnn(x)
        
        # mask inferebce
        masks = self.fc_mi(rnn_output)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(
            batch_size,
            frame_length,
            self.output_dim,
            2
            )
        masks = masks.permute(0, 2, 1, 3)
        
        # embedding calculation
        net_embed = self.fc_dc(rnn_output)  # B x T x (F*dim_embed)
        net_embed = net_embed.reshape(batch_size, -1, self.embed_dim)  # B x TF x embed_dim 
        net_embed = F.normalize(net_embed, p=2, dim=-1)
        net_embed = net_embed.reshape(batch_size,
                                      frame_length,
                                      -1,
                                      self.embed_dim) 
        net_embed = net_embed.permute(0, 2, 1, 3)
        return masks[..., 0], masks[..., 1], net_embed