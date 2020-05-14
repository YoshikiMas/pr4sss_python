# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

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
        self.init_gru(self.rnn)
        
    def init_gru(self, m):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    torch.nn.init.xavier_uniform_(ih)
                    
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)
                    
            else:
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
        self.init_gru(self.rnn)
        
    def init_gru(self, m):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    torch.nn.init.xavier_uniform_(ih)
                    
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)
                    
            else:
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
