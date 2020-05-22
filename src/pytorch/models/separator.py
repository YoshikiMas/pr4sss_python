# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

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
    
class RBiLSTM2SPK(nn.Module):
    
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=300,
                 num_layers=4,
                 dropout=0.3
                 ):
        
        super(RBiLSTM2SPK, self).__init__()
        self.output_dim = output_dim
        self.rnn1 = nn.LSTM(hidden_dim*2,
                          hidden_dim,
                          num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True
                          )
        self.rnn2 = nn.LSTM(hidden_dim*2,
                          hidden_dim,
                          num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True
                          )
        self.fcin = nn.Linear(input_dim,
                            hidden_dim*2)
        self.fcout = nn.Linear(hidden_dim*2,
                            output_dim*2)
        self.init_rnn(self.rnn1)
        
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
        
        h0 = self.fcin(x)
        
        rnn_output1, _ = self.rnn1(h0)
        h1 = h0 + rnn_output1
        rnn_output2, _ = self.rnn2(h1)
        h2 = h1 + rnn_output2
        
        masks = self.fcout(h2)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(
            batch_size,
            frame_length,
            self.output_dim,
            2
            )
        masks = masks.permute(0, 2, 1, 3)
        return masks[..., 0], masks[..., 1]
    
class BiLSTMRI(nn.Module):
    
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=300,
                 num_layers=4,
                 dropout=0.3
                 ):
        
        super(BiLSTMRI, self).__init__()
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
    
class BiGRU2SPKDC(nn.Module):
    
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 embed_dim,
                 hidden_dim=300,
                 num_layers=4,
                 dropout=0.3
                 ):
        
        super(BiGRU2SPKDC, self).__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.rnn = nn.GRU(input_dim,
                          hidden_dim,
                          num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True
                          )
        self.fc_dc = nn.Linear(hidden_dim*2,
                            output_dim*embed_dim)
        self.init_rnn(self.rnn)
        self.kmeans = KMeans(n_clusters=2)
        
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
        
        # embedding calculation
        net_embed = self.fc_dc(rnn_output)  # B x T x (F*dim_embed)
        net_embed = net_embed.reshape(batch_size, -1, self.embed_dim)  # B x TF x embed_dim 
        net_embed = F.normalize(net_embed, p=2, dim=-1)
        net_embed = net_embed.reshape(batch_size,
                                      frame_length,
                                      -1,
                                      self.embed_dim) 
        net_embed = net_embed.permute(0, 2, 1, 3)
        return net_embed
    
    def separate(self, x):
        net_embed = (self.forward(x)).detach().clone().to("cpu").numpy()  # 1 x F x T x D
        _, freq_length, frame_length, _ = net_embed.shape
        net_embed = net_embed[0,...].reshape(freq_length*frame_length, -1)  # TF x D
        mix_cluster = self.kmeans.fit_predict(net_embed)
        masks = []
        for i in range(2):
            mask = (mix_cluster == i)
            masks.append(mask.reshape(freq_length, frame_length))
        
        return masks[0], masks[1]
        
if __name__ == '__main__':

    device = torch.device("cpu")
    
    B = 1
    T = 30
    K = 50
    D = 10
    H = 30
    N = 2
    
    model = BiGRU2SPKDC(K, K, D, H, 1).to(device)
    mask1, mask2 = model.separate(torch.randn(B, K, T).to(torch.float32).to(device))