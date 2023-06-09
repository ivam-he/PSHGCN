import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layer, dropout, bias=True):
        super(MLP, self).__init__()
        self.num_layer = num_layer
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()

        if self.num_layer == 1:
            self.lins.append(nn.Linear(in_dim, out_dim, bias=bias))
        else:
            self.lins.append(nn.Linear(in_dim, hidden, bias=bias))
            self.norms.append(nn.BatchNorm1d(hidden))
            self.acts.append(nn.PReLU())
            for i in range(self.num_layer - 2):
                self.lins.append(nn.Linear(hidden, hidden, bias=bias))
                self.norms.append(nn.BatchNorm1d(hidden))
                self.acts.append(nn.PReLU())
            self.lins.append(nn.Linear(hidden, out_dim, bias=bias))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lin in self.lins:
            stdv = 1. / math.sqrt(lin.weight.size(1))
            nn.init.uniform_(lin.weight, -stdv, stdv)
            if lin.bias is not None: 
                nn.init.zeros_(lin.bias)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < self.num_layer -1: 
                x = self.norms[i](x)
                x = self.acts[i](x)
                x = self.dropout(x)
        return x


class PSHGCN(nn.Module):
    def __init__(self, emb_dim, hidden_x, hidden_l, nclass, node_types, label_keys, layers_x, layers_l, coe_num, expander, dropout, input_drop, bias=False):
        super(PSHGCN, self).__init__()
        self.coe_num = coe_num
        self.expander = expander
        self.label_keys = label_keys
        self.layers_emb = 2
        
        self.embedings_x = nn.ParameterDict({})
        for node in node_types:
            if node == "P":
                self.embedings_x[node] = MLP(128, emb_dim, emb_dim, self.layers_emb, dropout, bias=bias)
            else:
                self.embedings_x[node] = MLP(256, emb_dim, emb_dim, self.layers_emb, dropout, bias=bias)

        self.embedings_l = nn.ParameterDict({})
        for key in label_keys:
            self.embedings_l[key] = MLP(nclass, emb_dim, emb_dim, self.layers_emb, dropout, bias=bias)
        
        self.lin_residue = nn.Linear(128, emb_dim, bias=False)
        self.lin_x = MLP(emb_dim, hidden_x, nclass, layers_x, dropout)
        self.lin_l = MLP(nclass, hidden_l, nclass, layers_l, dropout)
  
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.W_x = nn.Parameter(torch.tensor(self.coe_init_x()))
        self.W_l = nn.Parameter(torch.tensor(self.coe_init_l()))

        self.input_dropout = nn.Dropout(input_drop)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lin_residue.weight, gain=gain)

    def coe_init_x(self):
        TEMP = np.zeros(self.coe_num)
        TEMP[0] = 1.0
        return TEMP[:self.coe_num]

    def coe_init_l(self):
        num_coe = len(self.label_keys)
        TEMP = np.zeros(num_coe)
        for i in range(num_coe):
            TEMP[i] = 1.0/num_coe
        return TEMP

    def forward(self, feats, label_feats):
    	###feats and label_feats are both dicts
        init_x = self.input_dropout(feats['P'])
        for k, v in feats.items():
            feats[k] = self.embedings_x[k[-1]](self.input_dropout(v))

        for k, v in feats.items():
            if k in self.label_keys:
                feats[k] = feats[k]+self.embedings_l[k](label_feats[k])

        x =self.W_x[0]*self.W_x[0]*feats["P"]
        for i in range(1,len(self.expander)):
            w1 = int(self.expander[i][:2])-10
            w2 = int(self.expander[i][2:4])-10
            x+=self.W_x[w1]*self.W_x[w2]*(feats[self.expander[i][4:]])

        lx = self.W_l[0]*label_feats[self.label_keys[0]]
        for i in range(1,len(self.label_keys)):
            lx += self.W_l[i]*label_feats[self.label_keys[i]]

        x = x+self.lin_residue(init_x)
        x = self.dropout(self.prelu(x))
        x = self.lin_x(x)
        x = x + self.lin_l(lx)
        return x

