import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_sparse import matmul as torch_sparse_matmul
from torch_sparse import SparseTensor
from utils import add

class PSHGCN(nn.Module):
    def __init__(self, in_dims, num_classes,lis, lis_t, args, bns=False):
        super(PSHGCN, self).__init__()
        self.in_dims = in_dims
        self.num_classes = num_classes
        self.emb_dim = args.emb_dim
        self.h_dim = args.hidden
        self.K = args.K
        self.lis =lis
        self.lis_t =lis_t
        self.feat_project = nn.ModuleList([nn.Linear(in_dim, self.emb_dim, bias=bns)  for in_dim in self.in_dims])
        self.lin1 = nn.Linear(self.emb_dim, self.h_dim)
        self.lin2 = nn.Linear(self.h_dim, self.num_classes)

        self.W = nn.Parameter(torch.tensor(self.init_coe()))
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def init_coe(self):
        coe_num = 1+len(self.lis[1])
        for i in range(2,self.K+1):
            coe_num+=len(self.lis[i])

        ###Random
        bound = np.sqrt(3.0/(coe_num))
        TEMP = np.random.uniform(-bound, bound, coe_num)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        return TEMP[:coe_num]

    def normalize(self, x):
        means = x.mean(1, keepdim=True)
        deviations = x.std(1, keepdim=True)
        x = (x - means) / deviations
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        return x

    def forward(self, adjs, features_list):
        output = []
        for lin, feature in zip(self.feat_project, features_list):
            feature = self.input_drop(lin(feature))
            output.append(feature)
        
        x = torch.cat(output,0)
        x = F.relu(self.lin1(x))
        x = self.normalize(x)
        x = self.dropout(x)
        #g propgation
        coe_index = 0
        res = self.W[coe_index]*x
        coe_index += 1
        for k in range(1,self.K+1):
            temp_now = {}
            if k == 1:
                for i in self.lis[k]:
                    out = torch_sparse_matmul(adjs[i],x)
                    temp_now[i] = out
                    res += self.W[coe_index]*out
                    coe_index += 1
                temp_lst = temp_now
            else:
                for i, j in enumerate(self.lis[k]):
                    out = torch_sparse_matmul(adjs[j[:2]],temp_lst[j[2:]])
                    temp_now[j] = out
                    res += self.W[coe_index]*out
                    coe_index += 1
                temp_lst = temp_now

        #g^t propagation
        x = res
        coe_index = 0
        res = self.W[coe_index]*x
        coe_index += 1
        for k in range(1,self.K+1):
            temp_now = {}
            if k == 1:
                for i in self.lis_t[k]:
                    out = torch_sparse_matmul(adjs[i],x)
                    temp_now[i] = out
                    res += self.W[coe_index]*out
                    coe_index += 1
                temp_lst = temp_now
            else:
                for i, j in enumerate(self.lis_t[k]):
                    out = torch_sparse_matmul(adjs[j[:2]],temp_lst[j[2:]])
                    temp_now[j] = out
                    res += self.W[coe_index]*out
                    coe_index += 1
                temp_lst = temp_now

        res = self.lin2(res)
        return res