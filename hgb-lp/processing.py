import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
import torch
from scripts.data_loader import data_loader
from torch_sparse import SparseTensor, mul,eye
import sys
from torch_sparse import sum as sparsesum
from torch_sparse import matmul as torch_sparse_matmul


def load_dataset(args):
    name = args.dataset
    data = data_loader('./data/'+name)
    
    feature_list = []
    for i in range(len(data.nodes['count'])):
        feat = data.nodes['attr'][i]
        if feat is None:
            feature_list.append(torch.eye(data.nodes['count'][i]))
        else:
            feature_list.append(torch.FloatTensor(feat))

    num_nodes = data.nodes['total']
    num_train_links = data.links['total']
    num_test_links = data.links_test['total']

    adjs = []
    for i, (k, v) in enumerate(data.links['data'].items()):
        v = v.tocoo()
        row = v.row 
        col = v.col 
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=(num_nodes,num_nodes))
        adjs.append(adj)
        #print(adj)

        adj = SparseTensor(row=torch.LongTensor(col), col=torch.LongTensor(row), sparse_sizes=(num_nodes,num_nodes))
        adjs.append(adj)

    if name == "LastFM":
        row0, col0, _ = adjs[2].coo()
        row1, col1, _ = adjs[3].coo()
        UU = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=(num_nodes,num_nodes))
        UU = UU.coalesce()
        UU = UU.set_diag()

        adjss = [adjs[0],adjs[1],UU,adjs[4],adjs[5]]

    elif name == "amazon":
        row0, col0, _ = adjs[0].coo()
        row1, col1, _ = adjs[1].coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=(num_nodes,num_nodes))
        PP = PP.coalesce()
        PP = PP.set_diag()

        row0, col0, _ = adjs[2].coo()
        row1, col1, _ = adjs[3].coo()
        UU = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=(num_nodes,num_nodes))
        UU = UU.coalesce()
        UU = UU.set_diag()

        adjss = [PP, UU]

    
    adjs_new = []
    for adj in adjss:
        adj = adj.fill_value(1.)
        deg = sparsesum(adj, dim=1)
        deg_inv_sqrt = deg.pow_(-1.0)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) #D^-1A
        adjs_new.append(adj)

    adjs={}
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = args.device
    if name == "LastFM":
        adjs["UA"]=adjs_new[0].to(device)
        adjs["AU"]=adjs_new[1].to(device)
        adjs["UU"]=adjs_new[2].to(device)
        adjs["AT"]=adjs_new[3].to(device)
        adjs["TA"]=adjs_new[4].to(device)
    if name == "amazon":
        adjs["PP"]=adjs_new[0].to(device)
        adjs["UU"]=adjs_new[1].to(device)

    K = args.K 
    lis={}
    if name == "amazon":
        for k in range(1,K+1):
            lis[k]=[]
            if k==1:
                for i in adjs.keys():
                    lis[k].append(i)
            else:
                for i in lis[k-1]:
                    for j in adjs.keys():
                        lis[k].append(i+j)
    else:                    
        for k in range(1,K+1):
            lis[k]=[]
            if k==1:
                for i in adjs.keys():
                    lis[k].append(i)
            else:
                for i in lis[k-1]:
                    for j in adjs.keys():
                        if i[-1]==j[0]:
                            lis[k].append(i+j)

    lis_t={}
    for k, v in lis.items():
        lis_t[k] = []
        for i in v:
            temp = i[-2:][::-1]
            i = i[:-2]
            while(len(i)):
                temp += i[-2:][::-1]
                i = i[:-2]
            lis_t[k].append(temp)


    return feature_list, adjs, lis, lis_t

#if __name__ == '__main__':
#	load_dataset()


