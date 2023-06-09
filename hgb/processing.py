import torch
import sys
import numpy as np
from torch_sparse import SparseTensor, mul,eye
from torch_sparse import sum as sparsesum
from dataset_loader import data_loader
from torch_sparse import matmul as torch_sparse_matmul

def load_dataset(args):
    root = "./data/"
    dl = data_loader(root+args.dataset)

    #futures
    #use one-hot index vectors for nodes with no features
    feature_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            feature_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            feature_list.append(torch.FloatTensor(th))
    
    #labels and idx
    num_classes = dl.labels_test['num_classes']
    if args.dataset == "IMDB":
        labels = torch.FloatTensor(dl.labels_train['data']+dl.labels_test['data'])
    else:
        labels = torch.LongTensor(dl.labels_train['data']+dl.labels_test['data']).argmax(dim=1)

    val_ratio = 0.2 #split used in HGB 
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]

    adjs = []
    N = dl.nodes['total']

    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        row = v.row 
        col = v.col 
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=(N,N))
        adjs.append(adj)

    if args.dataset == "DBLP":
        adjs = [adjs[0], adjs[1], adjs[2], adjs[4], adjs[3], adjs[5]]
    
    adjs_new =[]
    for i in range(len(adjs)):
        if i%2==0:
            adjs_new.append(adjs[i])
    for i in range(len(adjs)):
        if i%2==0:
            adjs_new.append(adjs[i+1])

    if args.dataset == "ACM":
        row0, col0, _ = adjs_new[0].coo()
        row1, col1, _ = adjs_new[4].coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=(N,N))
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs_new[0]=PP

    for i in range(len(adjs_new)):
        adj = adjs_new[i]
        adj = adj.fill_value(1.)
        deg = sparsesum(adj, dim=1)
        deg_inv_sqrt = deg.pow_(-1.0)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) #D^-1A
        adjs_new[i] = adj

    #DBLP
    adjs={}
    if args.dataset=="DBLP":
        adjs["AP"]=adjs_new[0].to(args.device)
        adjs["PT"]=adjs_new[1].to(args.device)
        adjs["PV"]=adjs_new[2].to(args.device)
        adjs["PA"]=adjs_new[3].to(args.device)
        adjs["TP"]=adjs_new[4].to(args.device)
        adjs["VP"]=adjs_new[5].to(args.device)
    if args.dataset=="ACM":
        adjs["PP"]=adjs_new[0].to(args.device)
        adjs["PA"]=adjs_new[1].to(args.device)
        adjs["PC"]=adjs_new[2].to(args.device)
        adjs["PK"]=adjs_new[3].to(args.device)
        #adjs["IP"]=adjs_new[4].to(args.device)
        adjs["AP"]=adjs_new[5].to(args.device)
        adjs["CP"]=adjs_new[6].to(args.device)
        adjs["KP"]=adjs_new[7].to(args.device)
    if args.dataset=="IMDB":
        adjs["MD"]=adjs_new[0].to(args.device)
        adjs["MA"]=adjs_new[1].to(args.device)
        adjs["MK"]=adjs_new[2].to(args.device)
        adjs["DM"]=adjs_new[3].to(args.device)
        adjs["AM"]=adjs_new[4].to(args.device)
        adjs["KM"]=adjs_new[5].to(args.device)
    if args.dataset=="AMiner":
        adjs["PA"]=adjs_new[0].to(args.device)
        adjs["PR"]=adjs_new[1].to(args.device)
        adjs["AP"]=adjs_new[2].to(args.device)
        adjs["RP"]=adjs_new[3].to(args.device)

    lis={}
    for k in range(1,args.K+1):
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

    return feature_list, adjs, lis, lis_t, labels, num_classes, train_idx, val_idx, test_idx


