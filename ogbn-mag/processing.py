import os
import torch
import dgl
import gc
import time
import dgl.function as fn
import numpy as np
import sparse_tools
import torch.nn.functional as F
from torch_sparse import SparseTensor,mul
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from utils import propagate, propagate_clean

def expander(K):
    keys = ["AI","AP","PP","PF","IA","PA","FP"]
    lis = {}
    for k in range(1,K+1):
        lis[k] = []
        if k==1:
            for i in keys:
                lis[k].append(i)
        else:
            for i in lis[k-1]:
                for j in keys:
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

    gi = ["10"]
    index = 11
    for v in lis.values():
        for i in v:
            gi.append(str(index)+i)
            index+=1
    
    gi_t = ["10"]
    index = 11
    for v in lis_t.values():
        for i in v:
            gi_t.append(str(index)+i)
            index+=1
    res=[]
    for i in gi_t:
        for j in gi:
            if len(i)==2: 
                if len(j)==2 or j[2]=="P":
                    res.append(i+j)
            else:
                coe=i[:2]+j[:2]
                if len(j)==2 and i[2]=="P":
                    res.append(coe+i[2:])
                elif len(j)>2 and i[-1]==j[2] and i[2]=="P":
                    res.append(coe+i[2:]+j[2:])
    num_coe=len(gi)
    return res,num_coe

def load_dataset(args):
     #ogbn-mag:
     # A ---  P* --- F
     # |      |
     # I      P
     # author: [1134649, ]
     # paper : [736389, 128]
     # field  : [59965,  ]
     # institution : [8740, ]
     #train_set: 629571 (85.5%)
     #val_set: 64879 (8.8%)
     #test_set: 41939 (5.7%)
     #feat_dimension: 128
     #classes: 349

    dataset = DglNodePropPredDataset(name=args.dataset,root=args.root)
    g, labels = dataset[0]
    labels = labels['paper'].squeeze()

    ###fixed split
    splitted_idx = dataset.get_idx_split()
    train_idx = splitted_idx['train']['paper']
    val_idx = splitted_idx['valid']['paper']
    test_idx = splitted_idx['test']['paper']

    ###node features
    features={}
    features["P"] = g.nodes['paper'].data['feat'] #736389 X 128
    if args.extra_emb:
        path = "./complEx"
        features["A"] = torch.load(os.path.join(path, 'author.pt'), map_location=torch.device('cpu')).float()
        features["F"] = torch.load(os.path.join(path, 'field_of_study.pt'), map_location=torch.device('cpu')).float()
        features["I"] = torch.load(os.path.join(path, 'institution.pt'), map_location=torch.device('cpu')).float()
        print("Use the extra embedding for nodes without features.")
    else:
        feat_dim = 256
        features["A"] = torch.Tensor(g.num_nodes('author'), feat_dim).uniform_(-0.5, 0.5) #1134649 X 256 
        features["F"] = torch.Tensor(g.num_nodes('field_of_study'), feat_dim).uniform_(-0.5, 0.5) #59965 X 256
        features["I"] = torch.Tensor(g.num_nodes('institution'), feat_dim).uniform_(-0.5, 0.5) #8740 X 256
    
    ###adjacency matrix
    adjs = []
    for i, etype in enumerate(g.etypes):
        src, dst, eid = g._graph.edges(i)
        adj = SparseTensor(row=src, col=dst)
        adjs.append(adj)
    adjs[2] = adjs[2].to_symmetric()
    assert torch.all(adjs[2].get_diag() == 0)
    edges = {}
    etypes = [('A', 'A-I', 'I'), ('A', 'A-P', 'P'), ('P', 'P-P', 'P'), ('P', 'P-F', 'F')]
    for etype, adj in zip(etypes, adjs):
        stype, rtype, dtype = etype
        src, dst, _ = adj.coo()
        src = src.numpy()
        dst = dst.numpy()
        if stype == dtype:
            edges[(stype, rtype, dtype)] = (src,dst)
        else:
            edges[(stype, rtype, dtype)] = (src, dst)
            edges[(dtype, rtype[::-1], stype)] = (dst, src) 

    new_g = dgl.heterograph(edges)
    new_g.nodes['P'].data['P'] = features["P"]
    new_g.nodes['A'].data['A'] = features["A"]
    new_g.nodes['I'].data['I'] = features["I"]
    new_g.nodes['F'].data['F'] = features["F"]

    ###for label propagation
    diag_name = f'./data/{args.dataset}_PFFP_diag.pt'
    if not os.path.exists(diag_name):
        PF = adjs[3]
        PFFP_diag = sparse_tools.spspmm_diag_sym_ABA(PF)
        torch.save(PFFP_diag, diag_name)

    diag_name = f'./data/{args.dataset}_PPPP_diag.pt'
    if not os.path.exists(diag_name):
        PP = adjs[2]
        PPPP_diag = sparse_tools.spspmm_diag_sym_AAA(PP)
        torch.save(PPPP_diag, diag_name)

    diag_name = f'./data/{args.dataset}_PAAP_diag.pt'
    if not os.path.exists(diag_name):
        PA = adjs[1].t()
        PAAP_diag = sparse_tools.spspmm_diag_sym_ABA(PA)
        torch.save(PAAP_diag, diag_name)

    ###features processing
    g = propagate(new_g, args.K)
    feats = {}
    for feat_key in list(g.nodes["P"].data.keys()):
        feats[feat_key] = g.nodes["P"].data.pop(feat_key)
    g = propagate_clean(g)
    gc.collect()
    expand_polynomial, num_coe = expander(int(args.K/2))
    #print(expand_polynomial)
    #print(num_coe)

    ###for label propagation
    diag_name = f'./data/{args.dataset}_PFFP_diag.pt'
    if not os.path.exists(diag_name):
        PF = adjs[3]
        PFFP_diag = sparse_tools.spspmm_diag_sym_ABA(PF)
        torch.save(PFFP_diag, diag_name)

    diag_name = f'./data/{args.dataset}_PPPP_diag.pt'
    if not os.path.exists(diag_name):
        PP = adjs[2]
        PPPP_diag = sparse_tools.spspmm_diag_sym_AAA(PP)
        torch.save(PPPP_diag, diag_name)

    diag_name = f'./data/{args.dataset}_PAAP_diag.pt'
    if not os.path.exists(diag_name):
        PA = adjs[1].t()
        PAAP_diag = sparse_tools.spspmm_diag_sym_ABA(PA)
        torch.save(PAAP_diag, diag_name)

    return feats,labels,expand_polynomial,train_idx,val_idx,test_idx,num_coe,g