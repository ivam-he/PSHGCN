"""
Some methods benefit from SeHGNN (https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn)
"""
import random
import torch
import gc
import numpy as np
from tqdm import tqdm
import dgl.function as fn
from ogb.nodeproppred import Evaluator
import torch.nn.functional as F

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def train(model, train_loader, loss_fcn, optimizer, evaluator, device,
          feats, label_feats, labels_cuda, mask=None, scalar=None):
    model.train()
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        batch_y = labels_cuda[batch]
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output_att = model(batch_feats, batch_labels_feats)
            loss_train = loss_fcn(output_att, batch_y)
        scalar.scale(loss_train).backward()
        scalar.step(optimizer)
        scalar.update()
        y_true.append(batch_y.cpu().to(torch.long))
        y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1

    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc
    

def train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device,
                      feats, label_feats, labels, predict_prob, gama, scalar=None):
    model.train()
    loss_fcn = torch.nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0
    loss_l1, loss_l2 = 0., 0.
    iter_num = 0
    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
        L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

        batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
        y = labels[idx_1].to(torch.long).to(device)
        extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
        extra_weight = extra_weight.to(device)
        extra_y = extra_y.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output_att = model(batch_feats, batch_labels_feats)
            L1 = loss_fcn(output_att[:len(idx_1)],  y)
            L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
            L2 = (L2 * extra_weight).sum() / len(idx_2)
            loss_train = L1_ratio * L1 + gama * L2_ratio * L2
        scalar.scale(loss_train).backward()
        scalar.step(optimizer)
        scalar.update()

        y_true.append(labels[idx_1].to(torch.long))
        y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        loss_l1 += L1.item()
        loss_l2 += L2.item()
        iter_num += 1
    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0),torch.cat(y_pred, dim=0))
    return loss, approx_acc

def propagate_clean(g):
    for ntype in g.ntypes:
        keys = list(g.nodes[ntype].data.keys())
        if len(keys):
            for k in keys:
                g.nodes[ntype].data.pop(k)
    return g

def propagate(g, K):
    #g: dgl heterograph
    #K: multipolynomial oder
    for k in range(1,K+1):
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            for feat_key in list(g.nodes[stype].data.keys()):
                if k == 1 and len(feat_key) == 1:
                    new_feat_key = f'{dtype}{feat_key}'
                    g[etype].update_all(fn.copy_u(feat_key,'m'),fn.mean('m',new_feat_key),etype=etype)
                elif k > 1 and len(feat_key) == 2*(k-1):
                    new_feat_key = f'{dtype}{feat_key[0]}{feat_key}'
                    if k == K and dtype != "P":
                        continue
                    g[etype].update_all(fn.copy_u(feat_key,'m'),fn.mean('m',new_feat_key),etype=etype)

        for ntype in g.ntypes:
            if ntype != "P":
                removes = []
                for feat_key in g.nodes[ntype].data.keys():
                    if (k == 1 and len(feat_key) <= k) or (k >1 and len(feat_key) <= 2*(k-1)):
                        removes.append(feat_key)
                for feat_key in removes:
                    g.nodes[ntype].data.pop(feat_key)
        gc.collect()
    return g

@torch.no_grad()
def gen_output_torch(model, feats, label_feats, test_loader, device):
    model.eval()
    preds = []
    for batch in test_loader:
        batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        preds.append(model(batch_feats, batch_labels_feats).cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

