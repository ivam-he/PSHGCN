import os
import sys
import argparse
import torch
import dgl
import gc
import time
import uuid
import numpy as np
import sparse_tools
import torch.nn.functional as F
from processing import load_dataset
from utils import *
from model import PSHGCN

def main(args):
    set_seed(args.seed)
    #load data
    feats,init_labels,expander,train_idx,val_idx,test_idx,coe_num,g = load_dataset(args)

    # rearange node idx (for feats & labels)
    train_nodes = len(train_idx)
    valid_nodes = len(val_idx)
    test_nodes = len(test_idx)
    total_nodes = len(train_idx) + len(val_idx) + len(test_idx) #736389
    num_nodes = total_nodes
    n_classes = int(init_labels.max()) + 1

    init2sort = torch.cat([train_idx, val_idx, test_idx])
    sort2init = torch.argsort(init2sort)
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    labels = init_labels[init2sort]
    feats = {k:v[init2sort] for k,v in feats.items()}
    #data_size = {k: v.size(-1) for k, v in feats.items()}

    all_loader = torch.utils.data.DataLoader(
                torch.arange(num_nodes), 
                batch_size=args.batch_size, 
                shuffle=False, 
                drop_last=False)
    
    scalar = torch.cuda.amp.GradScaler()
    device = "cuda:{}".format(args.dev) if torch.cuda.is_available() else 'cpu'
    labels_cuda = labels.long().to(device)
    evaluator = get_ogb_evaluator(args.dataset)

    ###training begin
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex

    for stage in range(args.stage):
        if stage > 0:
            preds = raw_preds.argmax(dim=-1)
            predict_prob = raw_preds.softmax(dim=1)

            train_acc = evaluator(preds[:train_nodes], labels[:train_nodes])
            val_acc = evaluator(preds[train_nodes:(train_nodes+valid_nodes)], labels[train_nodes:(train_nodes+valid_nodes)])
            test_acc = evaluator(preds[(train_nodes+valid_nodes):total_nodes], labels[(train_nodes+valid_nodes):total_nodes])
            print(f'Stage {stage-1} history model:\n\t'+ 
                    f'Train acc {train_acc*100:.4f} Val acc {val_acc*100:.4f} Test acc {test_acc*100:.4f}')

            confident_mask = predict_prob.max(1)[0] > args.threshold
            val_enhance_offset  = torch.where(confident_mask[train_nodes:(train_nodes+valid_nodes)])[0]
            test_enhance_offset = torch.where(confident_mask[(train_nodes+valid_nodes):total_nodes])[0]
            val_enhance_nid     = val_enhance_offset + train_nodes
            test_enhance_nid    = test_enhance_offset + train_nodes+valid_nodes
            enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

            print(f'Stage: {stage}, threshold {args.threshold}, confident nodes: {len(enhance_nid)} / {total_nodes - train_nodes}')
            val_confident_level = (predict_prob[val_enhance_nid].argmax(1) == labels[val_enhance_nid]).sum() / len(val_enhance_nid)
            print(f'\t\t val confident nodes: {len(val_enhance_nid)} / {valid_nodes},  val confident level: {val_confident_level}')
            test_confident_level = (predict_prob[test_enhance_nid].argmax(1) == labels[test_enhance_nid]).sum() / len(test_enhance_nid)
            print(f'\t\ttest confident nodes: {len(test_enhance_nid)} / {test_nodes}, test confident_level: {test_confident_level}')

            del train_loader
            train_batch_size = int(args.batch_size * len(train_idx) / (len(enhance_nid) + len(train_idx)))
            train_loader = torch.utils.data.DataLoader(
                            torch.arange(train_nodes), 
                            batch_size=train_batch_size, 
                            shuffle=True, 
                            drop_last=False)
            enhance_batch_size = int(args.batch_size * len(enhance_nid) / (len(enhance_nid) + len(train_idx)))
            enhance_loader = torch.utils.data.DataLoader(
                            enhance_nid, 
                            batch_size=enhance_batch_size, 
                            shuffle=True, 
                            drop_last=False)
        else:
            train_loader = torch.utils.data.DataLoader(
                            torch.arange(train_nodes), 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            drop_last=False)
        if stage > 0:
            label_onehot = predict_prob[sort2init].clone()
        else:
            label_onehot = torch.zeros((num_nodes, n_classes))
        label_feats = {}
        label_onehot[train_idx] = F.one_hot(init_labels[train_idx], n_classes).float()
        ###label processing
        g.nodes['P'].data['P'] = label_onehot
        g = propagate(g, args.K)
        keys = list(g.nodes["P"].data.keys())
        for k in keys:
            if k == "P": continue
            label_feats[k] = g.nodes["P"].data.pop(k)
        g = propagate_clean(g)
        for k in ['PPPP', 'PAAP', 'PFFP']:
            if k in label_feats:
                diag = torch.load(f'./data/{args.dataset}_{k}_diag.pt')
                label_feats[k] = label_feats[k] - diag.unsqueeze(-1) * label_onehot
                assert torch.all(label_feats[k] > -1e-6)
                print(k, torch.sum(label_feats[k] < 0), label_feats[k].min())
        label_feats = {k: v[init2sort] for k, v in label_feats.items()}
        
        if stage > 0:
            del eval_loader
        eval_loader = []
        for batch_idx in range((total_nodes-train_nodes-1) // args.batch_size + 1):
            batch_start = batch_idx * args.batch_size + train_nodes
            batch_end = min(num_nodes, (batch_idx+1) * args.batch_size + train_nodes)
            batch_feats = {k: v[batch_start:batch_end] for k,v in feats.items()}
            batch_label_feats = {k: v[batch_start:batch_end] for k,v in label_feats.items()}
            eval_loader.append((batch_feats, batch_label_feats))

        node_types = ['P','A','I','F']
        label_keys = ['PPPP', 'PAAP', 'PP', 'PFFP']
        model = PSHGCN(args.emb_dim, args.hidden_x, args.hidden_l, n_classes, node_types, label_keys, args.layers_x,
                    args.layers_l, coe_num, expander, args.dropout, args.input_drop, args.bias).to(device)

        print("#Params:", get_n_params(model))
        loss_fcn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_epoch = 0
        best_val_acc = 0
        best_test_acc = 0
        count = 0
        for epoch in range(args.epochs):
            gc.collect()
            torch.cuda.empty_cache()
            if stage == 0:
                loss, acc = train(model, train_loader, loss_fcn, optimizer, evaluator, 
                            device, feats, label_feats, labels_cuda, scalar=scalar)
            else:
                loss, acc = train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, 
                            evaluator, device, feats, label_feats, labels_cuda, predict_prob, args.gama, scalar=scalar)
            log = "Epoch {},estimated train loss {:.4f}, acc {:.4f}\n".format(epoch,loss, acc*100)           
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                raw_preds = []
                for batch_feats, batch_label_feats in eval_loader:
                    batch_feats = {k: v.to(device) for k,v in batch_feats.items()}
                    batch_label_feats = {k: v.to(device) for k,v in batch_label_feats.items()}
                    raw_preds.append(model(batch_feats, batch_label_feats).cpu())
                
                raw_preds = torch.cat(raw_preds, dim=0)
                loss_val = loss_fcn(raw_preds[:valid_nodes], labels[train_nodes:(train_nodes+valid_nodes)]).item()
                loss_test = loss_fcn(raw_preds[valid_nodes:(valid_nodes+test_nodes)], labels[(train_nodes+valid_nodes):total_nodes]).item()
                preds = raw_preds.argmax(dim=-1)
                val_acc = evaluator(preds[:valid_nodes], labels[train_nodes:(train_nodes+valid_nodes)])
                test_acc = evaluator(preds[valid_nodes:(valid_nodes+test_nodes)], labels[(train_nodes+valid_nodes):total_nodes])                
                log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(val_acc*100, test_acc*100)
            if val_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = val_acc
                best_test_acc = test_acc
                torch.save(model.state_dict(), f'{checkpt_file}_{stage}.pkl')
                count = 0
            else:
                count += 1
                if count >= args.patience:
                    break
            log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100)
            print(log, flush=True)
        model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
        raw_preds = gen_output_torch(model, feats, label_feats, all_loader, device)
        print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100))
    
    return best_val_acc, best_test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHGCN')
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dev", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--hidden_x", type=int, default=1024)
    parser.add_argument("--hidden_l", type=int, default=512)
    parser.add_argument("--layers_x", type=int, default=3)
    parser.add_argument("--layers_l", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input_drop", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--stage", type=int,default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--gama", type=float, default=10)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--bias", action='store_true', default=False)
    parser.add_argument("--extra_emb", action='store_true',default=False)
    args = parser.parse_args()
    print(args)

    results = []
    for run in range(1,args.runs+1):
        print("Run:",run)
        args.seed = run
        val_acc, test_acc = main(args)
        results.append([val_acc,test_acc])
    val_acc_mean, test_acc_mean = np.mean(results, axis=0) * 100
    val_acc_std, test_acc_std = np.sqrt(np.var(results, axis=0)) * 100
    print(f'Val acc = {val_acc_mean:.2f} ± {val_acc_std*100:.2f}  \t Test acc = {test_acc_mean:.2f} ± {test_acc_std*100:.2f}')

