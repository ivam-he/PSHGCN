import os
import sys
sys.path.append('../../')
import time
import torch
import argparse
import random
import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from utils.pytorchtools import EarlyStopping
from utils.data import load_data

from processing import load_dataset
from pshgcn import PSHGCN

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def training(args):
    set_seed(args.seed)
    
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d-%H:%M:%S")

    _, _, dl = load_data(args.dataset)
    device = args.device
    #######################
    feature_list, adjs, lis, lis_t = load_dataset(args)
    in_dims =[feat.shape[1] for feat in feature_list]
    feature_list = [feat.to(device) for feat in feature_list]
    #######################
    first_flag = True
    res_2hop = defaultdict(float)
    total = len(list(dl.links_test['data'].keys()))
    for test_edge_type in dl.links_test['data'].keys():
        train_pos, valid_pos = dl.get_train_valid_pos()#edge_types=[test_edge_type])
        train_pos = train_pos[test_edge_type]
        valid_pos = valid_pos[test_edge_type]
        
        model = PSHGCN(in_dims, args.output_dim, lis, lis_t, args)
        model.to(device)
        #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW([{ 'params': model.feat_project.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                { 'params': model.lin1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                { 'params': model.lin2.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                {'params': model.W, 'weight_decay': args.prop_wd, 'lr': args.prop_lr}])

        early_stopping = EarlyStopping(patience=args.patience, verbose=False, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, current_time_str))
        loss_func = nn.BCELoss()
        
        for epoch in range(args.epoch):
          train_neg = dl.get_train_neg(edge_types=[test_edge_type])[test_edge_type]
          train_pos_head_full = np.array(train_pos[0])
          train_pos_tail_full = np.array(train_pos[1])
          
          train_neg_head_full = np.array(train_neg[0])
          train_neg_tail_full = np.array(train_neg[1])
          
          train_idx = np.arange(len(train_pos_head_full))
          
          np.random.shuffle(train_idx)
          batch_size = args.batch_size
          for step, start in enumerate(range(0, len(train_pos_head_full), args.batch_size)):
            t_start = time.time()
            # training
            model.train()
            train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
            train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
            
            train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
            train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
            
            left = np.concatenate([train_pos_head, train_neg_head])
            right = np.concatenate([train_pos_tail, train_neg_tail])
            mid = np.zeros(train_pos_head.shape[0]+train_neg_head.shape[0], dtype=np.int32)
            
            labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

            #logits = net(features_list, e_feat, left, right, mid)
            logits = model(adjs, feature_list, left, right, mid)

            logp = torch.sigmoid(logits)
            train_loss = loss_func(logp, labels)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            #print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            model.eval()
            with torch.no_grad():
                valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[test_edge_type]
                valid_pos_head = np.array(valid_pos[0])
                valid_pos_tail = np.array(valid_pos[1])
                valid_neg_head = np.array(valid_neg[0])
                valid_neg_tail = np.array(valid_neg[1])
                left = np.concatenate([valid_pos_head, valid_neg_head])
                right = np.concatenate([valid_pos_tail, valid_neg_tail])
                mid = np.zeros(valid_pos_head.shape[0]+valid_neg_head.shape[0], dtype=np.int32)
                labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                
                #logits = net(features_list, e_feat, left, right, mid)
                logits = model(adjs, feature_list, left, right, mid)
                
                logp = torch.sigmoid(logits)
                val_loss = loss_func(logp, labels)
            t_end = time.time()
            # print validation info
            #print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                #print('Early stopping!')
                break
          if early_stopping.early_stop:
              #print('Early stopping!')
              break

        # testing with evaluate_results_nc
        model.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, current_time_str)))
        model.eval()
        test_logits = []
        with torch.no_grad():
            test_neigh, test_label = dl.get_test_neigh()
            test_neigh = test_neigh[test_edge_type]
            test_label = test_label[test_edge_type]
            # save = np.array([test_neigh[0], test_neigh[1], test_label])
            # print(save)
            # np.savetxt(f"{args.dataset}_{test_edge_type}_label.txt", save, fmt="%i")
            if os.path.exists(os.path.join(dl.path, f"{args.dataset}_ini_{test_edge_type}_label.txt")):
                save = np.loadtxt(os.path.join(dl.path, f"{args.dataset}_ini_{test_edge_type}_label.txt"), dtype=int)
                test_neigh = [save[0], save[1]]
                if save.shape[0] == 2:
                    test_label = np.random.randint(2, size=save[0].shape[0])
                else:
                    test_label = save[2]
            left = np.array(test_neigh[0])
            right = np.array(test_neigh[1])
            mid = np.zeros(left.shape[0], dtype=np.int32)
            mid[:] = test_edge_type
            labels = torch.FloatTensor(test_label).to(device)
            
            logits = model(adjs, feature_list, left, right, mid)

            pred = torch.sigmoid(logits).cpu().numpy()
            edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
            labels = labels.cpu().numpy()
            dl.gen_file_for_evaluate(test_neigh, pred, test_edge_type, file_path=f"{args.dataset}_{args.run}.txt", flag=first_flag)
            first_flag = False
            res = dl.evaluate(edge_list, pred, labels)
            print("Run:",args.seed, "Result:", res)
            for k in res:
                res_2hop[k] += res[k]
    for k in res_2hop:
        res_2hop[k] /= total
    return res_2hop["roc_auc"], res_2hop["MRR"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LP')
    parser.add_argument('--dataset', type=str, default="LastFM")
    parser.add_argument("--seed", type=int, default=1,help="The seed used in the training.")
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--patience', type=int, default=50, help='Patience.')
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--decode', type=str, default='dot')
    parser.add_argument('--run', type=int, default=5)
    parser.add_argument('--device', type=int, default=0, help='Device index.')
    ####################
    ####################
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=16)

    parser.add_argument("--input_drop", type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.6)

    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wd", type=float, default=0.8)
    parser.add_argument("--prop_lr", type=float, default=0.05)
    parser.add_argument("--prop_wd", type=float, default=5e-5)
    args = parser.parse_args()
    args.device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

    os.makedirs('checkpoint', exist_ok=True)
    start_time = time.time()
    results = []
    for i in range(1,args.run+1):
        args.seed = i
        roc, MRR = training(args)
        results.append([roc,MRR])
    end_time = time.time()
    roc_mean, MRR_mean = np.mean(results, axis=0) * 100
    roc_std, MRR_std = np.sqrt(np.var(results, axis=0)) * 100
    print(f'PSHGCN on dataset {args.dataset}, in {args.run} repeated experiment:')
    print(f'ROC mean = {roc_mean:.4f} ± {roc_std:.4f}  \t  MRR mean =  {MRR_mean:.4f} ± {MRR_std:.4f}')
    print(f"Run time: {end_time - start_time} s")
