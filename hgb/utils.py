import random
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics
from torch_sparse import SparseTensor

def add(src,other):
    rowA, colA, valueA = src.coo()
    rowB, colB, valueB = other.coo()
    
    row = torch.cat([rowA, rowB], dim=0)
    col = torch.cat([colA, colB], dim=0)
    
    value: Optional[Tensor] = None
    if valueA is not None and valueB is not None:
        value = torch.cat([valueA, valueB], dim=0)

    M = max(src.size(0), other.size(0))
    N = max(src.size(1), other.size(1))
    sparse_sizes = (M, N)

    out = SparseTensor(row=row, col=col, value=value,
                           sparse_sizes=sparse_sizes)
    out = out.coalesce(reduce='sum')    
    return out

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")

def evaluate(model_pred, labels, dataset):
    labels = labels.cpu()
    if dataset == "IMDB":
        micro,macro = Evaluation(model_pred,labels)

    else:
        pred_result = model_pred.cpu().argmax(dim=1)
        micro = f1_score(labels, pred_result, average='micro')
        macro = f1_score(labels, pred_result, average='macro')

    return micro, macro

def test(model, adjs, feature_list, labels, train_idx, val_idx, test_idx, dataset, LOSS):
    model.eval()
    num_target = feature_list[0].shape[0]
    logits = model(adjs, feature_list)
    logits = logits[:num_target]

    train_micro, train_macro = evaluate(logits[train_idx], labels[train_idx], dataset)
    valid_micro, valid_macro = evaluate(logits[val_idx], labels[val_idx],dataset)
    test_micro, test_macro = evaluate(logits[test_idx], labels[test_idx],dataset)

    val_loss = LOSS(logits[val_idx], labels[val_idx])
    return train_micro, train_macro, valid_micro, valid_macro, test_micro, test_macro, val_loss
