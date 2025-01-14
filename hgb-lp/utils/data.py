import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
from scripts.data_loader import data_loader

def load_data(prefix='DBLP'):
    dl = data_loader('./data/'+prefix)
    

    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)


    num_nodes = dl.nodes['total']
    num_train_links = dl.links['total']
    num_test_links = dl.links_test['total']

    adjM = sum(dl.links['data'].values())
    
    return features,\
           adjM, \
            dl
