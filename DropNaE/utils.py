import torch
import numpy as np
import scipy.sparse as sp
from random import sample
import json
import sys
import networkx as nx
import os
from networkx.readwrite import json_graph
from utils import *

def DropNaE(p_s, p_d, prefix, Drop_type = 'E'):
    """
    Drop node and edge to get new adj

    Args:
        p_s (float): ratio of score to choose drop node
        p_d (float): ratio of degree to choose drop node
        prefix (string): directory containing the above graph related files
        Drop_type (string): choose DropNaE-N or DropNaE-E
    others:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test train_nodes.
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test train_nodes in adj_train are all-zero.
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test train_nodes. The value is the list of IDs of train_nodes belonging to
                            the train/val/test sets.
    Returns:
        sp.csr_matrix: new adj with drop node/edge
    """
    adj_full = sp.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}

    edge_num = adj_train.nnz
    scores = torch.load('{}/score.pt'.format(prefix))
    degree = np.sum(adj_train,0).tolist()[0]
    train_node = role['tr']

    node_num = len(train_node)
    train_drop = del_node(degree, train_node, score=scores, per_s=p_s, per_d=p_d)
    Drop_node_num = len(train_drop)

    if Drop_type == 'E':
        # new_adj_full, Drop_edge_num = del_edge(adj=adj_full, del_node_list=train_drop, class_map=class_map)
        new_adj_train, Drop_edge_num = del_edge(adj=adj_full, del_node_list=train_drop, class_map=class_map)
        # sp.save_npz('./{}/adj_full_DE.npz'.format(prefix),new_adj_full)
        sp.save_npz('./{}/adj_train_E.npz'.format(prefix),new_adj_train)
    else:
        new_adj_train = del_rows_from_csr_mtx(adj_train, train_drop)
        # new_adj_full = del_rows_from_csr_mtx(adj_full, train_drop)
        Drop_edge_num = edge_num - new_adj_train.nnz
        # sp.save_npz('./{}/adj_full_DR.npz'.format(prefix),new_adj_full)
        sp.save_npz('./{}/adj_train_N.npz'.format(prefix),new_adj_train)

    print("drop edge num : {:.2f}".format(Drop_edge_num))
    print("drop edge rate : {:.4f}".format(Drop_edge_num/edge_num))
    print("drop node num : {:.2f}".format(Drop_node_num))
    print("Drop node rata : {:.4f}".format(Drop_node_num/node_num))

def DropNaE_pg(p_s, p_d, prefix, Drop_type = 'E'):
    adj_full = sp.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}

    edge_num = adj_train.nnz
    scores = torch.load('{}/score.pt'.format(prefix))
    degree = np.sum(adj_train,0).tolist()[0]
    train_node = role['tr']

    node_num = len(train_node)
    train_drop = del_node(degree, train_node, score=scores, per_s=p_s, per_d=p_d)
    Drop_node_num = len(train_drop)

    if Drop_type == 'E':
        # new_adj_full, Drop_edge_num = del_edge(adj=adj_full, del_node_list=train_drop, class_map=class_map)
        new_adj_train, Drop_edge_num = del_edge(adj=adj_full, del_node_list=train_drop, class_map=class_map)
        # sp.save_npz('./{}/adj_full_DE_pg.npz'.format(prefix),new_adj_full)
        sp.save_npz('./{}/adj_train_E_pg.npz'.format(prefix),new_adj_train)
    else:
        new_adj_train = del_rows_from_csr_mtx(adj_train, train_drop)
        # new_adj_full = del_rows_from_csr_mtx(adj_full, train_drop)
        Drop_edge_num = edge_num - new_adj_train.nnz
        # sp.save_npz('./{}/adj_full_DR_pg.npz'.format(prefix),new_adj_full)
        sp.save_npz('./{}/adj_train_N_pg.npz'.format(prefix),new_adj_train)

    print("drop edge num : {:.2f}".format(Drop_edge_num))
    print("drop edge rate : {:.4f}".format(Drop_edge_num/edge_num))
    print("drop node num : {:.2f}".format(Drop_node_num))
    print("Drop node rata : {:.4f}".format(Drop_node_num/node_num))

def DropNaE_clu(p_s, p_d, dataset_str, Drop_type = 'E'):
    prefix = 'data/' + dataset_str
    baseline_str='data.ignore/'+dataset_str+'/'
    if not os.path.exists(baseline_str[:-1]):
        os.mkdir(baseline_str[:-1])
    adj_full = sp.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = sp.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}

    edge_num = adj_train.nnz
    scores = torch.load('{}/score.pt'.format(prefix))
    degree = np.sum(adj_train,0).tolist()[0]
    train_node = role['tr']

    node_num = len(train_node)
    train_drop = del_node(degree, train_node, score=scores, per_s=p_s, per_d=p_d)
    Drop_node_num = len(train_drop)

    if Drop_type == 'E':
        new_adj_full, Drop_edge_num = del_edge(adj=adj_full, del_node_list=train_drop, class_map=class_map)
        new_adj_train, Drop_edge_num = del_edge(adj=adj_full, del_node_list=train_drop, class_map=class_map)
        # G.json
        print("Conver to sage format")
        G=nx.from_scipy_sparse_matrix(new_adj_full)
        print('nx: finish load graph')
        data=json_graph.node_link_data(G)

        te=set(role['te'])
        va=set(role['va'])
        for node in data['nodes']:
            node['test']=False
            node['val']=False
            if node['id'] in te:
                node['test']=True
            elif node['id'] in va:
                node['val']=True
        for edge in data['links']:
            del edge['weight']
            edge['target']=int(edge['target'])
        with open(baseline_str+'G_E.json','w') as f:
            json.dump(data,f)
            # id_map.json
        id_map={}
        for i in range(G.number_of_nodes()):
            id_map[str(i)]=i
        with open(baseline_str+'id_map_E.json','w') as f:
            json.dump(id_map,f)
    else:
        new_adj_train = del_rows_from_csr_mtx(adj_train, train_drop)
        new_adj_full = del_rows_from_csr_mtx(adj_full, train_drop)
        Drop_edge_num = edge_num - new_adj_train.nnz
        # G.json
        print("Conver to sage format")
        G=nx.from_scipy_sparse_matrix(new_adj_full)
        print('nx: finish load graph')
        data=json_graph.node_link_data(G)

        te=set(role['te'])
        va=set(role['va'])
        for node in data['nodes']:
            node['test']=False
            node['val']=False
            if node['id'] in te:
                node['test']=True
            elif node['id'] in va:
                node['val']=True
        for edge in data['links']:
            del edge['weight']
            edge['target']=int(edge['target'])
        with open(baseline_str+'G_N.json','w') as f:
            json.dump(data,f)
            # id_map.json
        id_map={}
        for i in range(G.number_of_nodes()):
            id_map[str(i)]=i
        with open(baseline_str+'id_map_N.json','w') as f:
            json.dump(id_map,f)

    print("drop edge num : {:.2f}".format(Drop_edge_num))
    print("drop edge rate : {:.4f}".format(Drop_edge_num/edge_num))
    print("drop node num : {:.2f}".format(Drop_node_num))
    print("Drop node rata : {:.4f}".format(Drop_node_num/node_num))
    # feats.npy
    feats=np.load(dataset_str+'feats.npy')
    np.save(baseline_str+'feats.npy',feats)

    # class_map.json
    class_map=json.load(open(dataset_str+'class_map.json','r'))
    for k,v in class_map.items():
        class_map[k]=v
    with open(baseline_str+'class_map.json','w') as f:
        json.dump(class_map,f)
    

def del_node(degree, train_node, score, per_s, per_d):
    """
    get drop node list wiht node degree > max_D and node score > max_S

    Args:
        degree (list): node degree list
        train_node (list): train node id list
        score (list): node score list
        per_s (float): choose top per_s node to drop
        per_d (float): choose top per_s node to drop

    Returns:
        list: drop node id list
    """
    float
    node_num = len(degree)
    mask = np.zeros(node_num)
    mask[train_node] = 1
    tr_score = mask * score
    
    score1 = tr_score.tolist()
    score1.sort(reverse = True)
    K1 = int(len(train_node)*per_s)
    max_S = score1[K1]

    degree = np.array(degree)
    val_d = degree * mask
    val_d = val_d.tolist()
    val_d.sort(reverse = True)
    K2 = int(len(train_node)*per_d)
    max_D = val_d[K2]
    
    nodes = []
    for i in train_node:
        if score[i] >= max_S and degree[i] >= max_D:
            nodes.append(i)
    return nodes

def del_edge(adj, del_node_list, class_map = None):
    n = adj.shape[0]
    adj = adj.toarray()
    p = 0.7
    drop_edge_num = 0
    for i in del_node_list:
        neighbor_id = list(adj[i].nonzero()[0])
        neighbor_id = sample(neighbor_id, int(len(neighbor_id)*p))

        for j in neighbor_id:
            if class_map[j] != class_map[i]:
                adj[i][j] = 0
                adj[j][i] = 0
                drop_edge_num += 1

    adj = torch.from_numpy(adj)
    edge = adj.nonzero()
    edge_index = np.array(edge.T)
    val = np.ones(edge_index.shape[1])
    adj = sp.coo_matrix((val,edge_index),shape=(n,n))
    return adj.tocsr(), drop_edge_num

def del_rows_from_csr_mtx(csr_mtx, row_indices):
    """ 
    delete rows in csr matrix
    """

    indptr = csr_mtx.indptr
    indices = csr_mtx.indices
    data = csr_mtx.data
    m, n = csr_mtx.shape

    target_row_ele_indices = [i for idx in row_indices for i in range(indptr[idx], indptr[idx+1])]
    new_indices = np.delete(indices, target_row_ele_indices)
    new_data = np.delete(data, target_row_ele_indices)

    off_vec = np.zeros((m+1,), dtype=np.int)
    for idx in row_indices:
        off_vec[idx+1:] = off_vec[idx+1:] + (indptr[idx+1] - indptr[idx])
    new_indptr = indptr - off_vec

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=(m, n))
