import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import entropy as kldiv
from datetime import datetime
from torch_geometric.utils import to_dense_batch
from dataer.SpatioTemporalDataset import continue_learning_Dataset
from torch_geometric.loader import DataLoader
import torch
from scipy.spatial import distance
import os.path as osp

def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size))
    dataloader = DataLoader(continue_learning_Dataset(data), batch_size=data.shape[0], shuffle=False, pin_memory=True, num_workers=32)
    for data in dataloader:
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(data, adj), batch=data.batch)
        node_size = feature.size()[1]
        feature = feature.permute(1,0,2)
        return feature.cpu().detach().numpy()

def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)

def score_func(pre_data, cur_data, args):
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]

def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    if args.detect_strategy == 'original':
        pre_data = pre_data[-288*7-1:-1,:]
        cur_data = cur_data[-288*7-1:-1,:]
        node_size = pre_data.shape[1]
        score = []
        for node in range(node_size):
            max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
            min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
            pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            score.append(kldiv(pre_prob, cur_prob))
        return score
    elif args.detect_strategy == 'feature':
        model.eval()
        pre_adj = get_adj(args.year-1, args)
        cur_adj = get_adj(args.year, args)

        pre_data = get_feature(pre_data, pre_graph, args, model, pre_adj)
        cur_data = get_feature(cur_data, cur_graph, args, model, cur_adj)
        score = []
        for i in range(pre_data.shape[0]):
            score_ = 0.0
            for j in range(pre_data.shape[2]):

                pre_data[i,:,j] = (pre_data[i,:,j] - min(pre_data[i,:,j]))/(max(pre_data[i,:,j]) - min(pre_data[i,:,j]))
                cur_data[i,:,j] = (cur_data[i,:,j] - min(cur_data[i,:,j]))/(max(cur_data[i,:,j]) - min(cur_data[i,:,j]))

                pre_prob, _ = np.histogram(pre_data[i,:,j], bins=10, range=(0, 1))
                pre_prob = pre_prob *1.0 / sum(pre_prob)
                cur_prob, _ = np.histogram(cur_data[i,:,j], bins=10, range=(0, 1))
                cur_prob = cur_prob * 1.0 /sum(cur_prob)
                score_ += distance.jensenshannon(pre_prob, cur_prob)
            score.append(score_)
        return score
    else: args.logger.info("node selection mode illegal!")
