import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_conv import BatchGCNConv, ChebGraphConv

class EAC_Model(nn.Module):

    def __init__(self, args):
        super(EAC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()

        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))

        self.year = args.year
        self.num_nodes = args.base_node_size

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def forward(self, data, adj):
        N = adj.shape[0]

        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))

        B, N, T = x.shape

        adaptive_params = torch.mm(self.U[:N, :], self.V)
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)

        x = F.relu(self.gcn1(x, adj))
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))

        x = self.tcn1(x)

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))
        x = self.gcn2(x, adj)
        x = x.reshape((-1, self.args.gcn["out_channel"]))

        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            self.num_nodes = new_num_nodes

class TrafficStream_Model(EAC_Model):
    pass

class STKEC_Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(
            args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False
        )
        self.gcn2 = BatchGCNConv(
            args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False
        )
        self.tcn1 = nn.Conv1d(
            in_channels=args.tcn["in_channel"],
            out_channels=args.tcn["out_channel"],
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"],
            padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
        )
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.ReLU()
        self.memory = nn.Parameter(torch.zeros(args.cluster, args.gcn["out_channel"]))
        nn.init.xavier_uniform_(self.memory, gain=1.414)

    def _encode(self, data, adj):
        n = adj.shape[0]
        x = data.x.reshape(-1, n, self.args.gcn["in_channel"])
        x = F.relu(self.gcn1(x, adj))
        x = x.reshape(-1, 1, self.args.gcn["hidden_channel"])
        x = self.tcn1(x)
        x = x.reshape(-1, n, self.args.gcn["hidden_channel"])
        x = self.gcn2(x, adj).reshape(-1, self.args.gcn["out_channel"])
        attention = torch.matmul(x, self.memory.transpose(-1, -2))
        scores = F.softmax(attention, dim=1)
        z = torch.matmul(attention, self.memory)
        return x + data.x + z, scores

    def feature(self, data, adj, scores=None):
        x, _ = self._encode(data, adj)
        return x

    def forward(self, data, adj, scores=None):
        x, scores = self._encode(data, adj)
        pred = self.fc(self.activation(x))
        pred = F.dropout(pred, p=self.dropout, training=self.training)
        return pred, scores

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total}")
        self.args.logger.info(f"Trainable Parameters: {trainable}")

class Universal_Model(EAC_Model):
    pass
