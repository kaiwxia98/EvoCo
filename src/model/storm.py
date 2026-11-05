from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def scaled_laplacian(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    if lam < 1e-8:
        return -np.eye(n)
    return 2 * L / lam - np.eye(n)

def cheb_poly(L: np.ndarray, ks: int) -> np.ndarray:
    n = L.shape[0]
    ll = [np.eye(n), L.copy()]
    for _ in range(2, ks):
        ll.append(np.matmul(2 * L, ll[-1]) - ll[-2])
    return np.asarray(ll[:ks])

def adj_to_lk(adj, ks: int, device) -> torch.Tensor:
    if isinstance(adj, torch.Tensor):
        a = adj.detach().float().cpu().numpy()
    else:
        a = np.asarray(adj, dtype=np.float64)
    a = (a + a.T) * 0.5
    L = scaled_laplacian(a)
    lk = cheb_poly(L, ks)
    return torch.tensor(lk.astype(np.float32), device=device)

def _expand_layernorm(ln: nn.LayerNorm, new_n: int):

    old_n, c = ln.weight.shape
    if new_n <= old_n:
        return
    w = torch.ones(new_n, c, device=ln.weight.device, dtype=ln.weight.dtype)
    b = torch.zeros(new_n, c, device=ln.bias.device, dtype=ln.bias.dtype)
    w[:old_n] = ln.weight.data
    b[:old_n] = ln.bias.data
    ln.weight = nn.Parameter(w)
    ln.bias = nn.Parameter(b)
    ln.normalized_shape = (new_n, c)

def apply_node_ln(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:

    n, c = int(x.shape[-2]), int(x.shape[-1])
    w, b = ln.weight, ln.bias
    n_ln = int(w.shape[0])
    if n_ln == n:
        return ln(x)
    if n_ln > n:
        return F.layer_norm(x, (n, c), w[:n], b[:n], ln.eps)
    w2 = w.new_ones(n, c)
    b2 = b.new_zeros(n, c)
    w2[:n_ln] = w
    b2[:n_ln] = b
    return F.layer_norm(x, (n, c), w2, b2, ln.eps)

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x

class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super().__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1 :, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, : self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out :, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, lk):
        super().__init__()
        self.register_buffer("Lk", lk, persistent=False)
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc + x)

class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, lk):
        super().__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconv = spatio_conv_layer(ks, c[1], lk)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = apply_node_ln(self.ln, x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)

class output_layer(nn.Module):
    def __init__(self, c, t_left, n, horizon):
        super().__init__()
        self.tconv1 = temporal_conv_layer(t_left, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = nn.Conv2d(1, horizon, kernel_size=(1, c), bias=True)

    def features(self, x):
        x_t1 = self.tconv1(x)
        x_ln = apply_node_ln(self.ln, x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.tconv2(x_ln).permute(0, 2, 3, 1)

    def forward(self, x):
        return self.fc(self.features(x))

class STGCN(nn.Module):
    def __init__(self, ks, kt, bs, t_in, n, lk, p):
        super().__init__()
        self.ks = ks
        self.num_nodes = n
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, lk)
        self.output = output_layer(bs[1][2], t_in - 4 * (kt - 1), n, t_in)

    def replace_lk(self, lk):
        self.st_conv1.sconv.Lk = lk
        self.st_conv2.sconv.Lk = lk

    def expand_nodes(self, new_n: int, lk=None):
        new_n = int(new_n)
        _expand_layernorm(self.st_conv1.ln, new_n)
        _expand_layernorm(self.st_conv2.ln, new_n)
        _expand_layernorm(self.output.ln, new_n)
        if lk is not None:
            self.replace_lk(lk)
        self.num_nodes = max(self.num_nodes, new_n)

    def encode(self, x):

        h = x.permute(0, 3, 1, 2)
        h = self.st_conv2(self.st_conv1(h))
        feat = self.output.features(h)
        return feat.squeeze(1)

    def forward(self, x):

        h = x.permute(0, 3, 1, 2)
        return self.output(self.st_conv2(self.st_conv1(h)))

class StormHead(nn.Module):

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, trunk):
        out = self.conv(trunk.unsqueeze(1))
        return out.squeeze(-1).permute(0, 2, 1)

class Storm_Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = float(getattr(args, "dropout", 0.0) or 0.0)
        self.ks = int(getattr(args, "storm_ks", 2))
        self.kt = int(getattr(args, "storm_kt", 3))
        channels = getattr(args, "storm_channels", [[1, 16, 64], [64, 16, 64]])
        if isinstance(channels, str):
            channels = eval(channels)
        self.channels = channels
        self.t_in = int(getattr(args, "x_len", 12))
        self.horizon = int(getattr(args, "y_len", 12))
        n = int(getattr(args, "base_node_size", getattr(args, "graph_size", 1)))
        p = float(getattr(args, "dropout", 0.0))
        device = getattr(args, "device", "cpu")
        dummy_lk = torch.zeros(self.ks, n, n, device=device)
        self.net = STGCN(self.ks, self.kt, channels, self.t_in, n, dummy_lk, p)
        self.fc = StormHead(self.net.output.fc)
        self.activation = nn.Identity()
        self.num_nodes = n
        self._lk_n = -1
        log_fn = args.logger.info if hasattr(args, "logger") else print
        log_fn(f"Storm STGCN initialized with n={n}, channels={channels}")

    def load_state_dict(self, state_dict, strict=True):
        sd = dict(state_dict)
        for key in list(sd.keys()):
            if key.endswith(".Lk"):
                sd.pop(key)
        return super().load_state_dict(sd, strict=strict)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_fn = self.args.logger.info if hasattr(self.args, "logger") else print
        log_fn(f"Total Parameters: {total}")
        log_fn(f"Trainable Parameters: {trainable}")

    def _sync_graph(self, adj):
        n = int(adj.shape[0])
        if n > self.num_nodes:
            self.expand_adaptive_params(n)
        if n != self._lk_n or self.net.st_conv1.sconv.Lk.shape[-1] != n:
            lk = adj_to_lk(adj, self.ks, adj.device)
            self.net.replace_lk(lk)
            self._lk_n = n
        return n

    def expand_adaptive_params(self, new_num_nodes):
        new_num_nodes = int(new_num_nodes)
        if new_num_nodes <= self.num_nodes:
            return
        device = next(self.parameters()).device
        dummy = torch.zeros(self.ks, new_num_nodes, new_num_nodes, device=device)
        self.net.expand_nodes(new_num_nodes, dummy)
        self.num_nodes = new_num_nodes
        self._lk_n = -1

    def _as_btn(self, data, n):
        x = data.x
        if x.dim() == 2:
            x = x.reshape(-1, n, self.t_in)
        return x.permute(0, 2, 1).unsqueeze(-1).contiguous()

    def trunk_features(self, data, adj):
        n = self._sync_graph(adj)
        return self.net.encode(self._as_btn(data, n))

    def forward(self, data, adj):
        n = self._sync_graph(adj)
        out = self.net(self._as_btn(data, n))
        return out.squeeze(-1).permute(0, 2, 1).contiguous().reshape(-1, self.horizon)
