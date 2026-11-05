from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import EAC_Model

def _mlp(d_in: int, d_hid: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))

class LowRankAdapter(nn.Module):

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, dim) / math.sqrt(dim))
        self.B = nn.Parameter(torch.zeros(dim, rank))

    def forward(self, h):
        return F.gelu(h @ self.A.t()) @ self.B.t()

class EvoCoPlugin:
    FREEZE_PREFIXES = "__none__"

    def _init_coevo(self, args, trunk_dim: int):
        self.L = int(getattr(args, "x_len", None) or args.gcn["in_channel"])
        self.H = int(args.y_len)
        d = int(getattr(args, "coevo_dim", 64))
        dd = int(getattr(args, "coevo_trans_dim", 32))
        c_h = int(getattr(args, "caqra_cnn_hidden", 48))
        self.d = d
        self.img_ch = int(getattr(args, "caqra_image_channels", 3))
        self.anchor_k = int(getattr(args, "caqra_anchor_topk", 4))
        self.lag = int(getattr(args, "coevo_graph_lag_steps", 3))
        self.kappa = float(getattr(args, "coevo_kappa", 0.0) or 0.0)
        self.ema_m = float(getattr(args, "coevo_ema_momentum", 0.99))
        self.c_temp = float(getattr(args, "coevo_c_temp", 1.0))

        self.g_proj = nn.Linear(trunk_dim, d)
        self.g_norm = nn.LayerNorm(d)
        self.img_stem = nn.Sequential(
            nn.Conv2d(self.img_ch, c_h, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(c_h, c_h, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_h, c_h, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.tok_proj = nn.Linear(c_h, d)
        self.v_norm = nn.LayerNorm(d)

        self.psi_n = _mlp(3 * d, 2 * dd, dd)
        self.psi_v = _mlp(3 * d, 2 * dd, dd)
        self.f_n2v = _mlp(dd + d, 2 * dd, dd)
        self.f_v2n = _mlp(dd + d, 2 * dd, dd)
        self.w_zn = nn.Linear(dd, dd)
        self.w_zv = nn.Linear(dd, dd)

        rk = int(getattr(args, "coevo_adapter_rank", 8))
        self.adapter_n = LowRankAdapter(d, rk)
        self.adapter_v = LowRankAdapter(d, rk)
        self.gate_n = nn.Linear(dd, 1)
        self.gate_v = nn.Linear(dd, 1)
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(2 * d, d)
        self.w_v = nn.Linear(d, d)
        self.w_alpha = nn.Linear(d + dd, 1)
        self.f_norm = nn.LayerNorm(d)
        self.w_y = nn.Linear(d, self.H)
        for layer in (self.w_y, self.w_alpha, self.gate_n, self.gate_v):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        size = int(getattr(args, "caqra_image_size", 128))
        self.register_buffer("anchor_idx", torch.zeros(1, self.anchor_k, dtype=torch.long), persistent=False)
        self.register_buffer("anchor_wgt", torch.zeros(1, self.anchor_k), persistent=False)
        self.register_buffer("anchor_mat", torch.zeros(1, 1), persistent=False)
        self.register_buffer("img_mean", torch.zeros(self.img_ch, size, size), persistent=False)
        self.register_buffer("img_std", torch.ones(1), persistent=False)
        self.register_buffer("steps", torch.arange(1, self.H + 1, dtype=torch.float32), persistent=False)
        self.register_buffer("s_bar_n", torch.zeros(1))
        self.register_buffer("s_bar_v", torch.zeros(1))
        self.n_feat = (0, 0)
        self._aux: dict = {}

    def set_year_anchor(self, args, num_nodes: int, device=None):
        from utils.geo_anchor import build_topk_anchor

        dev = device or next(self.parameters()).device
        size = int(getattr(args, "caqra_image_size", 128))
        with torch.no_grad():
            hf, wf = self.img_stem(torch.zeros(1, self.img_ch, size, size, device=dev)).shape[-2:]
        if bool(getattr(args, "caqra_normalize", True)):
            from utils.caqra_io import frame_statistics

            try:
                mean, std = frame_statistics(args, int(args.year))
                self.img_mean = torch.from_numpy(np.asarray(mean)).to(dev)
                self.img_std = torch.tensor([float(std)], device=dev)
            except Exception as exc:
                args.logger.info(f"[EvoCo] frame statistics unavailable ({exc}); using raw frames")
        idx, wgt, _ = build_topk_anchor(args, int(args.year), num_nodes, int(hf), int(wf), self.anchor_k)
        if idx is None:
            raise RuntimeError("EvoCo needs station coordinates (station_meta_pattern) to anchor visual tokens")
        self.anchor_idx = idx.to(dev)
        self.anchor_wgt = wgt.to(dev)
        mat = torch.zeros(num_nodes, int(hf) * int(wf), device=dev)
        self.anchor_mat = mat.scatter_add_(1, self.anchor_idx, self.anchor_wgt)
        self.n_feat = (int(hf), int(wf))

    def _lagged(self, x):
        k = max(1, min(self.lag, self.L - 1))
        return torch.cat([x[:, :, :1].expand(-1, -1, k), x[:, :, :-k]], dim=-1)

    def encode_numeric(self, x, adj):
        b = x.shape[0]
        trunk = self._trunk(torch.cat([x, self._lagged(x)], dim=0), adj)
        h = self.g_norm(self.g_proj(trunk.detach()))
        return h[:b], h[b:], trunk[:b]

    def encode_visual(self, img, n):
        b = img.shape[0]
        frames = img[:, : 2 * self.img_ch].reshape(b * 2, self.img_ch, img.shape[-2], img.shape[-1])
        frames = (frames - self.img_mean) / self.img_std
        feat = self.img_stem(frames)
        tok = self.tok_proj(feat.flatten(2).transpose(1, 2)).view(b, 2, -1, self.d)
        p = tok.shape[2]
        h_v = self.v_norm(torch.einsum("np,bkpd->bknd", self.anchor_mat[:n, :p], tok))
        nb = tok[:, :, self.anchor_idx[:n].reshape(-1)].view(b, 2, n, self.anchor_k, self.d)
        return h_v[:, 0], h_v[:, 1], nb[:, 0], nb[:, 1]

    @staticmethod
    def _evolution(psi, h, h_prev):
        return psi(torch.cat([h, h_prev, h - h_prev], dim=-1))

    def coordinate(self, d_n, d_v, h_n_prev, h_v_prev):
        d_v_hat = self.f_n2v(torch.cat([d_n, h_n_prev], dim=-1))
        d_n_hat = self.f_v2n(torch.cat([d_v, h_v_prev], dim=-1))
        r_n, r_v = d_n - d_n_hat, d_v - d_v_hat

        if self.training:
            self.s_bar_n.mul_(self.ema_m).add_((1.0 - self.ema_m) * d_n.detach().pow(2).mean())
            self.s_bar_v.mul_(self.ema_m).add_((1.0 - self.ema_m) * d_v.detach().pow(2).mean())

        eta_n = r_n.pow(2).mean(-1, keepdim=True) / (self.s_bar_n + 1e-6)
        eta_v = r_v.pow(2).mean(-1, keepdim=True) / (self.s_bar_v + 1e-6)
        c = torch.exp(-(eta_n + eta_v) / (2.0 * self.c_temp)).detach()

        z_c = c * (self.w_zn(d_n_hat.detach()) + self.w_zv(d_v_hat.detach()))
        return d_n_hat, d_v_hat, r_n, r_v, c, z_c

    def extract(self, h_n, tok, tok_prev, n):
        q = self.w_q(h_n).unsqueeze(2)
        k = self.w_k(torch.cat([tok, tok - tok_prev], dim=-1))
        bias = torch.log(self.anchor_wgt[:n].clamp_min(1e-6))
        att = torch.softmax((q * k).sum(-1) / math.sqrt(self.d) + bias, dim=-1)
        return (att.unsqueeze(-1) * self.w_v(tok)).sum(dim=2), att

    def forward(self, data, adj):
        n = adj.shape[0]
        x = data.x.reshape(-1, n, self.L)

        h_n, h_n_prev, trunk = self.encode_numeric(x, adj)
        y_n = self.fc(self.activation(trunk))
        h_v, h_v_prev, tok, tok_prev = self.encode_visual(data.img, n)

        d_n = self._evolution(self.psi_n, h_n, h_n_prev)
        d_v = self._evolution(self.psi_v, h_v, h_v_prev)
        d_n_hat, d_v_hat, r_n, r_v, c, z_c = self.coordinate(d_n, d_v, h_n_prev, h_v_prev)

        h_n = h_n + torch.sigmoid(self.gate_n(r_n.detach())) * self.adapter_n(h_n)
        h_v = h_v + torch.sigmoid(self.gate_v(r_v.detach())) * self.adapter_v(h_v)
        ctx, att = self.extract(h_n, tok, tok_prev, n)

        alpha = torch.sigmoid(self.w_alpha(torch.cat([h_v, z_c], dim=-1)))
        h_f = self.f_norm(h_n + alpha * ctx)
        delta = self.w_y(h_f)
        if self.kappa > 0:
            delta = self.kappa * torch.tanh(delta / self.kappa)
        y = F.dropout(y_n.detach() + delta, p=self.dropout, training=self.training)

        self._aux = {
            "y_n": y_n, "h_n": h_n, "h_v": h_v, "d_n": d_n, "d_v": d_v,
            "d_n_hat": d_n_hat, "d_v_hat": d_v_hat,
            "r_n": r_n, "r_v": r_v, "c": c, "z_c": z_c, "att": att,
            "alpha": alpha, "ctx": ctx, "h_f": h_f, "delta": delta,
        }
        return y.reshape(-1, self.H)

    def map_loss(self):
        a = self._aux
        return F.l1_loss(a["d_v_hat"], a["d_v"].detach()) + F.l1_loss(a["d_n_hat"], a["d_n"].detach())

    def variance_loss(self):
        a = self._aux
        return sum(F.relu(1.0 - z.reshape(-1, z.shape[-1]).float().std(dim=0)).mean()
                   for z in (a["d_n"], a["d_v"], a["h_v"]))

    def online_params(self):
        spec = str(getattr(self.args, "coevo_freeze_prefixes", "") or self.FREEZE_PREFIXES)
        frozen = tuple(s.strip() for s in spec.split(",") if s.strip() and s.strip() != "__none__")
        online, base = [], []
        for name, prm in self.named_parameters():
            (base if name.startswith(frozen) else online).append((name, prm))
        return online, base

class CoEvo_ST_Model(EvoCoPlugin, EAC_Model):
    FREEZE_PREFIXES = "gcn1,gcn2,tcn1,fc."

    def __init__(self, args):
        EAC_Model.__init__(self, args)
        self._init_coevo(args, int(args.gcn["out_channel"]))

    def _trunk(self, x, adj):
        n = x.shape[1]
        h = x + torch.mm(self.U[:n, :], self.V).unsqueeze(0)
        h = F.relu(self.gcn1(h, adj))
        h = self.tcn1(h.reshape(-1, 1, self.args.gcn["hidden_channel"]))
        h = self.gcn2(h.reshape(-1, n, self.args.gcn["hidden_channel"]), adj)
        return h + x
