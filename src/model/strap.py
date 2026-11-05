from __future__ import annotations

import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gcn_conv import BatchGCNConv

class STGNN_Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn = nn.Conv1d(
            in_channels=args.tcn["in_channel"],
            out_channels=args.tcn["out_channel"],
            kernel_size=args.tcn["kernel_size"],
            dilation=args.tcn["dilation"],
            padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2),
        )

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        b, n, h = x.shape
        x = x.reshape(b * n, 1, h)
        x = self.tcn(x)
        x = x.reshape(b, n, h)
        return self.gcn2(x, adj)

class PatternLibraryManager:

    def __init__(self, args):
        self.base_dir = None
        if hasattr(args, "path") and args.path:
            self.base_dir = os.path.join(args.path, "pattern_libraries")
            os.makedirs(self.base_dir, exist_ok=True)
        self._cache = {}

    def _key(self, year, pattern_type):
        return f"{int(year)}::{pattern_type}"

    def _file_path(self, year, pattern_type):
        if self.base_dir is None:
            return None
        return os.path.join(self.base_dir, f"{int(year)}_{pattern_type}.pkl")

    def get_library_for_year(self, year, pattern_type="spatiotemporal"):
        key = self._key(year, pattern_type)
        if key in self._cache:
            return self._cache[key]
        file_path = self._file_path(year, pattern_type)
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self._cache[key] = data
            return data
        return None

    def update_library(self, year, library_data, metadata=None, pattern_type="spatiotemporal"):
        key = self._key(year, pattern_type)
        payload = {
            "patterns": library_data.get("patterns", []),
            "values": library_data.get("values", []),
            "metadata": metadata or {},
        }
        self._cache[key] = payload
        file_path = self._file_path(year, pattern_type)
        if file_path:
            with open(file_path, "wb") as f:
                pickle.dump(payload, f)
        return True

class RandomProjection(nn.Module):
    def __init__(self, input_dim, output_dim, seed=42):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        weight = torch.randn(input_dim, output_dim, generator=generator) / max(output_dim, 1) ** 0.5
        self.register_buffer("weight", weight)

    def forward(self, x):
        return x @ self.weight

class STRAP(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        gcn_cfg = getattr(args, "gcn", {})
        if isinstance(gcn_cfg, dict):
            self.feature_dim = gcn_cfg.get("hidden_channel", gcn_cfg.get("out_channel", 64))
        else:
            self.feature_dim = getattr(gcn_cfg, "hidden_channel", getattr(gcn_cfg, "out_channel", 64))
        self.k_neighbors = int(getattr(args, "k_neighbors", 16))
        self.max_patterns = int(getattr(args, "max_patterns", 2048))
        self.fusion_weight = float(getattr(args, "fusion_weight", 0.7))
        self.pattern_manager = PatternLibraryManager(args)
        self.current_year = None
        self.projector = None
        self.patterns = {"spatiotemporal": None}
        self.values = {"spatiotemporal": None}

    def _ensure_projector(self, input_dim, device):
        if self.projector is None or self.projector.weight.shape[0] != input_dim:
            self.projector = RandomProjection(input_dim, self.feature_dim).to(device)

    def _to_tensor(self, array_like, device):
        if isinstance(array_like, torch.Tensor):
            return array_like.to(device=device, dtype=torch.float32)
        return torch.tensor(array_like, device=device, dtype=torch.float32)

    def _normalize(self, x):
        return F.normalize(x, dim=-1, eps=1e-8)

    def _build_library_from_data(self, data):
        x = data.x
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.detach().float()
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        self._ensure_projector(x.shape[-1], x.device)
        feats = self._normalize(self.projector(x))
        if feats.shape[0] > self.max_patterns:
            idx = torch.randperm(feats.shape[0], device=feats.device)[: self.max_patterns]
            feats = feats[idx]
        return feats.cpu(), feats.clone().cpu()

    def switch_to_year(self, year):
        lib = self.pattern_manager.get_library_for_year(year, "spatiotemporal")
        if lib is None:
            return False
        self.current_year = int(year)
        self.patterns["spatiotemporal"] = lib["patterns"]
        self.values["spatiotemporal"] = lib["values"]
        return True

    def extract_patterns(self, data, adj=None, year=None):
        if year is None:
            year = getattr(self.args, "year", None)
        if year is None:
            return False
        patterns, values = self._build_library_from_data(data)
        payload = {"patterns": patterns, "values": values}
        meta = {
            "method": "simplified_strap",
            "num_patterns": int(patterns.shape[0]),
            "feature_dim": int(patterns.shape[1]),
        }
        self.pattern_manager.update_library(year, payload, meta, "spatiotemporal")
        self.current_year = int(year)
        self.patterns["spatiotemporal"] = patterns
        self.values["spatiotemporal"] = values
        return True

    def _retrieve(self, query):
        patterns = self.patterns["spatiotemporal"]
        values = self.values["spatiotemporal"]
        if patterns is None or values is None:
            return query
        patterns = self._to_tensor(patterns, query.device)
        values = self._to_tensor(values, query.device)
        query_n = self._normalize(query)
        patterns_n = self._normalize(patterns)
        sim = query_n @ patterns_n.t()
        k = max(1, min(self.k_neighbors, sim.shape[1]))
        topk_val, topk_idx = torch.topk(sim, k=k, dim=1)
        neighbor_values = values[topk_idx]
        weights = F.softmax(topk_val, dim=1).unsqueeze(-1)
        return (neighbor_values * weights).sum(dim=1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.shape[-1] != self.feature_dim:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.feature_dim).squeeze(1)
        retrieved = self._retrieve(x)
        out = self.fusion_weight * x + (1.0 - self.fusion_weight) * retrieved
        mode = getattr(self.args, "return_pattern_or_value", "value")
        if mode == "pattern":
            return self._normalize(out)
        return out

class STRAP_Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.backbone = STGNN_Backbone(args)
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        self.use_strap = bool(getattr(args, "use_strap", True))
        if self.use_strap:
            self.strap = STRAP(args)
            self.strap_adapter = nn.Linear(args.gcn["out_channel"], self.strap.feature_dim)
            setattr(args, "return_pattern_or_value", "value")
        self.current_year = getattr(args, "year", None)
        self.pattern_initialized = False
        log_fn = args.logger.info if hasattr(args, "logger") else print
        log_fn(f"RAP/STRAP initialized with STGNN backbone and year {self.current_year}")

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_fn = self.args.logger.info if hasattr(self.args, "logger") else print
        log_fn(f"Total Parameters: {total}")
        log_fn(f"Trainable Parameters: {trainable}")
        if self.use_strap:
            strap_params = sum(p.numel() for p in self.strap.parameters() if p.requires_grad)
            log_fn(f"strap Parameters: {strap_params}")

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(state_dict, dict) and "strap.projector.weight" in state_dict:
            has_proj = self.use_strap and getattr(self.strap, "projector", None) is not None
            if not has_proj:
                state_dict = dict(state_dict)
                state_dict.pop("strap.projector.weight", None)
        return super().load_state_dict(state_dict, strict=strict)

    def initialize_patterns(self, data, adj, force=False):
        if not self.use_strap:
            return False
        year = self.current_year if self.current_year is not None else getattr(self.args, "year", None)
        if year is None:
            return False
        has_library = self.strap.switch_to_year(year)
        log_fn = self.args.logger.info if hasattr(self.args, "logger") else print
        if not has_library or force:
            log_fn(f"Creating pattern library for year {year}...")
            success = self.strap.extract_patterns(data, adj, year)
            if success:
                self.pattern_initialized = True
                log_fn(f"Year {year} pattern library initialized")
                return True
            log_fn(f"Error in creating pattern library for year {year}")
            return False
        self.pattern_initialized = True
        log_fn(f"Loaded pattern library for year {year}")
        return True

    def set_year(self, year):
        self.current_year = year
        if self.use_strap:
            self.pattern_initialized = self.strap.switch_to_year(year)
        return self.pattern_initialized

    def _prepare_strap(self, data, adj):
        if self.use_strap and not self.pattern_initialized and self.training:
            self.initialize_patterns(data, adj)
        elif self.use_strap and not self.pattern_initialized:
            year = self.current_year if self.current_year is not None else getattr(self.args, "year", None)
            if year is not None:
                self.pattern_initialized = bool(self.strap.switch_to_year(year))
        return adj

    def _apply_strap(self, feature_mid):
        if not (self.use_strap and self.pattern_initialized):
            return feature_mid
        try:
            if self.current_year is not None:
                self.strap.switch_to_year(self.current_year)
            b, n, c = feature_mid.shape
            adapted = self.strap_adapter(feature_mid.reshape(-1, c))
            self.args.return_pattern_or_value = "value"
            enhanced = self.strap(adapted)
            return enhanced.reshape(b, n, -1)
        except Exception as exc:
            log_fn = self.args.logger.info if hasattr(self.args, "logger") else print
            log_fn(f"STRAP application error: {exc}")
            return feature_mid

    def forward(self, data, adj):
        adj = self._prepare_strap(data, adj)
        n = adj.shape[0]
        x = data.x.reshape((-1, n, self.args.gcn["in_channel"]))
        feature_mid = self.backbone(x, adj)
        enhanced = self._apply_strap(feature_mid)
        feature_out = enhanced.reshape(-1, enhanced.shape[-1])
        if feature_out.shape[-1] != self.args.gcn["out_channel"]:
            feature_out = F.adaptive_avg_pool1d(
                feature_out.unsqueeze(1), self.args.gcn["out_channel"]
            ).squeeze(1)
        x = feature_out + data.x
        x = self.fc(self.activation(x))
        return F.dropout(x, p=self.dropout, training=self.training)
