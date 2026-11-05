from __future__ import annotations

import os.path as osp
import warnings

import numpy as np
import torch

DEFAULT_EXTENT = (73.0, 136.0, 17.5, 54.5)

_COORD_CACHE: dict = {}
_ANCHOR_CACHE: dict = {}

def _station_csv_path(args, year: int) -> str:
    pat = getattr(
        args,
        "station_meta_pattern",
        "data/AIR/retained_stations/retained_stations_{year}_AQI_aqi.csv",
    )
    return osp.abspath(pat.format(year=int(year)))

def load_node_coords(args, year: int, num_nodes: int):

    key = (_station_csv_path(args, year), int(num_nodes))
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]

    path = _station_csv_path(args, year)
    if not osp.isfile(path):
        warnings.warn(f"[geo_anchor] station meta not found: {path}; using learned anchors.")
        _COORD_CACHE[key] = None
        return None

    lon, lat = [], []
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        try:
            i_lon, i_lat = header.index("lon"), header.index("lat")
        except ValueError:
            warnings.warn(f"[geo_anchor] no lon/lat columns in {path}; using learned anchors.")
            _COORD_CACHE[key] = None
            return None
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(i_lon, i_lat):
                continue
            lon.append(float(parts[i_lon]))
            lat.append(float(parts[i_lat]))

    lon_a = np.asarray(lon, dtype=np.float32)
    lat_a = np.asarray(lat, dtype=np.float32)
    if lon_a.shape[0] != num_nodes:
        warnings.warn(
            f"[geo_anchor] station rows {lon_a.shape[0]} != graph nodes {num_nodes} "
            f"({path}); using learned anchors."
        )
        _COORD_CACHE[key] = None
        return None

    lo0, lo1, la0, la1 = getattr(args, "caqra_geo_extent", None) or DEFAULT_EXTENT
    gx = (lon_a - lo0) / max(1e-6, (lo1 - lo0))
    gy = (la1 - lat_a) / max(1e-6, (la1 - la0))
    coords = np.stack([np.clip(gx, 0.0, 1.0), np.clip(gy, 0.0, 1.0)], axis=1)
    _COORD_CACHE[key] = coords
    return coords

def token_grid_coords(h_feat: int, w_feat: int) -> np.ndarray:

    ys = (np.arange(h_feat, dtype=np.float32) + 0.5) / float(h_feat)
    xs = (np.arange(w_feat, dtype=np.float32) + 0.5) / float(w_feat)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

def build_topk_anchor(
    args,
    year: int,
    num_nodes: int,
    h_feat: int,
    w_feat: int,
    topk: int,
    device=None,
):

    ck = (_station_csv_path(args, year), num_nodes, h_feat, w_feat, int(topk))
    if ck not in _ANCHOR_CACHE:
        coords = load_node_coords(args, year, num_nodes)
        p = h_feat * w_feat
        k = int(min(max(1, topk), p))
        if coords is None:
            _ANCHOR_CACHE[ck] = (None, None, None)
        else:
            tok = token_grid_coords(h_feat, w_feat)

            d2 = ((coords[:, None, :] - tok[None, :, :]) ** 2).sum(-1)
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            take = np.take_along_axis(d2, idx, axis=1)
            order = np.argsort(take, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            take = np.take_along_axis(take, order, axis=1)
            sig = float(getattr(args, "caqra_anchor_sigma", 0.05))
            w = np.exp(-take / (2.0 * sig * sig))
            w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
            _ANCHOR_CACHE[ck] = (idx.astype(np.int64), w.astype(np.float32), coords)

    idx, w, coords = _ANCHOR_CACHE[ck]
    if idx is None:
        return None, None, None
    ti = torch.from_numpy(idx)
    tw = torch.from_numpy(w)
    tc = torch.from_numpy(coords)
    if device is not None:
        ti, tw, tc = ti.to(device), tw.to(device), tc.to(device)
    return ti, tw, tc
