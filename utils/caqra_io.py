from __future__ import annotations

import os
import os.path as osp
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch

_YEAR_CACHE: dict[tuple, tuple] = {}
_HEATMAP_FLOAT: dict[tuple, np.ndarray] = {}
_MISSING_WARNED: set = set()

def visual_backend(args) -> str:

    b = str(getattr(args, "visual_backend", "") or "").lower()
    if b in ("heatmap", "pems", "row"):
        return "heatmap"
    if getattr(args, "heatmap_cache_dir", None):
        return "heatmap"
    return "caqra"

def image_mode(args) -> str:

    if visual_backend(args) == "heatmap":
        return "heatmap"
    mode = getattr(args, "caqra_image_mode", None)
    if mode:
        m = str(mode).lower().replace("-", "")
        if m in ("5min", "5m", "slot5", "native5", "heatmap"):
            return "5min" if m != "heatmap" else "heatmap"
        return m
    root = _root(args)
    for y in os.listdir(root) if osp.isdir(root) else []:
        ydir = osp.join(root, y)
        if not osp.isdir(ydir):
            continue
        for e in os.listdir(ydir):
            return "hourly" if osp.isdir(osp.join(ydir, e)) else "daily"
    return "daily"

def _root(args) -> str:
    return osp.abspath(getattr(args, "caqra_image_root", "data/caqra_aqi_images"))

def _geom(args) -> tuple:
    return int(getattr(args, "caqra_image_size", 128)), int(getattr(args, "caqra_image_channels", 3))

def _slots_in_year(mode: str, year: int) -> int:
    days = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
    if mode == "5min":
        return days * 288
    return days * 24 if mode == "hourly" else days

def _slot_of(mode: str, dt: datetime) -> int:
    start = datetime(dt.year, 1, 1)
    delta = dt - start
    if mode == "5min":
        return int(delta.total_seconds() // 300)
    return int(delta.total_seconds() // 3600) if mode == "hourly" else int(delta.days)

def _slot_to_datetime(mode: str, year: int, slot: int) -> datetime:
    base = datetime(year, 1, 1)
    if mode == "5min":
        return base + timedelta(minutes=5 * int(slot))
    return base + (timedelta(hours=slot) if mode == "hourly" else timedelta(days=slot))

def _resolve_path(root: str, mode: str, dt: datetime) -> Optional[str]:
    y, m, d, h = dt.year, dt.month, dt.day, dt.hour
    ymd = f"{y:04d}{m:02d}{d:02d}"
    if mode == "5min":
        hhmm = f"{h:02d}{(dt.minute // 5) * 5:02d}"
        cands = [osp.join(root, str(y), ymd, f"aqi_{ymd}_{hhmm}.png"),
                 osp.join(root, str(y), ymd, f"aqi_{ymd}_{h:02d}{dt.minute:02d}.png")]
    elif mode == "hourly":
        cands = [osp.join(root, str(y), ymd, f"aqi_{ymd}_{h:02d}.png"),
                 osp.join(root, str(y), ymd, f"caqra_aqi_{ymd}_{h:02d}.png")]
    else:
        cands = [osp.join(root, str(y), f"aqi_{ymd}.png"),
                 osp.join(root, str(y), f"caqra_aqi_{ymd}.png"),
                 osp.join(root, str(y), f"{ymd}.png")]
    for p in cands:
        if osp.isfile(p):
            return p
    return None

def timeline_to_datetime(args, year: int, t_index: int) -> datetime:

    res = str(getattr(args, "temporal_resolution", "hourly") or "hourly").lower().replace("-", "")
    if res == "daily":
        return datetime(int(year), 1, 1) + timedelta(days=int(t_index))
    mps = getattr(args, "minutes_per_step", None)
    if mps is not None and float(mps) > 0:
        return datetime(int(year), 1, 1) + timedelta(minutes=float(mps) * int(t_index))
    if res in ("5min", "5m"):
        return datetime(int(year), 1, 1) + timedelta(minutes=5 * int(t_index))
    hps = float(getattr(args, "hours_per_step", 1) or 1)
    return datetime(int(year), 1, 1) + timedelta(hours=float(t_index) * hps)

timeline_to_date = timeline_to_datetime

def _decode(path: str, size: int, ch: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB" if ch == 3 else "L").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] < ch:
        arr = np.repeat(arr, ch, axis=2)
    return np.ascontiguousarray(arr[:, :, :ch].transpose(2, 0, 1))

def _cache_paths(args, year: int, mode: str) -> tuple:
    size, ch = _geom(args)
    d = osp.abspath(getattr(args, "caqra_cache_dir", "data/AIR/caqra_cache"))
    tag = f"{osp.basename(_root(args))}_{mode}_{year}_{size}_{ch}"
    return osp.join(d, tag + ".npy"), osp.join(d, tag + "_mask.npy"), d

def build_year_cache(args, year: int, logger=None) -> tuple:

    mode, root = image_mode(args), _root(args)
    size, ch = _geom(args)
    data_p, mask_p, cache_d = _cache_paths(args, year, mode)
    if osp.isfile(data_p) and osp.isfile(mask_p):
        return np.load(data_p, mmap_mode="r"), np.load(mask_p)

    os.makedirs(cache_d, exist_ok=True)
    n = _slots_in_year(mode, year)
    arr = np.zeros((n, ch, size, size), dtype=np.uint8)
    mask = np.zeros(n, dtype=bool)
    jobs = {}
    for s in range(n):
        p = _resolve_path(root, mode, _slot_to_datetime(mode, year, s))
        if p is not None:
            jobs[s] = p

    workers = int(getattr(args, "caqra_decode_workers", 16))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for s, out in zip(jobs.keys(), ex.map(lambda p: _decode(p, size, ch), jobs.values())):
            arr[s] = out
            mask[s] = True

    np.save(data_p, arr)
    np.save(mask_p, mask)
    msg = (f"[caqra] built {mode} cache {year}: {int(mask.sum())}/{n} frames "
           f"({100.0 * mask.mean():.1f}%) -> {data_p}")
    (logger.info if logger else print)(msg)
    del arr
    return np.load(data_p, mmap_mode="r"), np.load(mask_p)

def _heatmap_path(args, year: int) -> str:
    size, ch = _geom(args)
    d = osp.abspath(getattr(args, "heatmap_cache_dir", "data/PEMS/heatmap_cache"))
    return osp.join(d, f"{int(year)}_{size}_{ch}.npy")

def _heatmap_arrays(args, year: int):
    path = _heatmap_path(args, year)
    key = ("heatmap", path)
    if key not in _YEAR_CACHE:
        arr = np.load(path, mmap_mode="r")
        mask = np.ones(arr.shape[0], dtype=bool)
        _YEAR_CACHE[key] = (arr, mask)
    return _YEAR_CACHE[key]

def _heatmap_float(args, year: int) -> np.ndarray:

    path = _heatmap_path(args, year)
    key = ("heatmap_f32", path)
    if key not in _HEATMAP_FLOAT:
        arr, _ = _heatmap_arrays(args, year)
        flipped = arr[:, :, ::-1, :]
        out = np.empty(arr.shape, dtype=np.float32)
        for i in range(0, arr.shape[0], 256):
            chunk = np.ascontiguousarray(flipped[i:i + 256])
            out[i:i + 256] = torch.from_numpy(chunk).float().div_(255.0).numpy()
        _HEATMAP_FLOAT[key] = out
    return _HEATMAP_FLOAT[key]

def _year_arrays(args, year: int):
    if visual_backend(args) == "heatmap":
        return _heatmap_arrays(args, year)
    mode = image_mode(args)
    size, ch = _geom(args)
    key = (_root(args), mode, int(year), size, ch)
    if key not in _YEAR_CACHE:
        _YEAR_CACHE[key] = build_year_cache(args, int(year), getattr(args, "logger", None))
    return _YEAR_CACHE[key]

def _frame_row(args, year: int, slot: int) -> torch.Tensor:

    size, ch = _geom(args)
    try:
        arr, mask = _year_arrays(args, int(year))
    except Exception:
        return torch.zeros(ch, size, size)
    t = int(arr.shape[0])
    s = int(np.clip(int(slot), 0, max(0, t - 1)))
    if not mask[s]:
        return torch.zeros(ch, size, size)
    if visual_backend(args) == "heatmap" and arr.ndim == 4:
        cache = _heatmap_float(args, int(year))
        return torch.from_numpy(cache[s].copy())
    frame = np.array(arr[s], dtype=np.uint8)
    return torch.from_numpy(frame).float().div_(255.0)

def _context_slots(args, t_end: int) -> list:
    stride = max(1, int(getattr(args, "caqra_frame_stride", 3)))
    k = max(1, int(getattr(args, "caqra_context_frames", 2)))
    return [int(t_end) - i * stride for i in range(k)]

def _frame(args, dt: datetime) -> torch.Tensor:

    mode = image_mode(args)
    size, ch = _geom(args)
    try:
        arr, mask = _year_arrays(args, dt.year)
    except Exception:
        return torch.zeros(ch, size, size)

    n = arr.shape[0]
    s = _slot_of(mode, dt)
    if not (0 <= s < n):
        return torch.zeros(ch, size, size)
    if not mask[s]:
        span = int(getattr(args, "caqra_missing_search", 6))
        found = -1
        for off in range(1, span + 1):
            for cand in (s - off, s + off):
                if 0 <= cand < n and mask[cand]:
                    found = cand
                    break
            if found >= 0:
                break
        if found < 0:
            wk = (dt.year, mode)
            if wk not in _MISSING_WARNED:
                _MISSING_WARNED.add(wk)
                warnings.warn(f"[caqra] {dt:%Y-%m-%d %H:00} unavailable and no frame within "
                              f"{span} steps; using zeros (year {dt.year}).")
            return torch.zeros(ch, size, size)
        s = found
    return torch.from_numpy(np.array(arr[s], dtype=np.uint8)).float().div_(255.0)

def _frame_step(args, mode: str) -> timedelta:
    stride = max(1, int(getattr(args, "caqra_frame_stride", 1)))
    if mode in ("5min", "heatmap"):
        mps = float(getattr(args, "minutes_per_step", 5) or 5)
        return timedelta(minutes=mps * stride)
    if mode == "hourly":
        return timedelta(hours=stride)
    return timedelta(days=1)

def _lag_delta(args, mode: str) -> timedelta:
    if mode == "5min":
        return timedelta(minutes=5 * int(getattr(args, "caqra_image_lag_hours", 0) or 0))
    if mode == "hourly":
        return timedelta(hours=int(getattr(args, "caqra_image_lag_hours", 0) or 0))
    return timedelta(days=int(getattr(args, "caqra_image_lag_days", 1) or 0))

def _step_hours(args, mode: str) -> int:
    if mode == "hourly":
        return max(1, int(getattr(args, "caqra_frame_stride", 1)))
    return 24

def _lag_hours(args, mode: str) -> int:
    if mode == "hourly":
        return int(getattr(args, "caqra_image_lag_hours", 0) or 0)
    return 24 * int(getattr(args, "caqra_image_lag_days", 1) or 0)

def _num_context(args, mode: str) -> int:
    key = "caqra_context_frames" if mode in ("hourly", "5min") else "caqra_context_days"
    return max(1, int(getattr(args, key, 2) or 1))

def context_datetimes(args, year: int, t_end: int) -> list:
    mode = image_mode(args)
    base = timeline_to_datetime(args, year, t_end) - _lag_delta(args, mode)
    step = _frame_step(args, mode)
    return [base - i * step for i in range(_num_context(args, mode))]

def load_caqra_stack(args, year: int, t_end: int, device=None) -> torch.Tensor:

    if visual_backend(args) == "heatmap":
        ten = torch.cat([_frame_row(args, year, s) for s in _context_slots(args, t_end)], dim=0)
    else:
        ten = torch.cat([_frame(args, dt) for dt in context_datetimes(args, year, t_end)], dim=0)
    return ten.to(device, non_blocking=True) if device is not None else ten

def load_caqra_tensor(args, year: int, t_end: int, device=None) -> torch.Tensor:

    if visual_backend(args) == "heatmap":
        ten = _frame_row(args, year, int(t_end))
    else:
        mode = image_mode(args)
        dt = timeline_to_datetime(args, year, t_end) - _lag_delta(args, mode)
        ten = _frame(args, dt)
    return ten.to(device, non_blocking=True) if device is not None else ten

def warm_caqra_cache(args, year: int, t_end_array, future_days: int = 0) -> None:

    if not bool(getattr(args, "caqra_cache_warm", True)):
        return
    if visual_backend(args) == "heatmap":
        try:
            _heatmap_float(args, int(year))
        except Exception as exc:
            warnings.warn(f"[heatmap] no cache for {year}: {exc}")
        return
    years = {int(year)}
    te = np.asarray(t_end_array, dtype=np.int64)
    if te.size:
        for t in (int(te.min()), int(te.max())):
            for dt in context_datetimes(args, int(year), t):
                years.add(dt.year)
    for y in sorted(years):
        try:
            _year_arrays(args, y)
        except Exception as exc:
            warnings.warn(f"[caqra] no image cache for {y}: {exc}")

def coverage(args, year: int) -> tuple:

    _, mask = _year_arrays(args, int(year))
    return int(mask.sum()), int(mask.size)

def frame_statistics(args, year: int) -> tuple:

    if visual_backend(args) == "heatmap":
        data_p = _heatmap_path(args, int(year))
        stat_p = data_p.replace(".npy", "_stats.npz")
    else:
        mode = image_mode(args)
        data_p, _, _ = _cache_paths(args, int(year), mode)
        stat_p = data_p.replace(".npy", "_stats.npz")
    if osp.isfile(stat_p):
        z = np.load(stat_p)
        return z["mean"], float(z["std"])

    arr, mask = _year_arrays(args, int(year))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros(arr.shape[1:], dtype=np.float32), 1.0
    acc = np.zeros(arr.shape[1:], dtype=np.float64)
    for i in range(0, idx.size, 512):
        acc += arr[idx[i:i + 512]].astype(np.float64).sum(axis=0)
    mean = (acc / idx.size / 255.0).astype(np.float32)
    var = 0.0
    for i in range(0, idx.size, 512):
        chunk = arr[idx[i:i + 512]].astype(np.float32) / 255.0
        var += float(((chunk - mean) ** 2).sum())
    std = float(np.sqrt(var / (idx.size * mean.size)))
    std = max(std, 1e-3)
    np.savez(stat_p, mean=mean, std=np.float32(std))
    return mean, std
