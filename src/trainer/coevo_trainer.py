from __future__ import annotations

import os
import os.path as osp
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as func
from torch import optim

from trainer.default_trainer import _air_dataset, build_loader, move_batch_to_device, test_model
from utils.common_tools import load_best_model, mkdirs
from utils.metric import MAE_torch, masked_mae_np

BACKBONE_PREFIXES = {
    "CoEvo_ST": ("U", "V", "gcn1", "gcn2", "tcn1", "fc."),
    "CoEvo_STBP": ("fconv1", "fconv2", "stmodule", "fc.", "pattern_bank"),
    "CoEvo_STRAP": ("backbone.", "fc.", "strap_adapter"),
    "CoEvo_Storm": ("net.", "fc."),
}

def _loss_fn(name: str):
    key = str(name or "huber").lower()
    if key == "mse":
        return lambda y, pred: func.mse_loss(pred, y)
    if key == "mae":
        return lambda y, pred: MAE_torch(pred, y, 0.0)
    return lambda y, pred: func.smooth_l1_loss(pred, y)

def _summary(a: dict) -> dict:
    return {
        "c": a["c"].mean().item(),
        "r_n": a["r_n"].norm(dim=-1).mean().item(),
        "r_v": a["r_v"].norm(dim=-1).mean().item(),
        "alpha": a["alpha"].mean().item(),
        "alpha_sd": a["alpha"].std().item(),
        "delta": a["delta"].abs().mean().item(),
    }

@torch.no_grad()
def dump_drift_trace(model, args, loader, adj, path):
    model.eval()
    rec: dict = {}
    for data in loader:
        model(move_batch_to_device(data, args.device, non_blocking=True), adj)
        for k, v in _summary(model._aux).items():
            rec.setdefault(k, []).append(v)
    np.savez(path, **{k: np.asarray(v) for k, v in rec.items()})
    args.logger.info(f"[EvoCo] drift trace -> {path}")

def _optimizer(model, args):
    g_lr = float(getattr(args, "coevo_graph_lr", 0.0) or 0.0)
    pfx = BACKBONE_PREFIXES.get(str(args.method), BACKBONE_PREFIXES["CoEvo_ST"])
    g_params, o_params = [], []
    for name, prm in model.named_parameters():
        if prm.requires_grad:
            (g_params if name.startswith(pfx) else o_params).append(prm)
    wd = float(getattr(args, "weight_decay", 0.01) or 0.0)
    if g_lr > 0 and g_params:
        groups = [{"params": g_params, "lr": g_lr}] + ([{"params": o_params, "lr": args.lr}] if o_params else [])
        args.logger.info(f"[EvoCo] backbone lr={g_lr:g} ({len(g_params)} tensors), plugin lr={args.lr:g} ({len(o_params)} tensors)")
        return optim.AdamW(groups, weight_decay=wd)
    return optim.AdamW(g_params + o_params, lr=args.lr, weight_decay=wd)

@torch.no_grad()
def _validate(model, args, loader):
    model.eval()
    fused, base, n = 0.0, 0.0, 0
    for data in loader:
        data = move_batch_to_device(data, args.device, non_blocking=True)
        pred = model(data, args.sub_adj)
        yt = data.y.cpu().numpy()
        fused += float(masked_mae_np(yt, pred.cpu().numpy(), 0))
        base += float(masked_mae_np(yt, model._aux["y_n"].reshape_as(pred).cpu().numpy(), 0))
        n += 1
    model.train()
    return fused / max(1, n), base / max(1, n)

def train(inputs, args):
    path = osp.join(args.path, str(args.year))
    mkdirs(path)
    loss_fn = _loss_fn(args.loss)
    n_loss_fn = _loss_fn(getattr(args, "coevo_n_loss", args.loss))
    lam_n = float(getattr(args, "coevo_lambda_n", 1.0))
    lam_map = float(getattr(args, "coevo_lambda_map", 0.1))
    lam_var = float(getattr(args, "coevo_lambda_var", 0.0))
    select = str(getattr(args, "coevo_val_select", "fused") or "fused").lower()
    win = int(getattr(args, "coevo_val_window", 0) or 0)
    warmup = int(getattr(args, "coevo_ckpt_warmup", 0) or 0)
    min_ep = int(getattr(args, "coevo_min_epochs", 0) or 0)
    patience = int(getattr(args, "early_stop_patience", 5))

    train_loader = build_loader(_air_dataset(inputs, "train", args), args, shuffle=True)
    val_loader = build_loader(_air_dataset(inputs, "val", args), args)
    test_loader = build_loader(_air_dataset(inputs, "test", args), args)
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")
    vars(args)["sub_adj"] = vars(args)["adj"]

    if args.init and args.year > args.begin_year:
        model = load_best_model(args)[0]
        model.expand_adaptive_params(args.graph_size)
        online, base = model.online_params()
        for _, prm in base:
            prm.requires_grad = False
        args.logger.info(f"[EvoCo] streaming year: {len(online)} online tensors, {len(base)} frozen")
    else:
        model = args.methods[args.method](args).to(args.device)
        model.expand_adaptive_params(args.graph_size)
    model = model.to(args.device)
    model.set_year_anchor(args, args.graph_size, args.device)
    model.count_parameters()
    optimizer = _optimizer(model, args)
    args.logger.info(f"[EvoCo] token grid {model.n_feat}, top-k {model.anchor_k}, kappa {model.kappa:g}, "
                     f"horizon {model.H}; "
                     f"L = {args.loss}(Y_hat) + {lam_n:g}*{getattr(args, 'coevo_n_loss', args.loss)}(Y_hat^N) "
                     f"+ {lam_map:g}*L_map + {lam_var:g}*L_var; val_select={select}")

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    best, counter, epoch = 1e7, 0, 0
    val_hist, use_time, total_time = [], [], 0.0
    model.train()

    for epoch in range(args.epoch):
        start_time = datetime.now()
        training_loss, terms, cn = 0.0, np.zeros(4), 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = move_batch_to_device(data, args.device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)
            l_pred = loss_fn(data.y, pred)
            l_n = n_loss_fn(data.y, model._aux["y_n"].reshape_as(pred))
            l_map = model.map_loss()
            l_var = model.variance_loss()
            loss = l_pred + lam_n * l_n + lam_map * l_map + lam_var * l_var
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            training_loss += float(loss)
            terms += np.array([float(l_pred), float(l_n), float(l_map), float(l_var)])
            cn += 1

        elapsed = (datetime.now() - start_time).total_seconds()
        total_time += elapsed
        use_time.append(elapsed)
        training_loss /= max(1, cn)
        terms /= max(1, cn)

        fused, base = _validate(model, args, val_loader)
        raw = {"fused": fused, "backbone": base}.get(select, 0.5 * (fused + base))
        val_hist.append(raw)
        validation_loss = float(np.mean(val_hist[-win:])) if win >= 2 else raw

        s = _summary(model._aux)
        args.logger.info(
            f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f} "
            f"(fused {fused:.4f} backbone {base:.4f}) | pred:{terms[0]:.4f} N:{terms[1]:.4f} map:{terms[2]:.4f} "
            f"var:{terms[3]:.4f} | c:{s['c']:.3f} |R^N|:{s['r_n']:.3f} |R^V|:{s['r_v']:.3f} "
            f"alpha:{s['alpha']:.3f}+-{s['alpha_sd']:.3f} |dY|:{s['delta']:.4f}"
        )

        eligible = epoch >= warmup
        if eligible and validation_loss <= best:
            counter = 0
            best = round(validation_loss, 4)
            ckpt = osp.join(path, str(best) + ".pkl")
            torch.save({"model_state_dict": model.state_dict()}, ckpt)
            for fn in os.listdir(path):
                if fn.endswith(".pkl") and osp.join(path, fn) != ckpt:
                    os.remove(osp.join(path, fn))
        elif eligible and epoch + 1 >= min_ep:
            counter += 1
            if counter > patience:
                break

    if best >= 1e6:
        best = round(float(validation_loss), 4)
        torch.save({"model_state_dict": model.state_dict()}, osp.join(path, str(best) + ".pkl"))

    best_model_path = osp.join(path, str(best) + ".pkl")
    model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    model = model.to(args.device)

    test_model(model, args, test_loader, True)
    fused_te, base_te = _validate(model, args, test_loader)
    args.logger.info(f"[EvoCo] test MAE fused {fused_te:.4f} backbone {base_te:.4f} "
                     f"delta {base_te - fused_te:+.4f}")
    dump_drift_trace(model, args, test_loader, args.adj, osp.join(path, "drift_trace.npz"))

    args.result[args.year] = {"total_time": total_time,
                              "average_time": sum(use_time) / max(1, len(use_time)),
                              "epoch_num": epoch + 1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))
