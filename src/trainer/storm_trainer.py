from __future__ import annotations

import copy
import os.path as osp
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from trainer.default_trainer import (
    _air_dataset,
    build_loader,
    move_batch_to_device,
    test_model,
)
from utils.common_tools import load_best_model, mkdirs
from utils.metric import masked_mae_np

class VarianceAlphaAdjuster:
    def __init__(self, alpha_min=0.4, alpha_max=0.6):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.prev_variance = None
        self.max_variance_change = 0.0
        self.min_variance_change = float("inf")

    def update_alpha(self, current_output):
        current_variance = current_output.var().item()
        if self.prev_variance is None:
            self.prev_variance = current_variance
            return self.alpha_min
        variance_change = abs(current_variance - self.prev_variance)
        self.max_variance_change = max(self.max_variance_change, variance_change)
        self.min_variance_change = min(self.min_variance_change, variance_change)
        self.prev_variance = current_variance
        if self.max_variance_change == self.min_variance_change:
            return self.alpha_min
        normalized = (variance_change - self.min_variance_change) / (
            self.max_variance_change - self.min_variance_change
        )
        return self.alpha_min + normalized * (self.alpha_max - self.alpha_min)

def update_teacher_model(teacher_model, student_model, alpha):
    with torch.no_grad():
        for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
            t_param.data.mul_(1.0 - alpha).add_(s_param.data, alpha=alpha)

def get_aug_x(x):

    seed = int(torch.randint(0, 3, (1,)).item())
    if seed == 0:
        return x + 0.1 * torch.randn_like(x)
    if seed == 1:
        return torch.roll(x, shifts=1, dims=1)
    return torch.flip(x, dims=[1])

@torch.no_grad()
def _linearize(model):
    signs = {}
    for name, param in model.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs

@torch.no_grad()
def _nonlinearize(model, signs):
    for name, param in model.state_dict().items():
        param.mul_(signs[name])

def get_synflow_scores(model, dummy, adj, device):
    model.eval()
    signs = _linearize(model)
    model.zero_grad(set_to_none=True)
    dummy = dummy.to(device)
    out = model(dummy, adj)
    torch.sum(out).backward()
    scores = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        scores[name] = (param.grad * param).detach().abs()
        param.grad.zero_()
    _nonlinearize(model, signs)
    model.train()
    return scores

def _is_fc_name(name: str) -> bool:
    return "fc" in name.split(".") or name.endswith("fc.conv.weight") or name.endswith("fc.conv.bias") or ".fc." in name

def apply_gpf(model, dummy, adj, args, state):

    device = args.device
    scores = get_synflow_scores(model, dummy, adj, device)
    freezeable = [
        (n, s) for n, s in scores.items()
        if n not in state["frozen"] and not _is_fc_name(n)
    ]
    if not freezeable:
        return
    if state["P"] is None:
        state["P"] = len(scores)
        state["P_reserve"] = max(0, state["P"] - state["reserve"])
        args.logger.info(
            f"[Storm GPF] params={state['P']} reserve={state['reserve']} "
            f"freezable={state['P_reserve']} steps={state['N']}"
        )
    state["step"] += 1
    k, n = state["step"], state["N"]
    if k > n or state["P_reserve"] <= 0:
        return
    freeze_k = int(round((n - k + 1) * state["P_reserve"] / max(state["sum_w"], 1)))
    remaining = state["P_reserve"] - state["cum"]
    freeze_k = min(freeze_k, remaining, len(freezeable))
    if freeze_k <= 0:
        return
    freezeable.sort(key=lambda kv: float(kv[1].mean()) if torch.is_tensor(kv[1]) else float(kv[1]))
    named = dict(model.named_parameters())
    n_skip_fc = 0
    frozen_now = 0
    for name, _ in freezeable:
        if frozen_now >= freeze_k:
            break
        if _is_fc_name(name):
            n_skip_fc += 1
            continue
        named[name].requires_grad = False
        state["frozen"].add(name)
        frozen_now += 1
    state["cum"] += frozen_now
    args.logger.info(
        f"[Storm GPF] epoch freeze {frozen_now} tensors "
        f"(cum {state['cum']}/{state['P_reserve']}; skipped fc={n_skip_fc})"
    )

class _XBatch:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

def _to_btn(data, n, t_in):
    x = data.x
    if x.dim() == 2:
        x = x.reshape(-1, n, t_in)
    return x.permute(0, 2, 1).unsqueeze(-1).contiguous()

def _from_btn(btn):

    x = btn.squeeze(-1).permute(0, 2, 1).contiguous()
    return x.reshape(-1, x.shape[-1])

def train(inputs, args):
    path = osp.join(args.path, str(args.year))
    mkdirs(path)

    if args.loss == "mse":
        lossfunc = F.mse_loss
    elif args.loss == "huber":
        lossfunc = F.smooth_l1_loss
    else:
        lossfunc = F.l1_loss

    train_loader = build_loader(_air_dataset(inputs, "train", args), args, shuffle=True)
    val_loader = build_loader(_air_dataset(inputs, "val", args), args)
    test_loader = build_loader(_air_dataset(inputs, "test", args), args)
    vars(args)["sub_adj"] = vars(args)["adj"]
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    if args.init is True and args.year > args.begin_year:
        model, _ = load_best_model(args)
        if hasattr(model, "expand_adaptive_params"):
            model.expand_adaptive_params(args.graph_size)
    else:
        model = args.methods[args.method](args).to(args.device)
        if hasattr(model, "expand_adaptive_params"):
            model.expand_adaptive_params(args.graph_size)

    model.count_parameters()

    use_mkd = args.year > args.begin_year
    teacher = None
    if use_mkd:
        teacher = copy.deepcopy(model).to(args.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        args.logger.info(
            f"[Storm] DKD on: ema_alpha={float(getattr(args, 'storm_alpha', 0.01)):g} "
            f"moe_lambda={float(getattr(args, 'storm_lambda', 0.55)):g} "
            f"gpf={bool(getattr(args, 'storm_gpf', True))}"
        )

    wd = float(getattr(args, "weight_decay", 1e-4) or 0.0)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        eps=1e-8,
        weight_decay=wd,
    )
    max_grad = float(getattr(args, "max_grad_norm", 5.0) or 0.0)
    ema_alpha = float(getattr(args, "storm_alpha", 0.01))
    moe_lambda = float(getattr(args, "storm_lambda", 0.55))
    use_gpf = bool(getattr(args, "storm_gpf", True)) and use_mkd
    alpha_adj = VarianceAlphaAdjuster(alpha_min=0.4, alpha_max=0.6)

    gpf_state = {
        "frozen": set(),
        "P": None,
        "P_reserve": None,
        "cum": 0,
        "step": 0,
        "N": max(1, int(args.epoch) // 10),
        "reserve": 1,
        "sum_w": None,
    }
    gpf_state["sum_w"] = gpf_state["N"] * (gpf_state["N"] + 1) // 2

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter, epoch = 0, 0
    patience = int(getattr(args, "early_stop_patience", 50))
    model.train()
    use_time = []
    total_time = 0.0
    n = args.sub_adj.shape[0]
    t_in = int(getattr(args, "x_len", 12))

    for epoch in range(args.epoch):
        start_time = datetime.now()
        if use_gpf and epoch > 0 and epoch % 10 == 0:
            dummy_data = next(iter(train_loader))
            dummy_data = move_batch_to_device(dummy_data, args.device)
            apply_gpf(model, dummy_data, args.sub_adj, args, gpf_state)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                eps=1e-8,
                weight_decay=wd,
            )

        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = move_batch_to_device(data, args.device, non_blocking=True)
            optimizer.zero_grad()

            if use_mkd:
                btn = _to_btn(data, n, t_in)
                dyn_alpha = alpha_adj.update_alpha(btn)
                if dyn_alpha > moe_lambda:
                    aug = get_aug_x(btn)
                    student_in = _XBatch(_from_btn(aug))
                    teacher_in = _XBatch(_from_btn(aug))
                    pred = model(student_in, args.sub_adj)
                    with torch.no_grad():
                        t_pred = teacher(teacher_in, args.sub_adj)
                    loss = lossfunc(pred, t_pred.detach())
                else:
                    pred = model(data, args.sub_adj)
                    loss = lossfunc(data.y, pred)
            else:
                pred = model(data, args.sub_adj)
                loss = lossfunc(data.y, pred)

            training_loss += float(loss)
            cn += 1
            loss.backward()
            if max_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()
            if teacher is not None:
                update_teacher_model(teacher, model, ema_alpha)

        dt = (datetime.now() - start_time).total_seconds()
        total_time = dt if epoch == 0 else total_time + dt
        use_time.append(dt)
        training_loss = training_loss / max(cn, 1)

        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for data in val_loader:
                data = move_batch_to_device(data, args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss / max(cn, 1))
        args.logger.info(
            f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}"
        )

        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save(
                {"model_state_dict": model.state_dict()},
                osp.join(path, str(round(validation_loss, 4)) + ".pkl"),
            )
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss) + ".pkl")
    best_model = model
    best_model.load_state_dict(torch.load(best_model_path, map_location=args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    test_model(best_model, args, test_loader, True)
    args.result[args.year] = {
        "total_time": total_time,
        "average_time": sum(use_time) / max(len(use_time), 1),
        "epoch_num": epoch + 1,
    }
    args.logger.info(
        "Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path)
    )
