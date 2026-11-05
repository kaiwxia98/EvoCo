import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch

from src.model.ewc import EWC
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data

from dataer.SpatioTemporalDataset import SpatioTemporalDataset, CaqraMultimodalDataset
from utils.metric import cal_metric, masked_mae_np
from utils.common_tools import mkdirs, load_best_model

def move_batch_to_device(data, device, non_blocking: bool = True):

    saved = {}
    for key in ("img", "ctx", "age", "t_end"):
        val = getattr(data, key, None)
        if val is not None and torch.is_tensor(val):
            saved[key] = val.detach()
            delattr(data, key)
    data = data.to(device, non_blocking=non_blocking)
    for key, val in saved.items():
        setattr(data, key, val.to(data.x.device, non_blocking=non_blocking))
    return data

def collate_caqra_batch(batch: list) -> Batch:

    if getattr(batch[0], "img", None) is None:
        return Batch.from_data_list(batch)
    imgs = torch.stack([d.img for d in batch], dim=0)
    clean = [Data(x=d.x, y=d.y) for d in batch]
    b = Batch.from_data_list(clean)
    b.img = imgs
    if getattr(batch[0], "ctx", None) is not None:
        b.ctx = torch.stack([d.ctx for d in batch], dim=0)
    if getattr(batch[0], "age", None) is not None:
        b.age = torch.stack([d.age for d in batch], dim=0).view(len(batch))
    if getattr(batch[0], "t_end", None) is not None:
        b.t_end = torch.stack([d.t_end for d in batch], dim=0).view(len(batch))
    return b

def _air_dataset(inputs, split, args):
    mm = bool(getattr(args, "multimodal_caqra", False)) and f"{split}_t_end" in inputs
    inc = args.strategy == "incremental" and args.year > args.begin_year
    if mm and not inc:
        return CaqraMultimodalDataset(inputs, split, args, args.year)
    return SpatioTemporalDataset(inputs, split)

def _caqra_worker_init(_worker_id):
    torch.set_num_threads(1)

def _loader_kw(args):

    if bool(getattr(args, "multimodal_caqra", False)):
        nw = int(getattr(args, "caqra_num_workers", 4))
        nw = max(0, nw)
        kw = {"pin_memory": True, "num_workers": nw, "collate_fn": collate_caqra_batch}
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = min(4, max(2, nw))
            kw["worker_init_fn"] = _caqra_worker_init
        return kw
    return {"pin_memory": True, "num_workers": int(getattr(args, "dataloader_workers", 32))}

def build_loader(dataset, args, batch_size=None, shuffle=False):

    kw = _loader_kw(args)
    bs = int(batch_size or args.batch_size)
    if "collate_fn" in kw:
        return TorchDataLoader(dataset, batch_size=bs, shuffle=shuffle, **kw)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, **kw)

def train(inputs, args):
    path = osp.join(args.path, str(args.year))
    mkdirs(path)

    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss

    if args.strategy == 'incremental' and args.year > args.begin_year:

        train_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()],            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, **_loader_kw(args))
        val_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()],            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, **_loader_kw(args))

        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:

        train_loader = build_loader(_air_dataset(inputs, "train", args), args, shuffle=True)
        val_loader = build_loader(_air_dataset(inputs, "val", args), args)
        vars(args)["sub_adj"] = vars(args)["adj"]

    test_loader = build_loader(_air_dataset(inputs, "test", args), args)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args)
        if args.ewc:
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)
            ewc_loader = build_loader(_air_dataset(inputs, "train", args), args)
            model.register_ewc_params(ewc_loader, lossfunc, args.device)
        else:
            model = gnn_model

        if args.method in ("EAC",):
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False

        if args.method in ("EAC",):
            model.expand_adaptive_params(args.graph_size)

        if args.method == 'Universal' and args.use_eac == True:
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False

        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)

    else:
        gnn_model = args.methods[args.method](args).to(args.device)
        model = gnn_model
        if args.method in ("EAC",):
            model.expand_adaptive_params(args.graph_size)

        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)

    if args.logname != 'trafficstream':
        model.count_parameters()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = int(getattr(args, "early_stop_patience", 5))
    model.train()
    use_time = []

    for epoch in range(args.epoch):

        start_time = datetime.now()

        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = move_batch_to_device(data, args.device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)

            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]

            loss = lossfunc(data.y, pred, reduction="mean")

            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()

            training_loss += float(loss)
            cn += 1

            loss.backward()
            optimizer.step()

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn

        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = move_batch_to_device(data, args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]

                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            if args.ewc:
                torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
            else:
                torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")

    if args.logname == 'trafficstream':
        best_model = args.methods[args.method](args)
    else:
        best_model = model

    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)

    test_model(best_model, args, test_loader, True)
    args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))

def test_model(model, args, testset, pin_memory):

    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = move_batch_to_device(data, args.device, non_blocking=pin_memory)
            pred = model(data, args.adj)

            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)

            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args)

def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.nan_to_num(mask)

    loss = torch.abs(prediction - target)
    loss = loss * mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)
