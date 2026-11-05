import sys, argparse, random, torch
sys.path.append("src/")

import numpy as np
import os.path as osp
import networkx as nx

from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from utils.data_convert import generate_samples
from src.model.model import TrafficStream_Model, STKEC_Model, EAC_Model, Universal_Model
from src.model.coevo_st import CoEvo_ST_Model
from src.model.coevo_stbp import CoEvo_STBP_Model
from src.model.coevo_strap import CoEvo_STRAP_Model
from src.model.coevo_storm import CoEvo_Storm_Model
from src.model.strap import STRAP_Model
from src.model.storm import Storm_Model
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from model import detect_default
from src.model import replay

from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs, load_best_model, long_term_pattern, load_test_best_model
from trainer.default_trainer import train, test_model, build_loader, _air_dataset
from trainer.coevo_trainer import train as coevo_train
from trainer.storm_trainer import train as storm_train

def raw_data_npz_path(args, year: int) -> str:

    suf = getattr(args, "raw_npz_suffix", None) or ""
    return osp.join(args.raw_data_path, f"{year}{suf}.npz")

def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {"3":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "6":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "12":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "Avg":{" MAE":{}, "MAPE":{}, "RMSE":{}}}
    mkdirs(args.save_data_path)
    vars(args)["graph_size_list"] = []

    for year in range(args.begin_year, args.end_year+1):

        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])

        if year == args.begin_year:
            vars(args)["init_graph_size"] = graph.number_of_nodes()
            vars(args)["subgraph"] = graph
            vars(args)["base_node_size"] = graph.number_of_nodes()
        else:
            vars(args)["init_graph_size"] = args.graph_size

        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        args.graph_size_list.append(graph.number_of_nodes())
        resume_from = int(getattr(args, "coevo_resume_from_year", 0) or 0)
        if resume_from and year < resume_from:
            args.logger.info("[*] Year {} skipped (resume from {})".format(year, resume_from))
            continue

        if args.data_process:
            raw_x = np.load(raw_data_npz_path(args, year))["x"]
            mm = bool(getattr(args, "multimodal_caqra", False))
            inputs = generate_samples(
                31,
                osp.join(args.save_data_path, str(year)),
                raw_x,
                graph,
                val_test_mix=False,
                x_len=int(getattr(args, "x_len", 12)),
                y_len=int(getattr(args, "y_len", 12)),
                attach_t_end=mm,
                t_end_align=str(getattr(args, "caqra_t_end_align", "last_x") or "last_x"),
                air_train_fraction=getattr(args, "air_train_fraction", None),
            )
        else:
            inputs = np.load(osp.join(args.save_data_path, str(year) + ".npz"), allow_pickle=True)
        if "station_median_cms" in inputs:
            vars(args)["station_median_cms"] = np.asarray(inputs["station_median_cms"], dtype=np.float64)
            vars(args)["_logged_eval_expm1"] = False

        args.logger.info("[*] Year {} load from {}.npz".format(args.year, osp.join(args.save_data_path, str(year))))

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        if year == args.begin_year and args.load_first_year:
            model, _ = load_test_best_model(args)
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)

            continue

        vars(args)["node_list"] = list()

        if year > args.begin_year and args.strategy == "incremental":

            vars(args)["init_graph_size"] = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]

            model, _ = load_test_best_model(args)

            node_list = list()

            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))

            pre_data = np.load(raw_data_npz_path(args, year - 1))["x"][0:31*288, :]
            cur_data = np.load(raw_data_npz_path(args, year))["x"][0:31*288, :]
            pre_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
            cur_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T

            topk = int(0.05 * args.graph_size)

            if args.detect or args.replay:
                influence_node_score = detect_default.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)

            if args.detect:
                influence_node_list = np.argpartition(np.asarray(influence_node_score), -topk)[-topk:]
                node_list.extend(list(influence_node_list))

            if args.replay:
                replay_node_list = replay.replay_node_selection(args, influence_node_score, topk)
                node_list.extend(list(replay_node_list))

            if args.logname == 'trafficStream_random':
                if len(node_list) < int(0.1*args.graph_size):
                    res=int(0.1 * args.graph_size) - len(node_list)
                    expand_node_list = random.sample(range(pre_node_size), res)
                    node_list.extend(list(expand_node_list))

            node_list = list(set(node_list))

            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)

            graph_node_from_edge = set()
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)

            node_list = list(set(node_list) & graph_node_from_edge)

            if len(node_list) != 0:
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            args.logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format(len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)

        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            mkdirs(osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year), loss+".pkl"))
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            args.logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue

        if args.train:
            if args.method in ("CoEvo_ST", "CoEvo_STBP", "CoEvo_STRAP", "CoEvo_Storm"):
                coevo_train(inputs, args)
            elif args.method == "Storm":
                storm_train(inputs, args)
            else:
                train(inputs, args)
        else:
            if args.auto_test:
                model, _ = load_test_best_model(args)
                if hasattr(model, "set_year_anchor"):
                    model.set_year_anchor(args, args.graph_size, args.device)
                test_loader = build_loader(_air_dataset(inputs, "test", args), args)
                test_model(model, args, test_loader, pin_memory=True)

    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info = ""
            info_list = []
            for year in range(args.begin_year, args.end_year+1):
                if i in args.result:
                    if j in args.result[i]:
                        if year in args.result[i][j]:
                            info += "{:>10.4f}\t".format(args.result[i][j][year])
                            info_list.append(args.result[i][j][year])
            args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.4f}".format(np.mean(info_list)))

    total_time = 0
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            info = "year\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(year, args.result[year]["total_time"], args.result[year]["average_time"], args.result[year]['epoch_num'])
            total_time += args.result[year]["total_time"]
            args.logger.info(info)
    args.logger.info("total time: {:.4f}".format(total_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/test.json")
    parser.add_argument("--seed", type = int, default = 43)
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--method", type = str, default = "TrafficStream")
    parser.add_argument("--load_first_year", type = int, default = 1, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "log/PEMS/eac-43/2011/14.7956.pkl", help='specify a pretrained model root')
    args = parser.parse_args()
    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["methods"] = {
        "TrafficStream": TrafficStream_Model,
        "STKEC": STKEC_Model,
        "EAC": EAC_Model,
        "CoEvo_ST": CoEvo_ST_Model,
        "CoEvo_STBP": CoEvo_STBP_Model,
        "CoEvo_STRAP": CoEvo_STRAP_Model,
        "CoEvo_Storm": CoEvo_Storm_Model,
        "STRAP": STRAP_Model,
        "RAP": STRAP_Model,
        "Storm": Storm_Model,
        "Universal": Universal_Model,
    }

    init(args)
    seed_anything(args.seed)
    init_log(args)

    main(args)
