import sys, argparse, random, torch
sys.path.append("src/")

import numpy as np
import os.path as osp
import networkx as nx
import os
import logging

from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from utils.data_convert import generate_samples
from src.model.model import (
    TrafficStream_Model, STKEC_Model,
    EAC, EAC_DeSCA,
    Universal_Model, STID_Model,
    GraphWaveNet_Model,
    DCRNN, DCRNN_DeSCA,
    PDFormer, PDFormer_DeSCA
)
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from model import detect_default
from src.model import replay

from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs, load_best_model#,  load_test_best_model
from trainer.default_trainer import train, test_model  # , test_model_with_ttc




def evaluate_past_years(model, args):
    args.logger.info("\n========== Catastrophic Forgetting Evaluation ==========")
    for year in range(args.begin_year, args.end_year + 1):
        args.logger.info(f"\n[CF Test] Evaluating Year {year}")
        # ---------- Load graph ----------
        adj = np.load(
            osp.join(args.graph_path, f"{year}_adj.npz")
        )["x"]

        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)

        current_adj = torch.from_numpy(adj).float().to(args.device)

        # update args.adj
        args.adj = current_adj

        # ---------- Load data ----------
        inputs = np.load(
            osp.join(args.save_data_path, f"{year}.npz"),
            allow_pickle=True
        )

        test_loader = DataLoader(
            SpatioTemporalDataset(inputs, "test"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        args.year = year

        test_model(model, args, test_loader, pin_memory=True)

def build_dev_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("dev_logger")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        fmt="%(asctime)s | Year=%(year)s | Epoch=%(epoch)s | Dt=%(Dt).4f | Ds=%(Ds).4f | G_t=%(G_t).4f | G_s=%(G_s).4f | tau_t=%(tau_t).4f | tau_s=%(tau_s).4f | Mode=%(mode)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
def main(args):

    args.logger.info("params : %s", vars(args))
    args.result = {"3":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "6":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "12":{" MAE":{}, "MAPE":{}, "RMSE":{}}, "Avg":{" MAE":{}, "MAPE":{}, "RMSE":{}}}
    mkdirs(args.save_data_path)
    vars(args)["graph_size_list"] = []
    if args.method in ['EAC_DeSCA', 'DCRNN_DeSCA','PDFormer_DeSCA]:
        args.dev_logger = build_dev_logger(osp.join(args.path, "deviation.log"))

    model = None

    for year in range(args.begin_year, args.end_year+1):  # Iterate through each year from the start year to the end year
        
        # Loading graph data
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
        
        # Choose whether to process data or load data directly based on the data_process flag
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=False) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+".npz"), allow_pickle=True)
        
        
        args.logger.info("[*] Year {} load from {}.npz".format(args.year, osp.join(args.save_data_path, str(year))))
        
        # Normalized adjacency matrix
        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)  # Convert the adjacency matrix to a PyTorch tensor and store it in args
        
        if year == args.begin_year and args.load_first_year:  # If it is the first year and you need to skip the first year, the model has been trained and does not need to be retrained
            # model, _ = load_test_best_model(args)
            model, _ = load_best_model(args)
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            # test_model_with_ttc(model, args, test_loader, pin_memory=True)
            continue
        
        vars(args)["node_list"] = list()
        
        if year > args.begin_year and args.strategy == "incremental":  # If it is an incremental strategy and the year is greater than the start year
            
            vars(args)["init_graph_size"] = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
            
            # model, _ = load_test_best_model(args)
            model, _ = load_best_model(args)
            

            node_list = list()
            
            if args.increase:  # Get the newly added node
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))
            
            
            pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"][0:31*288, :]
            cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"][0:31*288, :]
            pre_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]).edges)).T
            cur_graph = np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T
            
            # Sample most 5% of the current graph node size
            topk = int(0.05 * args.graph_size) 
            
            if args.detect or args.replay:
                influence_node_score = detect_default.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
            
            if args.detect:  # Get the affected nodes
                influence_node_list = np.argpartition(np.asarray(influence_node_score), -topk)[-topk:]
                node_list.extend(list(influence_node_list))
            
            if args.replay:  # Get sampling node
                replay_node_list = replay.replay_node_selection(args, influence_node_score, topk)  # Select the replay node
                node_list.extend(list(replay_node_list))
            
            if args.logname == 'trafficStream_random':
                if len(node_list) < int(0.1*args.graph_size):
                    res=int(0.1 * args.graph_size) - len(node_list)
                    expand_node_list = random.sample(range(pre_node_size), res)
                    node_list.extend(list(expand_node_list))
            
            node_list = list(set(node_list))  # Remove duplicate nodes
            
            
            # if len(node_list) > int(0.1 * args.graph_size):  # Limit the number of nodes
            #     node_list = random.sample(node_list, int(0.1 * args.graph_size))
            
            # Get a subgraph of a node list
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)  # Get the index of the edge of the current year
            edge_list = list(nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)  # Get the list of graph edges for the current year
            
            graph_node_from_edge = set()  # Collect all nodes connected by edges
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            
            node_list = list(set(node_list) & graph_node_from_edge)  # Get the list of nodes in the subgraph, that is, the intersection of the nodes to be modified and the existing edge nodes
            
            """
            If the node list is not empty
            Returns the original graph node set, the original graph edge index, and the new index of the node set used for query in the subgraph (central node set) of num_hops hops. 
            Since relabel_nodes is set to True, the nodes will be relabeled from 0, so the original graph edge index is changed to the index of the new graph.
            """
            if len(node_list) != 0:
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph  # Storing subgraphs
                vars(args)["subgraph_edge_index"] = subgraph_edge_index  # Store subgraph edge index
                vars(args)["mapping"] = mapping  # Storage Node Mapping
            args.logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format(len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)  # Storage Node List


        # When there are no nodes that need incremental training, skip this year
        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)  # Load the best model
            mkdirs(osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+"-"+str(args.seed), str(args.year), loss+".pkl"))
            test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            args.logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue
        
        if args.train:  # If training is required
            train(inputs, args)
        else:
            if args.auto_test:
                # model, _ = load_best_model(args)
                model, _ = load_test_best_model(args)
                test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
                test_model(model, args, test_loader, pin_memory=True)

                
                # test_model_with_ttc(model, args, test_loader, pin_memory=True)
    
    
    # Print different step metrics for each year
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info = ""
            info_list = []
            for year in range(args.begin_year, args.end_year+1):
                if i in args.result:
                    if j in args.result[i]:
                        if year in args.result[i][j]:
                            info += "{:>10.2f}\t".format(args.result[i][j][year])
                            info_list.append(args.result[i][j][year])
            args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(info_list)))

    # Print the total training time, average training time per epoch, and number of training rounds
    total_time = 0
    # for year in range(args.begin_year, args.end_year+1):
    #     if year in args.result:
    #         info = "year\t{:<4}\ttotal_time\t{:>10.4f}\taverage_time\t{:>10.4f}\tepoch\t{}".format(year, args.result[year]["total_time"], args.result[year]["average_time"], args.result[year]['epoch_num'])
    #         total_time += args.result[year]["total_time"]
    #         args.logger.info(info)
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            info = (
                "year\t{:<4}\t"
                "total_time\t{:>10.4f}\t"
                "average_time\t{:>10.4f}\t"
                "epoch\t{}\t"
                "params\t{}\t"
                "trainable\t{}\t"
                "train_iter_ms\t{:>10.4f}\t"
                "peak_mem_gb\t{:>10.4f}\t"
                "infer_epoch_s\t{:>10.4f}\t"
                "infer_iter_ms\t{:>10.4f}\t"
                "infer_mem\t{:>10.4f}"
            ).format(
                year,
                args.result[year]["total_time"],
                args.result[year]["average_time"],
                args.result[year]["epoch_num"],
                args.result[year].get("total_params", -1),
                args.result[year].get("trainable_params", -1),
                args.result[year].get("avg_train_iter_time_ms", -1.0),
                args.result[year].get("peak_mem_gb", -1.0),
                args.result[year].get("infer_epoch_time_s", -1.0),
                args.result[year].get("avg_infer_iter_time_ms", -1.0),
                args.result[year].get("peak_infer_mem_gb", -1.0),
            )
            total_time += args.result[year]["total_time"]
            args.logger.info(info)
    args.logger.info("total time: {:.4f}".format(total_time))
    # track per-period training time
    avg_train_times = []
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            avg_time = args.result[year]["average_time"]
            avg_train_times.append(avg_time)
    overall_avg_time = np.mean(avg_train_times)
    args.logger.info(f"[Efficiency] Average Training Time (s/period): {overall_avg_time:.4f}")
    args.avg_training_time = overall_avg_time


    infer_epoch_times = []
    infer_iter_times = []
    infer_mems = []

    for year in range(args.begin_year, args.end_year + 1):
        if year in args.result:
            infer_epoch_times.append(args.result[year]["infer_epoch_time_s"])
            infer_iter_times.append(args.result[year]["avg_infer_iter_time_ms"])
            infer_mems.append(args.result[year]["peak_infer_mem_gb"])

    args.logger.info(f"[Efficiency] Avg Inference Epoch Time (s): {np.mean(infer_epoch_times):.4f}")
    args.logger.info(f"[Efficiency] Avg Inference Iter Time (ms): {np.mean(infer_iter_times):.4f}")
    args.logger.info(f"[Efficiency] Avg Inference Memory (GB): {np.mean(infer_mems):.4f}")
    
    
    params_list = []
    trainable_list = []
    for year in range(args.begin_year, args.end_year+1):
        if year in args.result:
            params_list.append(args.result[year]["total_params"])
            trainable_list.append(args.result[year]["trainable_params"])
    args.logger.info(f"[Efficiency] Avg Params: {np.mean(params_list):.2f}")
    args.logger.info(f"[Efficiency] Avg Trainable Params: {np.mean(trainable_list):.2f}")

    # Catastrophic Forgetting Evaluation
    # args.logger.info("========== Catastrophic Forgetting Evaluation ==========")

    # final_year = args.end_year
    # args.year = final_year+1

    # model, _ = load_best_model(args)
    # model = model.to(args.device)

    # evaluate_past_years(model, args)

    # args.logger.info(
    #     "\n========== Catastrophic Forgetting Summary =========="
    # )

    # for i in ["3", "6", "12", "Avg"]:
    #     for j in [" MAE", "RMSE", "MAPE"]:

    #         info = ""
    #         info_list = []

    #         for year in range(args.begin_year, args.end_year + 1):

    #             if year in args.result[i][j]:
    #                 val = args.result[i][j][year]

    #                 info += "{:>10.2f}\t".format(val)
    #                 info_list.append(val)

    #         args.logger.info(
    #             "{:<4}\t{}\t{}\t{:>8.2f}".format(
    #                 i,
    #                 j,
    #                 info,
    #                 np.mean(info_list)
    #             )
    #         )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/test.json")
    parser.add_argument("--seed", type = int, default = 43)
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--method", type = str, default = "trafficstream")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "log/PEMS04/oneline_st_nn_pems04-18/0/1.8008.pkl", help='specify a pretrained model root')
    
    parser.add_argument('--tau_s', type=float, default=0.1, help='spatial gate tau')
    parser.add_argument('--tau_t', type=float, default=0.1, help='temporal gate tau')
    parser.add_argument('--prototype_num', type=int, default=10, help='number of prototypes')
    parser.add_argument('--rank', type=int, default=6, help='rank')

    args = parser.parse_args()
    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["methods"] = {'trafficstream': TrafficStream_Model, 
    'stkec': STKEC_Model, 
    'EAC': EAC,
    'EAC_DeSCA': EAC_DeSCA, 
    'Universal': Universal_Model,
    'STID': STID_Model,
    'gwnet': GraphWaveNet_Model,
    'DCRNN': DCRNN,
    'DCRNN_DeSCA': DCRNN_DeSCA,
    'PDFormer': PDFormer,
    'PDFormer_DeSCA': PDFormer_DeSCA
    }
     # , 'LSTM': LSTM_Model, 'MLP': MLP_Model, 'STLora': STLora_Model, 
    
    init(args)
    if not hasattr(args, "prototype_num"):
        args.prototype_num = 10
    seed_anything(args.seed)
    init_log(args)
    
    main(args)