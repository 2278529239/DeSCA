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
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np, MAE_torch
from utils.common_tools import mkdirs, load_best_model
import time
import json
def get_model_param_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def reset_cuda_stats(device):
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_gpu_memory_gb(device):
    if torch.cuda.is_available() and "cuda" in str(device):
        return torch.cuda.max_memory_allocated(device) / 1024**3
    return 0.0
def train(inputs, args):
    path = osp.join(args.path, str(args.year))  # Define the current year model save path
    mkdirs(path)
    
    # Setting the loss function
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss
    
    # Dataset definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        # Incremental Policy Data Loader
        train_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        # Construct the adjacency matrix of the subgraph
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)  # Convert to adjacency matrix
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)  # Normalized adjacency matrix
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        # Common Data Loader
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        vars(args)["sub_adj"] = vars(args)["adj"]  # Use the adjacency matrix of the entire graph
    
    # Test Data Loader
    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
    
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model definition
    if args.init == True and args.year > args.begin_year:
        # construct model with previous year's node size from checkpoint
        if args.method in ['EAC', 'EAC_DeSCA', 'DCRNN', 'DCRNN_DeSCA', 'PDFormer', 'PDFormer_DeSCA']:
            prev_num_nodes = np.load(
                osp.join(args.graph_path, str(args.year - 1) + "_adj.npz")
            )["x"].shape[0]
            vars(args)["base_node_size"] = prev_num_nodes
            args.logger.info(f"[*] Reset base_node_size to previous year node size: {prev_num_nodes}")


        gnn_model, _ = load_best_model(args)  # If it is not the first year, load the optimal model
        if args.ewc:  # If you use the ewc strategy, use the ewc model
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))  # Record EWC related parameters
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)  # Initialize the EWC model
            ewc_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            model.register_ewc_params(ewc_loader, lossfunc, args.device)  # Register EWC parameters
        else:
            model = gnn_model  # Otherwise, use the best model loaded
        
        if args.method == 'EAC'or args.method == 'EAC_DeSCA':
            args.logger.info("[CL] Before Freeze backbone parameters, trainable parameters:")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    args.logger.info(f"    [Trainable] {n}")
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False
            args.logger.info("[CL] After Freeze backbone parameters, trainable parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    args.logger.info(f"    [Trainable] {name}")


        if args.method == 'DCRNN'or args.method == 'DCRNN_DeSCA':
            args.logger.info("[CL] Before Freeze backbone parameters, trainable parameters:")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    args.logger.info(f"    [Trainable] {n}")
            for name, param in model.named_parameters():
                if any(k in name for k in [
                    "backbone.diffusion_conv_forward",
                    "backbone.diffusion_conv_backward",
                    "backbone.gru_cell",
                    "backbone.diffusion_conv_out"
                ]):
                    param.requires_grad = False
            args.logger.info("[CL] After Freeze backbone parameters, trainable parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    args.logger.info(f"    [Trainable] {name}")


        if args.method == 'PDFormer' or args.method == 'PDFormer_DeSCA':
            args.logger.info("[CL] Before Freeze backbone parameters, trainable parameters:")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    args.logger.info(f"    [Trainable] {n}")

            for name, param in model.named_parameters():
                if (
                    "backbone.enc_embed_layer" in name or
                    "backbone.encoder_blocks" in name or
                    "backbone.temporal_pool" in name or
                    "backbone.out_proj" in name or
                    "backbone.pattern_keys_param" in name
                ):
                    param.requires_grad = False

            args.logger.info("[CL] After Freeze backbone parameters, trainable parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    args.logger.info(f"    [Trainable] {name}")
            

        
        
        if args.method in ['EAC', 'EAC_DeSCA', 'STID', 'gwnet', 'itransformer','DCRNN','DCRNN_DeSCA', 'PDFormer', 'PDFormer_DeSCA']:
            model.expand_adaptive_params(args.graph_size)
        
        
        
        
        if args.method == 'Universal' and args.use_eac == True:
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False
        
        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
        
    else:
        gnn_model = args.methods[args.method](args).to(args.device)  # If it is the first year, use the base model
        model = gnn_model
        if args.method in ['EAC', 'EAC_DeSCA','DCRNN','DCRNN_DeSCA']:
            model.expand_adaptive_params(args.graph_size)
        
        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
    
    
    model.count_parameters()

    total_params, trainable_params = get_model_param_stats(model)
    args.logger.info(f"[Efficiency] Total Params: {total_params}")
    args.logger.info(f"[Efficiency] Trainable Params: {trainable_params}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []
    
    all_epoch_avg_iter_times = []
    all_epoch_peak_mems = []

    for epoch in range(args.epoch):
        

        epoch_iter_times = []
        epoch_peak_mems = []
        
        start_time = datetime.now()
        
        # Training the model
        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=True)

            reset_cuda_stats(args.device)
            iter_start = time.perf_counter()
    
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)

            if isinstance(pred, tuple):
                pred = pred[0]
            
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)  # to_dense_batch is used to convert a batch of sparse adjacency matrices into a batch of dense adjacency matrices
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]  # Slice according to the mapping to obtain the prediction and true value of the change node
                data.y = data.y[:, args.mapping, :]
            
            pred_loss = MAE_torch(pred, data.y, 0.0)
            #aux_loss = model.get_aux_loss() if args.method == 'EAC' else 0.0
            aux_loss = model.get_aux_loss() if args.method in ['EAC','DCRNNplus','PDFormerplus'] else 0.0
            loss = pred_loss + aux_loss
            
            
            
            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()  # Calculate and add ewc loss if necessary
            
            training_loss += float(loss)
            cn += 1
            
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available() and "cuda" in str(args.device):
                torch.cuda.synchronize(args.device)

            iter_end = time.perf_counter()
            iter_time_ms = (iter_end - iter_start) * 1000.0
            peak_mem_gb = get_peak_gpu_memory_gb(args.device)

            epoch_iter_times.append(iter_time_ms)
            epoch_peak_mems.append(peak_mem_gb)
        
        
        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn 
        
        # Validate the model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.sub_adj)

                if isinstance(pred, tuple):
                    pred = pred[0]


                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                
                loss = masked_mae_np(
                    data.y.cpu().numpy(),
                    pred.cpu().numpy(),
                    0
                )
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        



        avg_iter_time_ms = float(np.mean(epoch_iter_times)) if len(epoch_iter_times) > 0 else 0.0
        max_peak_mem_gb = float(np.max(epoch_peak_mems)) if len(epoch_peak_mems) > 0 else 0.0

        all_epoch_avg_iter_times.append(avg_iter_time_ms)
        all_epoch_peak_mems.append(max_peak_mem_gb)

        args.logger.info(
            f"[Efficiency][Epoch {epoch}] avg_train_iter_time_ms:{avg_iter_time_ms:.4f}, "
            f"peak_mem_gb:{max_peak_mem_gb:.4f}"
        )
        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        # Early Stopping Strategy
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
        
    if args.logname == 'trafficstream' or args.logname == 'stkec':
        best_model = args.methods[args.method](args)
    else:
        best_model = model
    
    ckpt = torch.load(best_model_path, map_location=args.device)
    if hasattr(best_model, "model"):
        best_model.model.load_state_dict(ckpt["model_state_dict"])
    else:
        best_model.load_state_dict(ckpt["model_state_dict"])
    # best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    
    # Test the Model
    #test_model(best_model, args, test_loader, True)
    test_stats = test_model(best_model, args, test_loader, True)

    args.result[args.year] = {
        "total_time": total_time,
        "average_time": sum(use_time) / len(use_time),
        "epoch_num": epoch + 1,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_train_iter_time_ms": float(np.mean(all_epoch_avg_iter_times)) if len(all_epoch_avg_iter_times) > 0 else 0.0,
        "peak_mem_gb": float(np.max(all_epoch_peak_mems)) if len(all_epoch_peak_mems) > 0 else 0.0,
        "infer_epoch_time_s": test_stats["infer_epoch_time_s"],
        "avg_infer_iter_time_ms": test_stats["avg_infer_iter_time_ms"],
        "peak_infer_mem_gb": test_stats["peak_infer_mem_gb"]
    }
    
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    infer_times = []
    
    if torch.cuda.is_available() and "cuda" in str(args.device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)

    epoch_infer_start = time.perf_counter()
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)

            
            infer_start = time.perf_counter()

            pred = model(data, args.adj)

            if isinstance(pred, tuple):
                    pred = pred[0]
            
            if torch.cuda.is_available() and "cuda" in str(args.device):
                torch.cuda.synchronize(args.device)
            infer_end = time.perf_counter()

            infer_times.append((infer_end - infer_start) * 1000.0)
            

            loss += MAE_torch(pred, data.y, 0.0)
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
                        
            pred_.append(pred.cpu().numpy())
            truth_.append(data.y.cpu().numpy())
            cn += 1
        if torch.cuda.is_available() and "cuda" in str(args.device):
            torch.cuda.synchronize(args.device)
        epoch_infer_end = time.perf_counter()
        
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))

        infer_epoch_time_s = epoch_infer_end - epoch_infer_start
        avg_infer_iter_time_ms = float(np.mean(infer_times)) if len(infer_times) > 0 else 0.0
        peak_infer_mem_gb = get_peak_gpu_memory_gb(args.device)
        args.logger.info(
            f"[Efficiency][Test] infer_epoch_time_s:{infer_epoch_time_s:.4f}, "
            f"avg_infer_iter_time_ms:{avg_infer_iter_time_ms:.4f}, "
            f"peak_infer_mem_gb:{peak_infer_mem_gb:.4f}"
        )
        
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)

        cal_metric(truth_, pred_, args)
    
    return {
        "infer_epoch_time_s": infer_epoch_time_s,
        "avg_infer_iter_time_ms": avg_infer_iter_time_ms,
        "peak_infer_mem_gb": peak_infer_mem_gb
    }



def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)


