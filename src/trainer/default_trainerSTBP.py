import torch
import numpy as np
import os.path as osp
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np, MAE_torch
from utils.common_toolsSTBP import mkdirs, load_best_model
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
    path = osp.join(args.path, str(args.year))
    mkdirs(path)

    train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    vars(args)["sub_adj"] = vars(args)["adj"] 
    
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    if args.init == True and args.year > args.begin_year:
        CSTL_model, _  = load_best_model(args) 
        model = CSTL_model
        
        if args.method == 'STBP':
            args.logger.info("[CL] Before Freeze backbone parameters, trainable parameters:")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    args.logger.info(f"    [Trainable] {n}")
            backbone_prefix = ("fconv1", "stmodule", "fconv2")
            for name, param in model.named_parameters():
                if name.startswith(backbone_prefix):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            args.logger.info("[CL] After Freeze backbone parameters, trainable parameters:")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    args.logger.info(f"    [Trainable] {n}")
            model.expand_adaptive_params(args.graph_size)
        
    else:
        CSTL_model = args.methods[args.method](args).to(args.device)  
        model = CSTL_model
        if args.method == 'STBP':
            model.expand_adaptive_params(args.graph_size)
    
    model.count_parameters()
    total_params, trainable_params = get_model_param_stats(model)
    args.logger.info(f"[Efficiency] Total Params: {total_params}")
    args.logger.info(f"[Efficiency] Trainable Params: {trainable_params}")
    
    for name, param in model.named_parameters():
        print(f"Parameter: {name} | Shape: {param.shape} | Requires Grad: {param.requires_grad}")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 30
    model.train()
    use_time = []

    
    
    for epoch in range(args.epoch):
        epoch_iter_times = []
        epoch_peak_mems = []
        
        start_time = datetime.now()
        
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
            
            pred_loss = MAE_torch(pred, data.y, 0.0)
            aux = model.aux_loss if hasattr(model, "aux_loss") else 0.0
            loss = pred_loss + aux
            
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
        
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)

        avg_iter_time_ms = float(np.mean(epoch_iter_times)) if len(epoch_iter_times) > 0 else 0.0
        max_peak_mem_gb = float(np.max(epoch_peak_mems)) if len(epoch_peak_mems) > 0 else 0.0

        args.logger.info(
            f"[Efficiency][Epoch {epoch}] avg_train_iter_time_ms:{avg_iter_time_ms:.4f}, "
            f"peak_mem_gb:{max_peak_mem_gb:.4f}"
        )

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
        
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    best_model = model
    
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    #test_model(best_model, args, test_loader, True)
    test_stats = test_model(best_model, args, test_loader, True)


    #args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.result[args.year] = {
        "total_time": total_time,
        "average_time": sum(use_time)/len(use_time),
        "epoch_num": epoch + 1,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_train_iter_time_ms": float(np.mean(epoch_iter_times)) if len(epoch_iter_times) > 0 else 0.0,
        "peak_mem_gb": float(np.max(epoch_peak_mems)) if len(epoch_peak_mems) > 0 else 0.0,
        "avg_infer_time_ms": test_stats["avg_infer_time_ms"],
        "peak_infer_mem_gb": test_stats["peak_infer_mem_gb"]
    }
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    infer_times = []
    infer_peak_mems = []
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)

            reset_cuda_stats(args.device)
            infer_start = time.perf_counter()

            pred = model(data, args.adj)

            if torch.cuda.is_available() and "cuda" in str(args.device):
                torch.cuda.synchronize(args.device)
            infer_end = time.perf_counter()

            infer_times.append((infer_end - infer_start) * 1000.0)
            infer_peak_mems.append(get_peak_gpu_memory_gb(args.device))

            loss += MAE_torch(pred, data.y, 0.0)
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))

        avg_infer_time_ms = float(np.mean(infer_times)) if len(infer_times) > 0 else 0.0
        peak_infer_mem_gb = float(np.max(infer_peak_mems)) if len(infer_peak_mems) > 0 else 0.0

        args.logger.info(
            f"[Efficiency][Test] avg_infer_time_ms:{avg_infer_time_ms:.4f}, "
            f"peak_infer_mem_gb:{peak_infer_mem_gb:.4f}"
        )
        
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)


        cal_metric(truth_, pred_, args)
    return {
        "avg_infer_time_ms": avg_infer_time_ms,
        "peak_infer_mem_gb": peak_infer_mem_gb
    }
