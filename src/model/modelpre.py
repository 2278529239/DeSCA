import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn_conv import BatchGCNConv, ChebGraphConv


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class MLP_Model(nn.Module):
    """Some Information about MLP"""
    def __init__(self, args):
        super(MLP_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).transpose(1, 2).unsqueeze(-1)
        
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden).squeeze(-1).reshape(1, 2)
        x = prediction.reshape(-1, 12)
        return x



class LSTM_Model(nn.Module):
    """Some Information about LSTM"""
    def __init__(self, args):
        super(LSTM_Model, self).__init__()
        self.args = args
        
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=12, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=12, hidden_size=48, num_layers=2, batch_first=True)
        
        self.end_linear1 = nn.Linear(48, 24)
        self.end_linear2 = nn.Linear(24, 12)

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"])).unsqueeze(-1).transpose(1, 2).transpose(1, 3)   # [bs, t, n, f]
        b, f, n, t = x.shape

        x = x.transpose(1,2).reshape(b*n, f, 1, t)  # (b, f, n, t) -> (b, n, f, t) -> (b * n, f, 1, t)
        x = self.start_conv(x).squeeze().transpose(1, 2)  # (b * n, f, 1, t) -> (b * n, init_dim, 1, t) -> (b * n, init_dim, t) -> (b * n, t, init_dim)

        out, _ = self.lstm(x)  # (b * n, t, hidden_dim) -> (b * n, t, hidden_dim)
        x = out[:, -1, :]

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b*n, t)
        return x


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=10):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_a = nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_dim, r)))
        self.lora_b = nn.Parameter(torch.zeros(r, out_dim))
        self.scaling = 1 / (r * in_dim)

    def forward(self, x):
        return x + self.scaling * torch.matmul(torch.matmul(x, self.lora_a.to(x.device)), self.lora_b.to(x.device))
    

class STLora_Model(nn.Module):
    """Some Information about TrafficStream_Model"""
    def __init__(self, args):
        super(STLora_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        self.lora_layers = nn.ModuleList()  # LoRA layers added over time
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    
    def add_lora_layer(self):
        in_dim = self.args.gcn["hidden_channel"]
        out_dim = self.args.gcn["hidden_channel"]
        lora_layer = LoRALayer(in_dim, out_dim)
        self.lora_layers.append(lora_layer)
        self.freeze_lora_layers()  # freeze previous LoRA layers
    
    def freeze_lora_layers(self):
        for lora_layer in self.lora_layers[:-1]:  # all but the latest layer stay frozen
            for param in lora_layer.parameters():
                param.requires_grad = False

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)
        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        x = self.tcn1(x)                                           # [bs * N, 1, feature]
        
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        
        x = x.reshape((-1, self.args.gcn["hidden_channel"]))    # [bs * N, feature]
        
        for lora_layer in self.lora_layers:
            x = lora_layer(x)
        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x


class EAC_Modelpre(nn.Module):
    """Some Information about EAC_Model"""
    def __init__(self, args):
        super(EAC_Modelpre, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  # Set a low rank value
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], 
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        # Initialize subspace and adjust matrix
        self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
        
        self.year = args.year
        self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        B, N, T = x.shape
        
        # Compute adaptive parameters using low-rank matrices
        adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            
            self.num_nodes = new_num_nodes
class DeviationGate(nn.Module):
    """
    Soft-Thresholding Deviation Gate
    g(δ) = δ * max(0, 1 - τ / ||δ||)
    """
    def __init__(self, dim, init_tau=0.10):
        super().__init__()
        # single dead-zone threshold
        self.tau = nn.Parameter(torch.tensor(init_tau))
    def forward(self, delta):
        norm = torch.norm(delta, dim=-1, keepdim=True)
        # ReLU gives zero gate when norm <= tau
        gate = F.relu(norm - self.tau) / (norm + 1e-8)
        g_delta = delta * gate
        return g_delta, gate.mean()  # mean activation for logging
class EAC_Model(nn.Module):
    """Some Information about EAC_Model"""
    def __init__(self, args):
        super(EAC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank  # Set a low rank value
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], 
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))

        self.in_dim = args.gcn["in_channel"]
        self.hidden_dim = args.gcn["hidden_channel"]
        self.backbone_out_dim = args.gcn["out_channel"]
                
        self.year = args.year
        self.num_nodes = args.base_node_size

        # Decoupling: H ≈ U X V^T
        self.U_basis = nn.Parameter(torch.empty(self.num_nodes, self.rank))
        self.V_basis = nn.Parameter(torch.empty(self.backbone_out_dim, self.rank))
        nn.init.orthogonal_(self.U_basis)
        nn.init.orthogonal_(self.V_basis)

        # Prompt on U and V (Additive)
        self.P_s = nn.Parameter(torch.zeros(self.num_nodes, self.rank))
        self.P_t = nn.Parameter(torch.zeros(self.backbone_out_dim, self.rank))
        nn.init.uniform_(self.P_s, -0.05, 0.05)
        nn.init.uniform_(self.P_t, -0.05, 0.05)

        # prediction head
        self.fc = nn.Linear(self.backbone_out_dim * 2, self.backbone_out_dim)
        self.output = nn.Linear(self.backbone_out_dim, args.y_len)
        self.activation = nn.GELU()

        # prototype bank
        self.prototypes = nn.Parameter(torch.randn(args.prototype_num, self.backbone_out_dim))
        self.Wq = nn.Linear(self.backbone_out_dim, self.backbone_out_dim)

        # running anchors: spatial [N, D], temporal [D]
        self.register_buffer("h_anchor_s", torch.zeros(self.num_nodes, self.backbone_out_dim)) # [N, D]
        self.register_buffer("h_anchor_t", torch.zeros(self.backbone_out_dim))                 # [D]
        
        # g(delta) 
        self.gate_s = DeviationGate(self.backbone_out_dim, init_tau=args.tau_s)
        self.gate_t = DeviationGate(self.backbone_out_dim, init_tau=args.tau_t)

    
    def get_aux_loss(self):
        return self.aux_loss if hasattr(self, "aux_loss") else 0.0

    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def query_prototypes(self, h):
        """
        h: [B, N, D]
        Returns:
            v: retrieved prototype mix [B, N, D]
            query: projected query [B, N, D]
            pos: nearest prototype [B, N, D]
            neg: second-nearest prototype [B, N, D]
        """
        B, N, D = h.shape
        query = self.Wq(h) # [B, N, D]
        
        # compute similarity between query and prototypes

        attn_score = torch.matmul(query, self.prototypes.t()) 
        attn_prob = torch.softmax(attn_score, dim=-1)
        
        # weighted prototype mix
        v = torch.matmul(attn_prob, self.prototypes) # [B, N, D]
        
        # top-2 prototypes for margin loss
        _, top2_idx = torch.topk(attn_score, k=2, dim=-1)
        pos = self.prototypes[top2_idx[:, :, 0]] # [B, N, D]
        neg = self.prototypes[top2_idx[:, :, 1]] # [B, N, D]
    
        return v, query, pos, neg

    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        
        B, N, T = x.shape
        
        # Apply Input Prompt (Original logic)
        adaptive_params = torch.mm(
            self.U_basis[:N, :],          # [N, K]
            self.V_basis.T                # [K, D]
        )                                 # => [N, D]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        h_st = self.gcn2(x, adj)                                      # [bs, N, feature]
        
        # Low-rank core: X = U^T H V
        U = self.U_basis[:N, :]                    # [N, K]
        V = self.V_basis                           # [D, K]

        X = torch.einsum('nk,bnd,dk->bkd', U, h_st, V)

        # Prompt injection on bases
        U_tilde = U + self.P_s[:N, :]               # [N, K]
        V_tilde = V + self.P_t                     # [D, K]

        # Reconstruct: H~ = U~ X V~^T
        h_decoupled = torch.einsum(
            'nk,bkd,dk->bnd',
            U_tilde, X, V_tilde
        )  # [B, N, D]


        v_cur, q_cur, p_cur, n_cur= self.query_prototypes(h_decoupled)
        # shift from running anchors
        delta_s = v_cur.mean(0) - self.h_anchor_s[:N]
        delta_t = v_cur.mean((0, 1)) - self.h_anchor_t
        # gate output
        g_s, gate_s_act = self.gate_s(delta_s)
        g_t, gate_t_act = self.gate_t(delta_t)

        # update anchors during training
        if self.training:
            with torch.no_grad():
                
                self.h_anchor_s += g_s
                self.h_anchor_t += g_t
            # unfreeze prompts when the gate fires (||delta|| > tau)
            eps = 1e-6
            update_s = (gate_s_act > eps).item()
            update_t = (gate_t_act > eps).item()
            
            self.P_s.requires_grad_(update_s)
            self.P_t.requires_grad_(update_t)
            
            self.current_mode = (
                "both" if update_s and update_t else
                "spatial" if update_s else
                "temporal" if update_t else
                "freeze"
            )



        # --- gating and prompt gradients ---
        if self.training:
            # margin loss on prototype retrieval
            margin = 0.5
            d_pos = torch.norm(q_cur - p_cur, dim=-1)
            d_neg = torch.norm(q_cur - n_cur, dim=-1)
            l_con = torch.mean(F.relu(d_pos - d_neg + margin))

            # subsample nodes to avoid O(N^2) pairwise distances
            sample_idx = torch.randperm(h_decoupled.size(1))[:10]

            h_sample = h_decoupled[:, sample_idx, :]
            v_sample = v_cur[:, sample_idx, :]

            h_flat = h_sample.mean(0)   # [sample, D]
            v_flat = v_sample.mean(0)

            dist_h = torch.cdist(h_flat, h_flat)
            dist_v = torch.cdist(v_flat, v_flat)
            
            l_proto = F.mse_loss(dist_h, dist_v)
            
            self.aux_loss = l_con + l_proto

        
        
        interaction = h_st * (h_decoupled + v_cur)  # interaction after prototype shift

        x_fusion = torch.cat([h_st, interaction], dim=-1) # [B, N, Out * 2]
        
        x_out = self.fc(self.activation(x_fusion))       # [B, N, D]
        x_out = x_out.reshape(-1, self.backbone_out_dim) # [B*N, D]
        
        if x_out.shape == data.x.shape:
            x_out = x_out + data.x
        
        x_out = self.output(self.activation(x_out))      # [B*N, y_len]
        x = F.dropout(x_out, p=self.dropout, training=self.training)
    
        if self.training and hasattr(self.args, "dev_logger"):
            self.args.dev_logger.info(
                "Update_Stat",
                extra={
                    "year": self.args.year,
                    "epoch": getattr(self.args, "epoch", 0),
                    "Dt": delta_t.norm().item(),
                    "Ds": delta_s.norm().item(),
                    "G_t": gate_t_act.item() if torch.is_tensor(gate_t_act) else float(gate_t_act),
                    "G_s": gate_s_act.item() if torch.is_tensor(gate_s_act) else float(gate_s_act),
                    "tau_t": self.gate_t.tau.item(),
                    "tau_s": self.gate_s.tau.item(),
                    "mode": self.current_mode
                }
            )
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return
        device = self.U_basis.device
        dtype = self.U_basis.dtype
        # Expand U basis
        new_U = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                device=device,
                dtype=dtype
            )
        )
        nn.init.orthogonal_(new_U)
        self.U_basis = nn.Parameter(
            torch.cat([self.U_basis, new_U], dim=0)
        )
        # Expand spatial prompt
        new_Ps = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                device=device,
                dtype=dtype
            )
        )
        nn.init.uniform_(new_Ps, -0.05, 0.05)

        self.P_s = nn.Parameter(
            torch.cat([self.P_s, new_Ps], dim=0)
        )

        # Expand spatial anchor
        if hasattr(self, "h_anchor_s") and self.h_anchor_s is not None:
            # pad with zeros: new nodes have no historical shift
            new_anchor_s = torch.zeros(
                new_num_nodes - self.num_nodes,
                self.h_anchor_s.size(1),
                device=self.h_anchor_s.device,
                dtype=self.h_anchor_s.dtype
            )
            expanded_anchor = torch.cat(
                [self.h_anchor_s, new_anchor_s],
                dim=0
            )
            self.register_buffer("h_anchor_s", expanded_anchor)

        self.num_nodes = new_num_nodes


# DCRNN-style Backbone
class DCRNN_Backbone(nn.Module):
    """
    A simplified DCRNN-style backbone:
    diffusion conv (forward/backward) + GRUCell + graph conv out
    Input : [B, N, F]
    Output: [B, N, D]
    """
    def __init__(self, args):
        super(DCRNN_Backbone, self).__init__()

        self.in_dim = args.gcn["in_channel"]
        self.hidden_dim = args.gcn["hidden_channel"]
        self.out_dim = args.gcn["out_channel"]

        self.diffusion_conv_forward = BatchGCNConv(
            self.in_dim,
            self.hidden_dim // 2,
            bias=True,
            gcn=False
        )

        self.diffusion_conv_backward = BatchGCNConv(
            self.in_dim,
            self.hidden_dim // 2,
            bias=True,
            gcn=False
        )

        self.gru_cell = nn.GRUCell(
            self.hidden_dim,
            self.hidden_dim
        )

        self.diffusion_conv_out = BatchGCNConv(
            self.hidden_dim,
            self.out_dim,
            bias=True,
            gcn=False
        )

    def forward(self, x, adj):
        """
        x:   [B, N, F]
        adj: [N, N]
        """
        B, N, Fdim = x.shape

        backward_adj = adj.transpose(0, 1)

        forward_diff = F.relu(self.diffusion_conv_forward(x, adj))           # [B, N, H/2]
        backward_diff = F.relu(self.diffusion_conv_backward(x, backward_adj))# [B, N, H/2]

        diff_features = torch.cat([forward_diff, backward_diff], dim=-1)     # [B, N, H]
        diff_features_flat = diff_features.reshape(B * N, -1)                # [B*N, H]

        h0 = torch.zeros_like(diff_features_flat)                            # [B*N, H]
        h = self.gru_cell(diff_features_flat, h0)                            # [B*N, H]

        h = h.reshape(B, N, -1)                                              # [B, N, H]
        out = self.diffusion_conv_out(h, adj)                                # [B, N, D]

        return out


# Baseline DCRNN-style model
class DCRNN_Modelpre(nn.Module):
    """
    Baseline DCRNN-style model without plugin
    """
    def __init__(self, args):
        super(DCRNN_Modelpre, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank
        self.year = args.year
        self.num_nodes = args.base_node_size

        self.in_dim = args.gcn["in_channel"]
        self.hidden_dim = args.gcn["hidden_channel"]
        self.backbone_out_dim = args.gcn["out_channel"]

        # DCRNN-style backbone
        self.backbone = DCRNN_Backbone(args)

        # original low-rank adaptive input params (keep same spirit as EAC_Modelpre)
        self.U = nn.Parameter(
            torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1)
        )
        self.V = nn.Parameter(
            torch.empty(self.rank, self.in_dim).uniform_(-0.1, 0.1)
        )

        # output head
        self.fc = nn.Linear(self.backbone_out_dim, args.y_len)
        self.activation = nn.GELU()

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def forward(self, data, adj):
        N = adj.shape[0]

        x = data.x.reshape((-1, N, self.in_dim))   # [B, N, F]
        B, N, _ = x.shape

        # adaptive input params
        adaptive_params = torch.mm(self.U[:N, :], self.V)                    # [N, F]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)

        # backbone
        h = self.backbone(x, adj)                                            # [B, N, D]

        # prediction head
        h = h.reshape((-1, self.backbone_out_dim))                           # [B*N, D]

        # residual only when shape matches
        if h.shape == data.x.shape:
            h = h + data.x

        h = self.fc(self.activation(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(
                torch.empty(
                    new_num_nodes - self.num_nodes,
                    self.rank,
                    dtype=self.U.dtype,
                    device=self.U.device
                ).uniform_(-0.1, 0.1)
            )
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            self.num_nodes = new_num_nodes


# DCRNN + Plugin
class DCRNN_Model(nn.Module):
    """
    DCRNN-style backbone + plugin
    """
    def __init__(self, args):
        super(DCRNN_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank

        self.in_dim = args.gcn["in_channel"]
        self.hidden_dim = args.gcn["hidden_channel"]
        self.backbone_out_dim = args.gcn["out_channel"]

        self.year = args.year
        self.num_nodes = args.base_node_size

        # backbone
        self.backbone = DCRNN_Backbone(args)

        # Decoupling: H ≈ U X V^T
        self.U_basis = nn.Parameter(torch.empty(self.num_nodes, self.rank))
        self.V_basis = nn.Parameter(torch.empty(self.backbone_out_dim, self.rank))
        nn.init.orthogonal_(self.U_basis)
        nn.init.orthogonal_(self.V_basis)

        # Prompt on U and V
        self.P_s = nn.Parameter(torch.zeros(self.num_nodes, self.rank))
        self.P_t = nn.Parameter(torch.zeros(self.backbone_out_dim, self.rank))
        nn.init.uniform_(self.P_s, -0.05, 0.05)
        nn.init.uniform_(self.P_t, -0.05, 0.05)

        # prediction head
        self.fc = nn.Linear(self.backbone_out_dim * 2, self.backbone_out_dim)
        self.output = nn.Linear(self.backbone_out_dim, args.y_len)
        self.activation = nn.GELU()

        # prototypes
        self.prototypes = nn.Parameter(
            torch.randn(args.prototype_num, self.backbone_out_dim)
        )
        self.Wq = nn.Linear(self.backbone_out_dim, self.backbone_out_dim)

        # anchors
        self.register_buffer(
            "h_anchor_s",
            torch.zeros(self.num_nodes, self.backbone_out_dim)
        )  # [N, D]
        self.register_buffer(
            "h_anchor_t",
            torch.zeros(self.backbone_out_dim)
        )  # [D]

        # gates
        self.gate_s = DeviationGate(self.backbone_out_dim, init_tau=args.tau_s)
        self.gate_t = DeviationGate(self.backbone_out_dim, init_tau=args.tau_t)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    
    def get_aux_loss(self):
        return self.aux_loss if hasattr(self, "aux_loss") else 0.0

    def query_prototypes(self, h):
        """
        h: [B, N, D]
        """
        B, N, D = h.shape
        query = self.Wq(h)                                                    # [B, N, D]

        attn_score = torch.matmul(query, self.prototypes.t())                 # [B, N, M]
        attn_prob = torch.softmax(attn_score, dim=-1)

        v = torch.matmul(attn_prob, self.prototypes)                          # [B, N, D]

        _, top2_idx = torch.topk(attn_score, k=2, dim=-1)
        pos = self.prototypes[top2_idx[:, :, 0]]
        neg = self.prototypes[top2_idx[:, :, 1]]

        return v, query, pos, neg

    def forward(self, data, adj):
        N = adj.shape[0]

        x = data.x.reshape((-1, N, self.in_dim))                              # [B, N, F]
        B, N, _ = x.shape

        # input adaptive enhancement
        adaptive_params = torch.mm(
            self.U_basis[:N, :],                                              # [N, K]
            self.V_basis.T                                                    # [K, F]
        )                                                                     # [N, F]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)

        # backbone feature
        h_st = self.backbone(x, adj)                                          # [B, N, D]

        # Low-rank core
        U = self.U_basis[:N, :]                                               # [N, K]
        V = self.V_basis                                                      # [D, K]

        X = torch.einsum('nk,bnd,dk->bkd', U, h_st, V)

        # Prompt injection
        U_tilde = U + self.P_s[:N, :]
        V_tilde = V + self.P_t

        # Reconstruction
        h_decoupled = torch.einsum('nk,bkd,dk->bnd', U_tilde, X, V_tilde)

        # prototype query
        v_cur, q_cur, p_cur, n_cur = self.query_prototypes(h_decoupled)

        # deviation
        delta_s = v_cur.mean(0) - self.h_anchor_s[:N]
        delta_t = v_cur.mean((0, 1)) - self.h_anchor_t

        # gating
        g_s, gate_s_act = self.gate_s(delta_s)
        g_t, gate_t_act = self.gate_t(delta_t)

        # update anchor + prompt mode control
        if self.training:
            with torch.no_grad():
                self.h_anchor_s += g_s
                self.h_anchor_t += g_t

            eps = 1e-6
            update_s = (gate_s_act > eps).item()
            update_t = (gate_t_act > eps).item()

            self.P_s.requires_grad_(update_s)
            self.P_t.requires_grad_(update_t)

            self.current_mode = (
                "both" if update_s and update_t else
                "spatial" if update_s else
                "temporal" if update_t else
                "freeze"
            )

        # auxiliary loss
        if self.training:
            margin = 0.5
            d_pos = torch.norm(q_cur - p_cur, dim=-1)
            d_neg = torch.norm(q_cur - n_cur, dim=-1)
            l_con = torch.mean(F.relu(d_pos - d_neg + margin))

            sample_num = min(10, h_decoupled.size(1))
            sample_idx = torch.randperm(h_decoupled.size(1), device=h_decoupled.device)[:sample_num]

            h_sample = h_decoupled[:, sample_idx, :]
            v_sample = v_cur[:, sample_idx, :]

            h_flat = h_sample.mean(0)
            v_flat = v_sample.mean(0)

            dist_h = torch.cdist(h_flat, h_flat)
            dist_v = torch.cdist(v_flat, v_flat)

            l_proto = F.mse_loss(dist_h, dist_v)

            self.aux_loss = l_con + l_proto

        # fusion + output
        interaction = h_st * (h_decoupled + v_cur)
        x_fusion = torch.cat([h_st, interaction], dim=-1)                     # [B, N, 2D]

        x_out = self.fc(self.activation(x_fusion))                            # [B, N, D]
        x_out = x_out.reshape(-1, self.backbone_out_dim)                      # [B*N, D]

        if x_out.shape == data.x.shape:
            x_out = x_out + data.x

        x_out = self.output(self.activation(x_out))                           # [B*N, y_len]
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        if self.training and hasattr(self.args, "dev_logger"):
            self.args.dev_logger.info(
                "Update_Stat",
                extra={
                    "year": self.args.year,
                    "epoch": getattr(self.args, "epoch", 0),
                    "Dt": delta_t.norm().item(),
                    "Ds": delta_s.norm().item(),
                    "G_t": gate_t_act.item() if torch.is_tensor(gate_t_act) else float(gate_t_act),
                    "G_s": gate_s_act.item() if torch.is_tensor(gate_s_act) else float(gate_s_act),
                    "tau_t": self.gate_t.tau.item(),
                    "tau_s": self.gate_s.tau.item(),
                    "mode": self.current_mode
                }
            )

        return x_out

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return

        device = self.U_basis.device
        dtype = self.U_basis.dtype

        # expand U_basis
        new_U = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                device=device,
                dtype=dtype
            )
        )
        nn.init.orthogonal_(new_U)
        self.U_basis = nn.Parameter(torch.cat([self.U_basis, new_U], dim=0))

        # expand P_s
        new_Ps = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                device=device,
                dtype=dtype
            )
        )
        nn.init.uniform_(new_Ps, -0.05, 0.05)
        self.P_s = nn.Parameter(torch.cat([self.P_s, new_Ps], dim=0))

        # expand h_anchor_s
        if hasattr(self, "h_anchor_s") and self.h_anchor_s is not None:
            new_anchor_s = torch.zeros(
                new_num_nodes - self.num_nodes,
                self.h_anchor_s.size(1),
                device=self.h_anchor_s.device,
                dtype=self.h_anchor_s.dtype
            )
            expanded_anchor = torch.cat([self.h_anchor_s, new_anchor_s], dim=0)
            self.register_buffer("h_anchor_s", expanded_anchor)

        self.num_nodes = new_num_nodes

from functools import partial
from src.model.PDFormer import (
    TokenEmbedding,
    PositionalEncoding,
    LaplacianPE,
    DataEmbedding,
    DropPath,
    Mlp,
    STSelfAttention,
    STEncoderBlock,
)

def build_laplacian_pe(adj: torch.Tensor, lape_dim: int) -> torch.Tensor:
    """
    Build Laplacian positional encoding from adjacency matrix.

    Args:
        adj: [N, N] torch.Tensor
        lape_dim: number of eigenvectors to keep

    Returns:
        lap_pe: [N, lape_dim] torch.Tensor
    """
    device = adj.device
    dtype = adj.dtype

    A = adj.detach().cpu().numpy().astype(np.float64)
    N = A.shape[0]

    # symmetrize for numerical stability
    A = 0.5 * (A + A.T)

    D = np.diag(A.sum(axis=1))
    L = D - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        eigvecs = np.zeros((N, lape_dim), dtype=np.float64)
        return torch.tensor(eigvecs, dtype=dtype, device=device)

    start_idx = 1 if eigvecs.shape[1] > 1 else 0
    lap_pe = eigvecs[:, start_idx:start_idx + lape_dim]

    if lap_pe.shape[1] < lape_dim:
        pad = np.zeros((N, lape_dim - lap_pe.shape[1]), dtype=lap_pe.dtype)
        lap_pe = np.concatenate([lap_pe, pad], axis=1)

    return torch.tensor(lap_pe, dtype=dtype, device=device)

def build_geo_mask_from_adj(adj: torch.Tensor, max_hop: int = 3) -> torch.Tensor:
    """
    Build geo mask from adjacency using k-hop reachability.
    Mask=True means masked / blocked.

    Returns:
        geo_mask: [N, N] bool tensor
    """
    N = adj.shape[0]
    A = (adj > 0).float()
    A = torch.maximum(A, A.t())
    A.fill_diagonal_(1.0)

    reach = A.clone()
    cur = A.clone()
    for _ in range(max_hop - 1):
        cur = (cur @ A > 0).float()
        reach = torch.maximum(reach, cur)

    geo_mask = ~(reach.bool())
    return geo_mask

def build_sem_mask_from_x(x: torch.Tensor, topk: int = 20) -> torch.Tensor:
    """
    Build semantic mask from current batch input.
    x: [B, N, T]

    Returns:
        sem_mask: [N, N] bool tensor
    """
    with torch.no_grad():
        # [N, T]
        x_mean = x.mean(dim=0)
        x_norm = F.normalize(x_mean, p=2, dim=-1)
        sim = torch.matmul(x_norm, x_norm.t())  # [N, N]

        N = sim.size(0)
        k = min(topk, N)
        _, topk_idx = torch.topk(sim, k=k, dim=-1)

        keep = torch.zeros_like(sim, dtype=torch.bool)
        keep.scatter_(1, topk_idx, True)
        keep = keep | keep.t()

        eye = torch.eye(N, device=x.device, dtype=torch.bool)
        keep = keep | eye

        sem_mask = ~keep
    return sem_mask

def build_time_features(B: int, T: int, N: int, device: torch.device) -> torch.Tensor:
    """
    Build simple proxy time features for PDFormer-style input.

    Returns:
        time_feats: [B, T, N, 2]
        channel 0: normalized time-in-day proxy in [0,1)
        channel 1: normalized day-of-week proxy in [0,1]
    """
    # keep values < 1.0; DataEmbedding scales by 1440 and will OOB otherwise
    tid = torch.arange(T, device=device).float() / T
    tid = tid.view(1, T, 1, 1).expand(B, T, N, 1)

    diw = torch.zeros(B, T, N, 1, device=device)

    return torch.cat([tid, diw], dim=-1)

def build_local_patterns(x_emb: torch.Tensor, s_attn_size: int) -> torch.Tensor:
    """
    Build local temporal patterns from embedded sequence.

    Args:
        x_emb: [B, T, N, D]

    Returns:
        x_patterns: [B, T, N, S, D, 1]
    """
    B, T, N, D = x_emb.shape
    pad = s_attn_size // 2

    # [B, N, D, T]
    z = x_emb.permute(0, 2, 3, 1).contiguous()

    if pad > 0:
        left = z[..., :1].expand(-1, -1, -1, pad)    # replicate left boundary
        right = z[..., -1:].expand(-1, -1, -1, pad)  # replicate right boundary
        z = torch.cat([left, z, right], dim=-1)

    # [B, N, D, T, S]
    z = z.unfold(dimension=-1, size=s_attn_size, step=1)

    # [B, T, N, S, D]
    z = z.permute(0, 3, 1, 4, 2).contiguous()

    # [B, T, N, S, D, 1]
    z = z.unsqueeze(-1)
    return z

class PDFormerSTSelfAttention(nn.Module):
    """
    A framework-adapted full PDFormer-style ST attention:
    temporal + geo + semantic + pattern-enhanced geo branch
    """
    def __init__(
        self,
        dim,
        s_attn_size,
        t_attn_size,
        geo_num_heads=4,
        sem_num_heads=2,
        t_num_heads=2,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        device=torch.device("cpu"),
        output_dim=1,
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0

        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.output_dim = output_dim

        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio

        geo_dim = int(dim * self.geo_ratio)
        sem_dim = int(dim * self.sem_ratio)
        t_dim = int(dim * self.t_ratio)

        self.geo_dim = geo_dim
        self.sem_dim = sem_dim
        self.t_dim = t_dim

        # pattern branch
        self.pattern_q_linears = nn.ModuleList([
            nn.Linear(dim, geo_dim) for _ in range(output_dim)
        ])
        self.pattern_k_linears = nn.ModuleList([
            nn.Linear(dim, geo_dim) for _ in range(output_dim)
        ])
        self.pattern_v_linears = nn.ModuleList([
            nn.Linear(dim, geo_dim) for _ in range(output_dim)
        ])

        # geo branch
        self.geo_q_conv = nn.Conv2d(dim, geo_dim, kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, geo_dim, kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, geo_dim, kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        # semantic branch
        self.sem_q_conv = nn.Conv2d(dim, sem_dim, kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, sem_dim, kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, sem_dim, kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        # temporal branch
        self.t_q_conv = nn.Conv2d(dim, t_dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, t_dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, t_dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _pattern_context(self, x_patterns, pattern_keys):
        """
        x_patterns:   [B, T, N, S, D, O]
        pattern_keys: [N, S, D, O]

        Returns:
            ctx: [B, T, N, geo_dim]
        """
        B, T, N, S, D, O = x_patterns.shape
        ctx_total = 0.0

        for i in range(self.output_dim):
            # [B, T, N, S, D]
            xp = x_patterns[..., i]
            # [N, S, D]
            pk = pattern_keys[..., i]

            # linear on last dim
            # q: [B, T, N, S, G]
            q = self.pattern_q_linears[i](xp)
            # k/v: [N, S, G]
            k = self.pattern_k_linears[i](pk)
            v = self.pattern_v_linears[i](pk)

            # attn over pattern slots
            # [B, T, N, S, S]
            attn = torch.einsum("btnsg,nkg->btnsk", q, k) * self.scale
            attn = attn.softmax(dim=-1)

            # [B, T, N, S, G]
            out = torch.einsum("btnsk,nkg->btnsg", attn, v)

            # aggregate local slots -> [B, T, N, G]
            out = out.mean(dim=3)
            ctx_total = ctx_total + out

        return ctx_total

    def forward(self, x, x_patterns=None, pattern_keys=None, geo_mask=None, sem_mask=None):
        """
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape

        # -------- temporal branch --------
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)

        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, self.t_dim).transpose(1, 2)

        # -------- geo branch --------
        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if x_patterns is not None and pattern_keys is not None:
            geo_k = geo_k + self._pattern_context(x_patterns, pattern_keys)

        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn = geo_attn.masked_fill(geo_mask, float("-inf"))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, self.geo_dim)

        # -------- semantic branch --------
        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn = sem_attn.masked_fill(sem_mask, float("-inf"))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, self.sem_dim)

        out = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        out = self.proj_drop(out)
        return out

class PDFormerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        s_attn_size,
        t_attn_size,
        geo_num_heads=4,
        sem_num_heads=2,
        t_num_heads=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        device=torch.device("cpu"),
        type_ln="pre",
        output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln

        self.norm1 = norm_layer(dim)
        self.st_attn = PDFormerSTSelfAttention(
            dim=dim,
            s_attn_size=s_attn_size,
            t_attn_size=t_attn_size,
            geo_num_heads=geo_num_heads,
            sem_num_heads=sem_num_heads,
            t_num_heads=t_num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            device=device,
            output_dim=output_dim,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        if self.type_ln == "pre":
            x = x + self.drop_path(
                self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)
            )
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm1(
                x + self.drop_path(
                    self.st_attn(x, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)
                )
            )
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

class PDFormer_Backbone(nn.Module):
    """
    Fuller PDFormer-style backbone adapted to your framework.

    Input:
        x:   [B, N, T]
        adj: [N, N]

    Output:
        h:   [B, N, D]
    """
    def __init__(self, args):
        super(PDFormer_Backbone, self).__init__()
        self.args = args

        self.seq_len = args.gcn["in_channel"]
        self.embed_dim = getattr(args, "embed_dim", 64)
        self.backbone_out_dim = args.gcn["out_channel"]

        self.lape_dim = getattr(args, "lape_dim", 8)
        self.geo_num_heads = getattr(args, "geo_num_heads", 4)
        self.sem_num_heads = getattr(args, "sem_num_heads", 2)
        self.t_num_heads = getattr(args, "t_num_heads", 2)
        self.mlp_ratio = getattr(args, "mlp_ratio", 4.0)
        self.qkv_bias = getattr(args, "qkv_bias", True)
        self.attn_drop = getattr(args, "attn_drop", args.dropout)
        self.drop_path = getattr(args, "drop_path", 0.0)
        self.s_attn_size = getattr(args, "s_attn_size", 3)
        self.t_attn_size = getattr(args, "t_attn_size", 3)
        self.enc_depth = getattr(args, "enc_depth", 3)
        self.type_ln = getattr(args, "type_ln", "pre")
        self.geo_mask_hop = getattr(args, "geo_mask_hop", 3)
        self.sem_topk = getattr(args, "sem_topk", 20)

        # use main value + proxy time features
        self.feature_dim = 1
        self.output_dim = 1

        self.enc_embed_layer = DataEmbedding(
            feature_dim=self.feature_dim,
            embed_dim=self.embed_dim,
            lape_dim=self.lape_dim,
            adj_mx=None,
            drop=args.dropout,
            add_time_in_day=True,
            add_day_in_week=False,
            device=args.device,
        )

        enc_dpr = torch.linspace(0, self.drop_path, self.enc_depth).tolist()
        self.encoder_blocks = nn.ModuleList([
            PDFormerEncoderBlock(
                dim=self.embed_dim,
                s_attn_size=self.s_attn_size,
                t_attn_size=self.t_attn_size,
                geo_num_heads=self.geo_num_heads,
                sem_num_heads=self.sem_num_heads,
                t_num_heads=self.t_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=args.dropout,
                attn_drop=self.attn_drop,
                drop_path=enc_dpr[i],
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                device=args.device,
                type_ln=self.type_ln,
                output_dim=self.output_dim,
            )
            for i in range(self.enc_depth)
        ])

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(self.embed_dim, self.backbone_out_dim)

        # learnable pattern memory
        max_nodes = getattr(args, "max_node_size", args.base_node_size)
        self.pattern_keys_param = nn.Parameter(
            torch.randn(max_nodes, self.s_attn_size, self.embed_dim, self.output_dim) * 0.02
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, T] -> [B, T, N, 1 + time_feats]
        """
        B, N, T = x.shape
        value = x.permute(0, 2, 1).unsqueeze(-1)  # [B, T, N, 1]
        time_feats = build_time_features(B, T, N, x.device)  # [B, T, N, 2]

        # DataEmbedding with add_time_in_day=True, add_day_in_week=False
        # expects: [:feature_dim] as value, [:, :, :, feature_dim] as time_in_day proxy
        x_full = torch.cat([value, time_feats[..., :1]], dim=-1)  # [B, T, N, 2]
        return x_full

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, T]
            adj: [N, N]

        Returns:
            h: [B, N, D]
        """
        B, N, T = x.shape

        x_in = self._prepare_input(x)                        # [B, T, N, 2]
        lap_mx = build_laplacian_pe(adj, self.lape_dim)     # [N, lape_dim]

        enc = self.enc_embed_layer(x_in, lap_mx)            # [B, T, N, embed_dim]

        # build pattern inputs
        x_patterns = build_local_patterns(enc, self.s_attn_size)          # [B, T, N, S, D, 1]
        pattern_keys = self.pattern_keys_param[:N]                        # [N, S, D, 1]

        # build masks
        geo_mask = build_geo_mask_from_adj(adj, max_hop=self.geo_mask_hop)   # [N, N]
        sem_mask = build_sem_mask_from_x(x, topk=self.sem_topk)              # [N, N]

        # expand to attention broadcast shape: [1, 1, 1, N, N]
        geo_mask = geo_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        sem_mask = sem_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        for block in self.encoder_blocks:
            enc = block(enc, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)

        # [B, T, N, D] -> [B, N, D]
        enc = enc.permute(0, 2, 3, 1)                  # [B, N, D, T]
        enc = enc.reshape(B * N, self.embed_dim, T)    # [B*N, D, T]
        enc = self.temporal_pool(enc).squeeze(-1)      # [B*N, D]
        enc = enc.reshape(B, N, self.embed_dim)        # [B, N, D]

        h = self.out_proj(enc)                         # [B, N, backbone_out_dim]
        return h

class PDFormer_Modelpre(nn.Module):
    """
    Baseline PDFormer-style model without plugin.
    Strictly matched with the enhanced PDFormer_Backbone.
    """
    def __init__(self, args):
        super(PDFormer_Modelpre, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank
        self.year = args.year
        self.num_nodes = args.base_node_size

        self.in_dim = args.gcn["in_channel"]           # history length, e.g. 12
        self.backbone_out_dim = args.gcn["out_channel"]

        self.backbone = PDFormer_Backbone(args)

        # keep the same low-rank adaptive input logic as other "pre" baselines
        self.U = nn.Parameter(
            torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1)
        )
        self.V = nn.Parameter(
            torch.empty(self.rank, self.in_dim).uniform_(-0.1, 0.1)
        )

        self.fc = nn.Linear(self.backbone_out_dim, args.y_len)
        self.activation = nn.GELU()

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def forward(self, data, adj):
        N = adj.shape[0]

        # [B*N, x_len] -> [B, N, x_len]
        x = data.x.reshape((-1, N, self.in_dim))
        B, N, _ = x.shape

        # low-rank adaptive input prompt
        adaptive_params = torch.mm(self.U[:N, :], self.V)   # [N, T]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)

        # backbone feature
        h = self.backbone(x, adj)                           # [B, N, D]

        # prediction head
        h = h.reshape((-1, self.backbone_out_dim))          # [B*N, D]

        if h.shape == data.x.shape:
            h = h + data.x

        h = self.fc(self.activation(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return

        new_params = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                dtype=self.U.dtype,
                device=self.U.device
            ).uniform_(-0.1, 0.1)
        )
        self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
        self.num_nodes = new_num_nodes

class PDFormer_Model(nn.Module):
    """
    PDFormer-style backbone + your plugin.
    Strictly matched with the enhanced PDFormer_Backbone.
    """
    def __init__(self, args):
        super(PDFormer_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.rank = args.rank

        self.in_dim = args.gcn["in_channel"]
        self.backbone_out_dim = args.gcn["out_channel"]

        self.year = args.year
        self.num_nodes = args.base_node_size

        # backbone
        self.backbone = PDFormer_Backbone(args)

        # Decoupling: H ≈ U X V^T
        self.U_basis = nn.Parameter(torch.empty(self.num_nodes, self.rank))
        self.V_basis = nn.Parameter(torch.empty(self.backbone_out_dim, self.rank))
        nn.init.orthogonal_(self.U_basis)
        nn.init.orthogonal_(self.V_basis)

        # Prompt on U and V
        self.P_s = nn.Parameter(torch.zeros(self.num_nodes, self.rank))
        self.P_t = nn.Parameter(torch.zeros(self.backbone_out_dim, self.rank))
        nn.init.uniform_(self.P_s, -0.05, 0.05)
        nn.init.uniform_(self.P_t, -0.05, 0.05)

        # Prediction head
        self.fc = nn.Linear(self.backbone_out_dim * 2, self.backbone_out_dim)
        self.output = nn.Linear(self.backbone_out_dim, args.y_len)
        self.activation = nn.GELU()

        # prototypes
        self.prototypes = nn.Parameter(
            torch.randn(args.prototype_num, self.backbone_out_dim)
        )
        self.Wq = nn.Linear(self.backbone_out_dim, self.backbone_out_dim)

        # anchors
        self.register_buffer(
            "h_anchor_s",
            torch.zeros(self.num_nodes, self.backbone_out_dim)
        )  # [N, D]
        self.register_buffer(
            "h_anchor_t",
            torch.zeros(self.backbone_out_dim)
        )  # [D]

        # gates
        self.gate_s = DeviationGate(self.backbone_out_dim, init_tau=args.tau_s)
        self.gate_t = DeviationGate(self.backbone_out_dim, init_tau=args.tau_t)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def get_decoupled_features(self, data, adj):
        """
        Used for warm-up / detection.
        """
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.in_dim))   # [B, N, T]
        B, N, _ = x.shape

        # input adaptive enhancement
        adaptive_params = torch.mm(self.U_basis[:N, :], self.V_basis.T)   # [N, T]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)

        # backbone feature
        h_st = self.backbone(x, adj)                                       # [B, N, D]

        # decoupling
        U = self.U_basis[:N, :]                                            # [N, K]
        V = self.V_basis                                                   # [D, K]

        X = torch.einsum('nk,bnd,dk->bkd', U, h_st, V)

        U_tilde = U + self.P_s[:N, :]
        V_tilde = V + self.P_t

        h_decoupled = torch.einsum('nk,bkd,dk->bnd', U_tilde, X, V_tilde)

        return h_decoupled, h_st

    def get_aux_loss(self):
        return self.aux_loss if hasattr(self, "aux_loss") else 0.0

    def query_prototypes(self, h):
        """
        h: [B, N, D]
        returns:
            v, query, pos, neg
        """
        query = self.Wq(h)                                                 # [B, N, D]

        attn_score = torch.matmul(query, self.prototypes.t())              # [B, N, M]
        attn_prob = torch.softmax(attn_score, dim=-1)

        v = torch.matmul(attn_prob, self.prototypes)                       # [B, N, D]

        _, top2_idx = torch.topk(attn_score, k=2, dim=-1)
        pos = self.prototypes[top2_idx[:, :, 0]]
        neg = self.prototypes[top2_idx[:, :, 1]]

        return v, query, pos, neg

    def forward(self, data, adj):
        N = adj.shape[0]

        x = data.x.reshape((-1, N, self.in_dim))                           # [B, N, T]
        B, N, _ = x.shape

        # input adaptive enhancement
        adaptive_params = torch.mm(
            self.U_basis[:N, :],                                           # [N, K]
            self.V_basis.T                                                 # [K, T]
        )                                                                  # [N, T]
        x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)

        # backbone feature
        h_st = self.backbone(x, adj)                                       # [B, N, D]

        # Low-rank core
        U = self.U_basis[:N, :]                                            # [N, K]
        V = self.V_basis                                                   # [D, K]
        X = torch.einsum('nk,bnd,dk->bkd', U, h_st, V)

        # Prompt injection
        U_tilde = U + self.P_s[:N, :]
        V_tilde = V + self.P_t

        # Reconstruction
        h_decoupled = torch.einsum('nk,bkd,dk->bnd', U_tilde, X, V_tilde)

        # prototype query
        v_cur, q_cur, p_cur, n_cur = self.query_prototypes(h_decoupled)

        # deviation
        delta_s = v_cur.mean(0) - self.h_anchor_s
        delta_t = v_cur.mean((0, 1)) - self.h_anchor_t

        # gating
        g_s, gate_s_act = self.gate_s(delta_s)
        g_t, gate_t_act = self.gate_t(delta_t)

        # update anchor + prompt mode control
        if self.training:
            with torch.no_grad():
                # if node size is dynamic, h_anchor_s has already been expanded by expand_adaptive_params
                self.h_anchor_s += g_s
                self.h_anchor_t += g_t

            eps = 1e-6
            update_s = (gate_s_act > eps).item()
            update_t = (gate_t_act > eps).item()

            self.P_s.requires_grad_(update_s)
            self.P_t.requires_grad_(update_t)

            self.current_mode = (
                "both" if update_s and update_t else
                "spatial" if update_s else
                "temporal" if update_t else
                "freeze"
            )

        # auxiliary loss
        if self.training:
            margin = 0.5
            d_pos = torch.norm(q_cur - p_cur, dim=-1)
            d_neg = torch.norm(q_cur - n_cur, dim=-1)
            l_con = torch.mean(F.relu(d_pos - d_neg + margin))

            sample_num = min(10, h_decoupled.size(1))
            sample_idx = torch.randperm(
                h_decoupled.size(1),
                device=h_decoupled.device
            )[:sample_num]

            h_sample = h_decoupled[:, sample_idx, :]
            v_sample = v_cur[:, sample_idx, :]

            h_flat = h_sample.mean(0)
            v_flat = v_sample.mean(0)

            dist_h = torch.cdist(h_flat, h_flat)
            dist_v = torch.cdist(v_flat, v_flat)

            l_proto = F.mse_loss(dist_h, dist_v)
            self.aux_loss = l_con + l_proto

        # fusion + output
        interaction = h_st * (h_decoupled + v_cur)
        x_fusion = torch.cat([h_st, interaction], dim=-1)                  # [B, N, 2D]

        x_out = self.fc(self.activation(x_fusion))                         # [B, N, D]
        x_out = x_out.reshape(-1, self.backbone_out_dim)                   # [B*N, D]

        if x_out.shape == data.x.shape:
            x_out = x_out + data.x

        x_out = self.output(self.activation(x_out))                        # [B*N, y_len]
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)

        if self.training and hasattr(self.args, "dev_logger"):
            self.args.dev_logger.info(
                "Update_Stat",
                extra={
                    "year": self.args.year,
                    "epoch": getattr(self.args, "epoch", 0),
                    "Dt": delta_t.norm().item(),
                    "Ds": delta_s.norm().item(),
                    "G_t": gate_t_act.item() if torch.is_tensor(gate_t_act) else float(gate_t_act),
                    "G_s": gate_s_act.item() if torch.is_tensor(gate_s_act) else float(gate_s_act),
                    "tau_t": self.gate_t.tau.item(),
                    "tau_s": self.gate_s.tau.item(),
                    "mode": self.current_mode
                }
            )

        return x_out

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return

        device = self.U_basis.device
        dtype = self.U_basis.dtype

        # expand U_basis
        new_U = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                device=device,
                dtype=dtype
            )
        )
        nn.init.orthogonal_(new_U)
        self.U_basis = nn.Parameter(torch.cat([self.U_basis, new_U], dim=0))

        # expand P_s
        new_Ps = nn.Parameter(
            torch.empty(
                new_num_nodes - self.num_nodes,
                self.rank,
                device=device,
                dtype=dtype
            )
        )
        nn.init.uniform_(new_Ps, -0.05, 0.05)
        self.P_s = nn.Parameter(torch.cat([self.P_s, new_Ps], dim=0))

        # expand h_anchor_s
        if hasattr(self, "h_anchor_s") and self.h_anchor_s is not None:
            new_anchor_s = torch.zeros(
                new_num_nodes - self.num_nodes,
                self.h_anchor_s.size(1),
                device=self.h_anchor_s.device,
                dtype=self.h_anchor_s.dtype
            )
            expanded_anchor = torch.cat([self.h_anchor_s, new_anchor_s], dim=0)
            self.register_buffer("h_anchor_s", expanded_anchor)

        # expand pattern memory inside backbone if needed
        if hasattr(self.backbone, "pattern_keys_param"):
            cur_max = self.backbone.pattern_keys_param.size(0)
            if new_num_nodes > cur_max:
                old = self.backbone.pattern_keys_param
                extra = nn.Parameter(
                    torch.randn(
                        new_num_nodes - cur_max,
                        self.backbone.s_attn_size,
                        self.backbone.embed_dim,
                        self.backbone.output_dim,
                        device=old.device,
                        dtype=old.dtype
                    ) * 0.02
                )
                self.backbone.pattern_keys_param = nn.Parameter(
                    torch.cat([old, extra], dim=0)
                )

        self.num_nodes = new_num_nodes


class TrafficStream_Model(nn.Module):
    """Some Information about TrafficStream_Model"""
    def __init__(self, args):
        super(TrafficStream_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    

    def feature(self, data, adj):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]        
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        x = x + data.x
        return x

class STKEC_Model(nn.Module):
    """Some Information about STKEC_Model"""
    def __init__(self, args):
        super(STKEC_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.ReLU()

        self.memory=nn.Parameter(torch.zeros(size=(args.cluster, args.gcn["out_channel"]), requires_grad=True))
        nn.init.xavier_uniform_(self.memory, gain=1.414)
        
    def forward(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]
        scores = F.softmax(attention, dim=1)                       # [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x, scores
    
    def feature(self, data, adj, scores=None):
        N = adj.shape[0]
        
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))   # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj))                              # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))          # [bs * N, feature]
        
        attention = torch.matmul(x, self.memory.transpose(-1, -2)) # [bs * N, feature] * [feature , K] = [bs * N, K]

        z = torch.matmul(attention, self.memory)                   # [bs * N, K] * [K, feature] = [bs * N, feature]
        x = x + data.x + z
        return x
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

class Universal_Model(nn.Module):
    def __init__(self, args):
        super(Universal_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.use_eac = args.use_eac
        
        # Initialize GCN layers based on spectral (sp) or spatial (st) options
        if args.gcn_type == 'st':
            self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
            self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["in_channel"], bias=True, gcn=False)
        elif args.gcn_type == 'sp':
            self.gcn1 = ChebGraphConv(args.gcn["in_channel"], args.gcn["hidden_channel"])
            self.gcn2 = ChebGraphConv(args.gcn["hidden_channel"], args.gcn["in_channel"])
        
        # Select TCN type based on args
        if args.tcn_type == 'conv':
            self.tcn = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], 
                                kernel_size=args.tcn["kernel_size"],
                                dilation=args.tcn["dilation"],
                                padding=int((args.tcn["kernel_size"] - 1) * args.tcn["dilation"] / 2))
        elif args.tcn_type == 'rec':
            self.tcn = nn.LSTM(input_size=args.gcn["hidden_channel"], hidden_size=args.gcn["hidden_channel"], batch_first=True)
        elif args.tcn_type == 'attn':
            self.tcn = nn.MultiheadAttention(embed_dim=args.gcn["hidden_channel"], num_heads=4)
        
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        
        if self.use_eac:
            self.rank = args.rank  # low-rank factor size
            self.U = nn.Parameter(torch.empty(args.base_node_size, self.rank).uniform_(-0.1, 0.1))
            self.V = nn.Parameter(torch.empty(self.rank, args.gcn["in_channel"]).uniform_(-0.1, 0.1))
            self.year = args.year
            self.num_nodes = args.base_node_size
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.gcn["in_channel"]))  # [bs, N, feature]
        
        B, N, T = x.shape
        
        if self.use_eac:
            adaptive_params = torch.mm(self.U[:N, :], self.V)  # [N, feature_dim]
            x = x + adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape)  # [bs, N, feature]
        
        # Apply the selected GCN layers
        x = F.relu(self.gcn1(x, adj))  # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))  # [bs * N, 1, feature]
        
        # Apply the selected TCN method
        if self.args.tcn_type == 'conv':
            x = self.tcn(x)  # temporal convolution
        elif self.args.tcn_type == 'rec':
            # x = x.reshape((-1, self.args.gcn["hidden_channel"])).unsqueeze(dim=-1)
            # out, _ = self.tcn(x)
            # x = out.reshape((-1, 1, self.args.gcn["hidden_channel"]))
            x = x.reshape(B, N, self.args.gcn["hidden_channel"])
            x, _ = self.tcn(x)
            x = x.reshape(B*N, 1, self.args.gcn["hidden_channel"])
        elif self.args.tcn_type == 'attn':
            x = x.reshape(B, N, self.args.gcn["hidden_channel"])
            x, _ = self.tcn(x, x, x)  # Multihead attention
            x = x.reshape(B*N, 1, self.args.gcn["hidden_channel"])
        
        
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))  # [bs, N, feature]
        x = self.gcn2(x, adj)  # [bs, N, feature]
        x = x.reshape((-1, self.args.gcn["out_channel"]))  # [bs * N, feature]
        
        x = x + data.x
        x = self.fc(self.activation(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            
            self.num_nodes = new_num_nodes


class STID_MLPBlock(nn.Module):
    """
    STID encoder block: 1x1 Conv -> ReLU -> Dropout -> 1x1 Conv + residual
    I/O shape: [B, C, N, 1]
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super(STID_MLPBlock, self).__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        hidden = self.fc2(self.drop(self.act(self.fc1(x))))
        hidden = hidden + x
        return hidden

class STID_Model(nn.Module):
    """
    STID baseline wired to this project's data/adj interface.
    data.x: [B*N, x_len]
    returns pred: [B*N, y_len]
    """
    def __init__(self, args):
        super(STID_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout

        # retrain baseline: node count follows current graph size
        #self.num_nodes = args.graph_size
        self.num_nodes = args.init_graph_size if hasattr(args, "init_graph_size") else args.graph_size

        # history length from gcn["in_channel"]
        self.input_len = args.gcn["in_channel"]
        self.input_dim = 1
        self.output_len = args.y_len

        stid_cfg = args.stid if hasattr(args, "stid") else {}

        self.node_dim = stid_cfg.get("node_dim", 32)
        self.embed_dim = stid_cfg.get("embed_dim", 32)
        self.num_layer = stid_cfg.get("num_layer", 3)
        self.temp_dim_tid = stid_cfg.get("temp_dim_tid", 0)
        self.temp_dim_diw = stid_cfg.get("temp_dim_diw", 0)

        self.if_node = stid_cfg.get("if_node", True)
        self.if_time_in_day = stid_cfg.get("if_T_i_D", False)
        self.if_day_in_week = stid_cfg.get("if_D_i_W", False)

        self.time_of_day_size = stid_cfg.get("time_of_day_size", 288)
        self.day_of_week_size = stid_cfg.get("day_of_week_size", 7)

        # spatial embedding
        if self.if_node:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # temporal embedding (enable when extra time features are available)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # time series embedding
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True
        )

        # hidden dim
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_node)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )

        # encoder
        self.encoder = nn.Sequential(
            *[STID_MLPBlock(self.hidden_dim, self.hidden_dim, dropout=0.15)
              for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True
        )
    def expand_adaptive_params(self, new_num_nodes):
        if not self.if_node:
            self.num_nodes = new_num_nodes
            return

        if new_num_nodes <= self.num_nodes:
            return

        device = self.node_emb.device
        dtype = self.node_emb.dtype

        new_emb = torch.empty(
            new_num_nodes - self.num_nodes,
            self.node_dim,
            device=device,
            dtype=dtype
        )
        nn.init.xavier_uniform_(new_emb)

        self.node_emb = nn.Parameter(torch.cat([self.node_emb.data, new_emb], dim=0))
        self.num_nodes = new_num_nodes
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def get_aux_loss(self):
        return 0.0

    def forward(self, data, adj):
        """
        data.x: [B*N, x_len]
        adj:    [N, N]   # unused by STID; kept for a shared forward signature

        return:
            [B*N, y_len]
        """
        N = adj.shape[0]
        total_rows = data.x.shape[0]

        if total_rows % N != 0:
            raise RuntimeError(
                f"[STID] data.x.shape={data.x.shape}, adj.shape={adj.shape}, "
                f"total_rows={total_rows} cannot be divided by N={N}."
            )

        B = total_rows // N

        # [B*N, x_len] -> [B, N, L]
        x = data.x.reshape(B, N, self.input_len)

        # STID official input: [B, L, N, C]
        history_data = x.permute(0, 2, 1).unsqueeze(-1)   # [B, L, N, 1]

        # prepare data
        input_data = history_data[..., :self.input_dim]   # [B, L, N, 1]

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()      # [B, N, L, 1]
        input_data = input_data.view(batch_size, num_nodes, -1)   # [B, N, L*C]
        input_data = input_data.transpose(1, 2).unsqueeze(-1)     # [B, L*C, N, 1]
        time_series_emb = self.time_series_emb_layer(input_data)  # [B, embed_dim, N, 1]

        # node embedding
        node_emb = []
        if self.if_node:
            node_emb.append(
                self.node_emb[:N].unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            )  # [B, node_dim, N, 1]

        # temporal branch empty when time embeddings are disabled
        tem_emb = []

        # concat
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)   # [B, y_len, N, 1]

        # [B, y_len, N, 1] -> [B*N, y_len]
        prediction = prediction.squeeze(-1).permute(0, 2, 1).reshape(-1, self.output_len)

        return prediction

    def feature(self, data, adj):
        return self.forward(data, adj)


# Graph WaveNet Components
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: [B, C, N, T]
        # A: [N, N]
        x = torch.einsum('bcnt,nm->bcmt', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.order = order
        self.dropout = dropout

        total_c_in = (order * support_len + 1) * c_in
        self.mlp = linear(total_c_in, c_out)

    def forward(self, x, support):
        out = [x]

        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)

            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gwnet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=1,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2
    ):
        super(gwnet, self).__init__()

        self.device = device
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.num_nodes = num_nodes

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1)
        )

        self.supports = supports
        if supports is None:
            self.supports_len = 1 if gcn_bool else 0
        else:
            self.supports_len = len(supports)

        receptive_field = 1

        if self.gcn_bool and self.addaptadj:
            if self.supports is None:
                self.supports = []

            if aptinit is None:
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10, device=device),
                    requires_grad=True
                )
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes, device=device),
                    requires_grad=True
                )
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1.to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2.to(device), requires_grad=True)

            self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1

            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, new_dilation)
                    )
                )

                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, new_dilation)
                    )
                )

                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1)
                    )
                )

                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1)
                    )
                )

                self.bn.append(nn.BatchNorm2d(residual_channels))

                if self.gcn_bool:
                    self.gconv.append(
                        gcn(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len
                        )
                    )

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True
        )

        self.receptive_field = receptive_field

    def _build_supports(self, adj):
        """
        Build supports from the current adj for full / subgraph / incremental runs.
        """
        supports = []

        if adj is not None:
            supports.append(adj)

        if self.gcn_bool and self.addaptadj:
            N = adj.size(0) if adj is not None else self.num_nodes
            nodevec1 = self.nodevec1[:N, :]
            nodevec2 = self.nodevec2[:, :N]
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            supports.append(adp)

        return supports

    def forward(self, input, adj=None):
        # input: [B, C, N, T]
        in_len = input.size(3)

        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = None

        supports = self._build_supports(adj) if self.gcn_bool else None

        for i in range(self.blocks * self.layers):
            residual = x

            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = self.skip_convs[i](x)
            if skip is None:
                skip = s
            else:
                skip = skip[:, :, :, -s.size(3):]
                skip = skip + s

            if self.gcn_bool and supports is not None and len(supports) > 0:
                x = self.gconv[i](x, supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)   # [B, out_dim, N, T_out]

        return x

    def expand_adaptive_params(self, new_num_nodes):
        """Expand adaptive adjacency when new nodes are added."""
        if new_num_nodes <= self.num_nodes:
            return

        device = self.nodevec1.device
        dtype = self.nodevec1.dtype
        old_num_nodes = self.num_nodes

        new_nodevec1 = torch.randn(
            new_num_nodes - old_num_nodes, 10,
            device=device, dtype=dtype
        )
        new_nodevec2 = torch.randn(
            10, new_num_nodes - old_num_nodes,
            device=device, dtype=dtype
        )

        self.nodevec1 = nn.Parameter(
            torch.cat([self.nodevec1.data, new_nodevec1], dim=0),
            requires_grad=True
        )
        self.nodevec2 = nn.Parameter(
            torch.cat([self.nodevec2.data, new_nodevec2], dim=1),
            requires_grad=True
        )

        self.num_nodes = new_num_nodes

# GraphWaveNet baseline for your project
class GraphWaveNet_Model(nn.Module):
    """
    GraphWaveNet baseline wired to this project's data/adj interface.
    data.x: [B*N, x_len]
    returns pred: [B*N, y_len]
    """
    def __init__(self, args):
        super(GraphWaveNet_Model, self).__init__()
        self.args = args
        self.dropout = args.dropout

        #init_num_nodes = args.base_node_size if hasattr(args, "base_node_size") else args.graph_size
        #init_num_nodes = args.graph_size
        init_num_nodes = args.init_graph_size if hasattr(args, "init_graph_size") else args.graph_size

        self.x_len = args.gcn["in_channel"]
        self.y_len = args.y_len

        gwnet_cfg = args.gwnet if hasattr(args, "gwnet") else {}

        self.model = gwnet(
            device=args.device,
            num_nodes=init_num_nodes,
            dropout=args.dropout,
            supports=None,
            gcn_bool=gwnet_cfg.get("gcn_bool", True),
            addaptadj=gwnet_cfg.get("addaptadj", True),
            aptinit=None,
            in_dim=1,
            out_dim=self.y_len,
            residual_channels=gwnet_cfg.get("residual_channels", 32),
            dilation_channels=gwnet_cfg.get("dilation_channels", 32),
            skip_channels=gwnet_cfg.get("skip_channels", 256),
            end_channels=gwnet_cfg.get("end_channels", 512),
            kernel_size=gwnet_cfg.get("kernel_size", 2),
            blocks=gwnet_cfg.get("blocks", 4),
            layers=gwnet_cfg.get("layers", 2)
        )
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")

    def get_aux_loss(self):
        return 0.0

    def expand_adaptive_params(self, new_num_nodes):
        self.model.expand_adaptive_params(new_num_nodes)

    def forward(self, data, adj):
        """
        data.x: [B*N, x_len]
        adj:    [N, N]

        return:
            [B*N, y_len]
        """
        N = adj.shape[0]

        # data.x -> [B, N, T]
        x = data.x.reshape((-1, N, self.x_len))

        # GraphWaveNet expects [B, C, N, T]
        x = x.unsqueeze(1)  # [B, 1, N, T]

        # forward
        out = self.model(x, adj)  # [B, y_len, N, T_out]

        # time dim is usually 1; take the last step if not
        if out.size(-1) > 1:
            out = out[:, :, :, -1:]
        out = out.squeeze(-1)      # [B, y_len, N]
        out = out.permute(0, 2, 1) # [B, N, y_len]

        out = out.reshape(-1, self.y_len)  # [B*N, y_len]
        return out

    def feature(self, data, adj):
        """
        Feature hook for downstream detection; returns the forward output for now.
        """
        return self.forward(data, adj)
