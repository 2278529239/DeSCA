import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        return input


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1,1))
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1,1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Conv(nn.Module):
    def __init__(self, features, dropout=0.2):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class DLGA(nn.Module): 
    def __init__(self, d_model, head, seq_length=1):
        super(DLGA, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head 
        self.head = head
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

    def forward(self, input, node_emb):
        _, d_model, num_nodes, seq_length = input.shape
        query, key, value = self.q(input), self.k(input), self.v(input)
        query = query.view(
            query.shape[0], -1, self.d_k, query.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)
        key = key.view(
            key.shape[0], -1, self.d_k, key.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)  
        node_emb = node_emb.view(
            node_emb.shape[0], -1, self.d_k, node_emb.shape[2], seq_length
        ).permute(0, 1, 4, 3, 2)

        key = torch.softmax(key / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        node_emb = torch.softmax(node_emb / math.sqrt(self.d_k), dim=-1)

        kv = torch.einsum("bhlnx, bhlny->bhlxy", key, value)
        attn_qkv1 = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)

        node_emb_v = torch.einsum("bhlnx, bhlny->bhlxy", node_emb, value)
        attn_qkv2 = torch.einsum("bhlnx, bhlxy->bhlny", query, node_emb_v)

        attn_qkv = attn_qkv1 + attn_qkv2

        x = (
            attn_qkv.permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(attn_qkv.shape[0],d_model, num_nodes, seq_length)
        )
        x = self.concat(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Encoder(nn.Module):
    def __init__(self, d_model, head, seq_length=1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head 
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model

        self.attn = DLGA(
            d_model, head, seq_length
        )

        self.FFW = Mlp(d_model)

        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()
    
    def forward(self, x, node_emb):
        gate, scale, memory = torch.chunk(node_emb, 3, dim=1)
        x = x + gate * self.attn(modulate(self.norm1(x), 0, scale), memory)
        x = x + gate * self.FFW(modulate(self.norm2(x), 0, scale))
        return x


class FConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, fc=True):
        super(FConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        self.fre_network = FreNetwork(out_features)
        if not fc:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x):
        input_x = x              
        output = self.fre_network(self.weight_neigh(input_x))       
        if self.weight_self is not None:
            output += self.weight_self(x)  
        return output


class FreNetwork(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.complex_weight = nn.Parameter(torch.randn(1, 1, time_dim//2 + 1, 2, dtype=torch.float32) * 0.02)
        nn.init.xavier_uniform_(self.complex_weight)

    def forward(self, x):
        B, N, T = x.shape
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        t_bank= torch.view_as_complex(self.complex_weight)
        x = x * t_bank
        x = torch.fft.irfft(x, n=self.time_dim, dim=2, norm='ortho')
        return x
    
class DeviationGate(nn.Module):
    """
    Soft-Thresholding Deviation Gate
    g(δ) = δ * max(0, 1 - τ / ||δ||)
    """
    def __init__(self, dim, init_tau=0.1):
        super().__init__()
        # dead-zone threshold tau
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, delta):
        # compute L2 norm of feature shift
        norm = torch.norm(delta, dim=-1, keepdim=True)
        # ReLU naturally creates a dead-zone:
        # if norm <= tau, gate output is 0
        # if norm > tau, compute retention ratio
        gate = F.relu(norm - self.tau) / (norm + 1e-8)
        # scale back to original delta direction
        g_delta = delta * gate
        return g_delta, gate.mean()    
class STBP_Model(nn.Module):
    """Some Information about EAC_Model"""

    def __init__(self, args):
        super(STBP_Model, self).__init__()
        self.args = args
        self.year = args.year
        self.num_nodes = args.base_node_size
        self.dropout = args.dropout
        self.rank = args.rank 
        self.dropout = args.dropout
        self.in_dim = args.model["in_channel"]
        self.hidden_dim = args.model["hidden_channel"]
        self.backbone_out_dim = args.model["out_channel"]
        
        self.fconv1 = FConv(args.model["in_channel"], args.model["hidden_channel"], bias=True, fc=False)
        self.stmodule = Encoder(
            d_model=args.model["hidden_channel"],
            head=1,
            seq_length=1,
        )
        self.fconv2 = FConv(args.model["hidden_channel"], args.model["out_channel"], bias=True, fc=False)
        # self.fc = nn.Linear(args.model["out_channel"], args.y_len)
        self.fc = nn.Linear(self.backbone_out_dim * 2, self.backbone_out_dim)
        self.output = nn.Linear(self.backbone_out_dim, args.y_len)
        self.activation = nn.GELU()
        
        self.pattern_bank = nn.Parameter(torch.empty(args.base_node_size, args.model["hidden_channel"]*3).uniform_(-0.1, 0.1))
        
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

        # prototype bank
        self.prototypes = nn.Parameter(torch.randn(args.prototype_num, self.backbone_out_dim))
        self.Wq = nn.Linear(self.backbone_out_dim, self.backbone_out_dim)

        # running anchors: spatial [N, D], temporal [D]
        self.register_buffer("h_anchor_s", torch.zeros(self.num_nodes, self.backbone_out_dim)) # [N, D]
        self.register_buffer("h_anchor_t", torch.zeros(self.backbone_out_dim))                 # [D]

        # g(delta)
        self.gate_s = DeviationGate(self.backbone_out_dim, init_tau=args.tau_s)
        self.gate_t = DeviationGate(self.backbone_out_dim, init_tau=args.tau_t)
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

        # compute similarity between query and prototypes via dot product
        attn_score = torch.matmul(query, self.prototypes.t())
        attn_prob = torch.softmax(attn_score, dim=-1)

        # weighted prototype mix
        v = torch.matmul(attn_prob, self.prototypes) # [B, N, D]

        # top-2 prototypes for margin loss
        _, top2_idx = torch.topk(attn_score, k=2, dim=-1)
        pos = self.prototypes[top2_idx[:, :, 0]] # [B, N, D]
        neg = self.prototypes[top2_idx[:, :, 1]] # [B, N, D]

        return v, query, pos, neg
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.args.logger.info(f"Total Parameters: {total_params}")
        self.args.logger.info(f"Trainable Parameters: {trainable_params}")
    
    # def expand_adaptive_params(self, new_num_nodes):
    #     if new_num_nodes > self.num_nodes:
    #         new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.args.model["hidden_channel"]*3, dtype=self.pattern_bank.dtype, device=self.pattern_bank.device).uniform_(-0.1, 0.1))
    #         self.pattern_bank = nn.Parameter(torch.cat([self.pattern_bank, new_params], dim=0))
    #         self.num_nodes = new_num_nodes
    
    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes <= self.num_nodes:
            return
        new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.args.model["hidden_channel"]*3, dtype=self.pattern_bank.dtype, device=self.pattern_bank.device).uniform_(-0.1, 0.1))
        self.pattern_bank = nn.Parameter(torch.cat([self.pattern_bank, new_params], dim=0))
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

    def forward(self, data, adj):
        N = adj.shape[0]
        x = data.x.reshape((-1, N, self.args.model["in_channel"]))  
        B, N, T = x.shape
        x = F.relu(self.fconv1(x))                              

        adaptive_params = self.pattern_bank 
        
        node_emb = adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape).permute(0,2,1).unsqueeze(-1)  # [bs, N, feature]
        x = x.permute(0,2,1).unsqueeze(-1)

        x = self.stmodule(x, node_emb).squeeze(-1)
        x = x.permute(0,2,1).squeeze(-1)
        h_st = self.fconv2(x)

        # Low-rank core: X = U^T H V
        U = self.U_basis[:N, :]                    # [N, K]
        V = self.V_basis                           # [D, K]

        X = torch.einsum('nk,bnd,dk->bkd', U, h_st, V)

        # Prompt injection on bases
        U_tilde = U + self.P_s[:N, :]               # [N, K]
        V_tilde = V + self.P_t                     # [D, K]

        # Reconstruct: H~ = U~ X V~^T
        h_decoupled = torch.einsum('nk,bkd,dk->bnd', U_tilde, X, V_tilde)  # [B, N, D]

        v_cur, q_cur, p_cur, n_cur = self.query_prototypes(h_decoupled)

        # compute deviation from running anchors
        h_anchor_s = self.h_anchor_s[:N]
        delta_s = v_cur.mean(0) - h_anchor_s
        delta_t = v_cur.mean((0, 1)) - self.h_anchor_t

        # gating
        g_s, gate_s_act = self.gate_s(delta_s)
        g_t, gate_t_act = self.gate_t(delta_t)

        # update anchors during training
        if self.training:
            with torch.no_grad():
                # update spatial anchor
                h_anchor_s += g_s
                self.h_anchor_s[:N] = h_anchor_s
                # update temporal anchor
                self.h_anchor_t += g_t

            # unfreeze prompts when gate fires (||delta|| > tau)
            update_s = (gate_s_act > 0).item()
            update_t = (gate_t_act > 0).item()

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
            # margin loss on prototype retrieval
            margin = 0.5
            d_pos = torch.norm(q_cur - p_cur, dim=-1)
            d_neg = torch.norm(q_cur - n_cur, dim=-1)
            l_con = torch.mean(F.relu(d_pos - d_neg + margin))

            # subsample nodes to avoid O(N^2) pairwise distances
            sample_num = min(10, h_decoupled.size(1))
            sample_idx = torch.randperm(h_decoupled.size(1))[:sample_num]

            h_sample = h_decoupled[:, sample_idx, :]
            v_sample = v_cur[:, sample_idx, :]

            h_flat = h_sample.mean(0)   # [sample, D]
            v_flat = v_sample.mean(0)

            dist_h = torch.cdist(h_flat, h_flat)
            dist_v = torch.cdist(v_flat, v_flat)

            l_proto = F.mse_loss(dist_h, dist_v)

            self.aux_loss = l_con + l_proto

        interaction = h_st * (h_decoupled + v_cur) # interaction after prototype shift
        x_fusion = torch.cat([h_st, interaction], dim=-1) # [B, N, Out * 2]
        

        x_out = self.fc(self.activation(x_fusion))
        x_out = x_out.reshape(-1, self.args.y_len)
        if x_out.shape == data.x.shape:
            x_out = x_out + data.x  # residual connection
        x = self.output(self.activation(x_out))

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.training and hasattr(self.args, "dev_logger"):
            self.args.dev_logger.info(
                "Update_Stat",
                extra={
                    "year": self.args.year,
                    "epoch": getattr(self.args, "epoch", 0),
                    "Dt": delta_t.norm().item(),
                    "Ds": delta_s.norm().item(),
                    "G_t": gate_t_act,
                    "G_s": gate_s_act,
                    "tau_t": self.gate_t.tau.item(),
                    "tau_s": self.gate_s.tau.item(),
                    "mode": self.current_mode
                }
            )

        return x
    
