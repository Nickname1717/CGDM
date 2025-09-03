import torch
import torch.nn as nn

import utils
from TUD.layers import Attention, Mlp
from TUD.conditions import TimestepEmbedder, CategoricalEmbedder, ClusterContinuousEmbedder

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


import torch
import torch.nn.functional as F
import torch

import torch
import torch.nn.functional as F
from torch import nn


def fps_downsample(x, node_mask, num_samples):
    """
    Downsample the point cloud using Farthest Point Sampling (FPS).
    Args:
        x (torch.Tensor): Input feature tensor of shape (B, N, D) where N is the number of nodes.
        node_mask (torch.Tensor): Mask indicating valid nodes (B, N).
        num_samples (int): The number of nodes to sample.
    Returns:
        torch.Tensor: Downsampled features (B, num_samples, D).
    """
    B, N, D = x.size()
    # Mask out invalid nodes
    valid_nodes = x * node_mask.unsqueeze(-1)

    # Compute the distance matrix
    dist = torch.norm(valid_nodes[:, :, None, :] - valid_nodes[:, None, :, :], dim=-1)
    dist = dist * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # Apply node mask to distances
    dist.fill_diagonal_(float('inf'))  # Ignore self-distance

    # FPS: Select the farthest points
    selected_idx = []
    selected_idx.append(torch.zeros(B, dtype=torch.long))  # Always start with the first point
    for b in range(B):
        selected = [selected_idx[-1]]
        remaining = list(range(N))
        remaining.remove(selected_idx[-1])

        for _ in range(num_samples - 1):
            dist_to_selected = dist[b, remaining, selected[-1]]
            farthest_point_idx = remaining[torch.argmax(dist_to_selected)]
            selected.append(farthest_point_idx)
            remaining.remove(farthest_point_idx)
        selected_idx.append(selected)

    selected_idx = torch.stack(selected_idx, dim=0).to(x.device)
    downsampled_x = torch.gather(x, 1, selected_idx.unsqueeze(-1).expand(-1, -1, D))
    return downsampled_x


def fps_upsample(x, node_mask, num_samples):
    """
    Upsample the point cloud using Farthest Point Sampling (FPS).
    Args:
        x (torch.Tensor): Input feature tensor of shape (B, N, D) where N is the number of nodes.
        node_mask (torch.Tensor): Mask indicating valid nodes (B, N).
        num_samples (int): The number of nodes to upsample to.
    Returns:
        torch.Tensor: Upsampled features (B, num_samples, D).
    """
    B, N, D = x.size()

    # If the number of samples is greater than current nodes, perform an increase in sampling
    if num_samples > N:
        # First, perform FPS downsampling to get initial farthest points
        downsampled_x = fps_downsample(x, node_mask, N)

        # Here you can apply some interpolation technique (e.g., using k-NN to fill intermediate points)
        # For simplicity, this part can be implemented with a trivial interpolation method.

        upsampled_x = downsampled_x
        # Example: copy the features (simplified)
        for _ in range(num_samples - N):
            upsampled_x = torch.cat([upsampled_x, upsampled_x[:, :1, :]], dim=1)
        return upsampled_x

    # Otherwise, return original tensor for now (no upsampling needed)
    return x


class TUD(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=384,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        drop_condition=0.1,
        Xdim=118,
        Edim=5,
        ydim=3,
        task_type='regression',
            use_fps=True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_fps=use_fps
        # self.ydim = ydim
        self.x_embedder = nn.Linear(input_size, hidden_size, bias=False)

        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedding_list = torch.nn.ModuleList()
        #
        # self.y_embedding_list.append(ClusterContinuousEmbedder(2, hidden_size, drop_condition))
        # for i in range(ydim - 2):
        #     if task_type == 'regression':
        #         self.y_embedding_list.append(ClusterContinuousEmbedder(1, hidden_size, drop_condition))
        #     else:
        #         self.y_embedding_list.append(CategoricalEmbedder(2, hidden_size, drop_condition))

        self.encoders = nn.ModuleList(
            [
                SELayer(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.downsample_times=2

        self.out_layer = OutLayer(
            final_size=input_size,
            hidden_size=hidden_size,
            atom_type=Xdim,
            bond_type=Edim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)

        for block in self.encoders :
            _constant_init(block.adaLN_modulation[0], 0)
        _constant_init(self.out_layer.adaLN_modulation[0], 0)

    def forward(self, x, t, node_mask):
        
        # force_drop_id = torch.zeros_like(y.sum(-1))
        # force_drop_id[torch.isnan(y.sum(-1))] = 1
        # if unconditioned:
        #     force_drop_id = torch.ones_like(y[:, 0])
        
        # x_in, e_in, y_in = x, e, y
        # bs, n, _ = x.size()
        # x = torch.cat([x, e.reshape(bs, n, -1)], dim=-1)
        x = self.x_embedder(x)


        c1 = self.t_embedder(t)

        if self.use_fps and self.num_samples and x.size(1) > self.num_samples:
            x = fps_downsample(x, node_mask, self.num_samples)
        # for i in range(1, self.ydim):
        #     if i == 1:
        #         c2 = self.y_embedding_list[i-1](y[:, :2], self.training, force_drop_id, t)
        #     else:
        #         c2 = c2 + self.y_embedding_list[i-1](y[:, i:i+1], self.training, force_drop_id, t)
        c = c1
        for i, block in enumerate(self.encoders):
            x = block(x, c, node_mask)


        # X: B * N * dx, E: B * N * N * de
        x = self.out_layer(x, c,node_mask)
        x = x * node_mask.unsqueeze(-1)
        return x


class SELayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.dropout = 0.
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, **block_kwargs
        )

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=self.dropout,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c, node_mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * modulate(self.norm1(self.attn(x, node_mask=node_mask)), shift_msa, scale_msa)
        x = x + gate_mlp.unsqueeze(1) * modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x


class OutLayer(nn.Module):
    # Structure Output Layer
    def __init__(self,final_size, hidden_size, atom_type, bond_type, mlp_ratio, num_heads=None):
        super().__init__()
        self.atom_type = atom_type
        self.bond_type = bond_type
        # final_size = atom_type + max_n_nodes * bond_type
        self.xedecoder = Mlp(in_features=hidden_size, 
                            out_features=final_size, drop=0)

        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x,  c, node_mask):
        x_all = self.xedecoder(x)
        B, N, D = x_all.size()
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_all = modulate(self.norm_final(x_all), shift, scale)
        


        return x_all
