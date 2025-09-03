from torch.jit import Final
import torch.nn.functional as F
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn

import copy
import os.path as osp
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops

from autoencoder.model import attention


class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME
        assert self.fast_attn, "scaled_dot_product_attention Not implemented"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def dot_product_attention(self, q, k, v):
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn_sfmx = attn.softmax(dim=-1)
        attn_sfmx = self.attn_drop(attn_sfmx)
        x = attn_sfmx @ v
        return x, attn

    def forward(self, x, node_mask):
        B, N, D = x.shape

        # B, head, N, head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, head, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = (node_mask[:, None, :, None] & node_mask[:, None, None, :]).expand(-1, self.num_heads, N, N)
        attn_mask[attn_mask.sum(-1) == 0] = True

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
            attn_mask=attn_mask,
        )

        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MolGNN(torch.nn.Module):
    def __init__(self,dim,Xdim,outdim):
        super().__init__()
        self.lin0 = torch.nn.Linear(Xdim, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, outdim)

    def forward(self, x,edge_index,edge_attr):
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)
        
        for i in range(3):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_q: int = None,
        d_k: int = None,
        d_v: int = None,
        d_model: int = None,
        n_head: int = 1,
        qkv_bias: bool = False,
        attn_drop: float = 0,
    ) -> None:
        super().__init__()
        self.num_heads = n_head
        self.hidden_dims = d_model
        self.attention = attention.MSRSA(num_heads=n_head, dropout=attn_drop)
        assert d_q is not None and d_k is not None and d_v is not None and d_model is not None, "Please specify the dimensions of Q, K, V and d_model"
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        # q: q_dims, k: k_dims, v: v_dims, d: hidden_dims, h: num_heads, d_i: dims of each head
        self.W_q = nn.Linear(d_q, d_model, bias=qkv_bias)  # (q, h*d_i=d)
        self.W_k = nn.Linear(d_k, d_model, bias=qkv_bias)  # (k, h*d_i=d)
        self.W_v = nn.Linear(d_v, d_model, bias=qkv_bias)  # (v, h*d_i=d)
        self.W_o = nn.Linear(d_model, d_model, bias=qkv_bias)  # (h*d_i=d, d)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor = None,

        distance_matrix: torch.Tensor = None,
    ):
        """Compute multi-head attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

        Args:
            - queries (torch.Tensor): Query, shape: (b, l, q)
            - keys (torch.Tensor): Key, shape: (b, l, k)
            - values (torch.Tensor): Value, shape: (b, l, v)
            - attention_mask (torch.Tensor, optional): Attention mask, shape: (b, l) or (b, l, l). Defaults to None.
            - adjacency_matrix (torch.Tensor, optional): Adjacency matrix, shape: (b, l, l). Defaults to None.
            - distance_matrix (torch.Tensor, optional): Distance matrix, shape: (b, l, l). Defaults to None.
        Returns:
            torch.Tensor: Output after multi-head attention pooling with shape (b, l, d)
        """
        # b: batch_size, h:num_heads, l: seq_len, d: d_hidden
        b, h, l, d_i = queries.shape[0], self.num_heads, queries.shape[1], self.hidden_dims // self.num_heads
        Q, K, V = self.W_q(queries), self.W_k(keys), self.W_v(values)  # (b, l, h*d_i=d)
        Q, K, V = [M.view(b, l, h, d_i).permute(0, 2, 1, 3) for M in (Q, K, V)]  # (b, h, l, d_i)
        attn_out = self.attention(Q, K, V, attention_mask, distance_matrix)
        out, attn_weight = attn_out["out"], attn_out["attn_weight"]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, l, h * d_i)  # (b, l, h*d_i=d)
        # return self.W_o(out)  # (b, l, d)
        return {
            "out": self.W_o(out),  # (b, l, d)
            "attn_weight": attn_weight,  # (b, l, l) | (b, h, l, l)
        }
class PositionWiseFFN(nn.Module):
    def __init__(self, d_in: int = None, d_hidden: int = None, d_out: int = None, dropout: float = 0) -> None:
        super().__init__()
        assert d_in is not None and d_hidden is not None and d_out is not None, "d_in, d_hidden, d_out must be specified."
        self.ffn01 = nn.Linear(d_in, d_hidden)
        self.ffn02 = nn.Linear(d_hidden, d_out)
        # self.active = nn.ReLU()
        self.active = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor):
        return self.ffn02(self.dropout(self.active(self.ffn01(X))))
class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return X + Y
class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout: float = 0, pre_ln: bool = True) -> None:
        """Residual connection and LayerNorm Module.

        Args:
            - norm_shape (Union[Sequence[int], int]): reference to LayerNorm.
            - dropout (float, optional): dropout rate. Defaults to 0.
            - pre_ln (bool, optional): whether to apply LayerNorm before residual connection. Defaults to True.
        """
        super().__init__()
        self.pre_ln = pre_ln
        self.residual = Residual()
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        if self.pre_ln:
            # layer norm before residual
            X = self.residual(X, self.layer_norm(self.dropout(Y)))
        else:
            # layer norm after residual
            X = self.residual(X, self.dropout(Y))
            X = self.layer_norm(X)
        return X



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


