import torch
import torch.nn as nn
import torch.nn.functional as F

import zutils
from utils.graph_utils import mask_adjs, mask_x
# import src.utils as utils
from .conditions import TimestepEmbedder
from .layers import Attention, Mlp, MultiHeadAttention, AddNorm, PositionWiseFFN
# from models.conditions import TimestepEmbedder, CategoricalEmbedder, ClusterContinuousEmbedder
from vector_quantize_pytorch import VectorQuantize

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class SElayer(nn.Module):
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

    def forward(self, x, c,node_mask):
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





class DiffEncoder(nn.Module):
    def __init__(
            self,
            max_n_nodes,
            hidden_size=384,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            drop_condition=0.1,
            Xdim=118,
            Edim=5,
            ydim=3,
            task_type='regression',
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        self.x_embedder = nn.Linear(Xdim + max_n_nodes*Edim, hidden_size, bias=False)
        self.hidden_size=hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.norm=nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.encoders = nn.ModuleList(
            [
                NDRSA_block(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)


            ]
        )
        # self.mean_layer = nn.Linear(hidden_size, hidden_size)  # Mean of latent space
        # self.logvar_layer = nn.Linear(hidden_size, hidden_size)

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



    def process(self,x,e):




        bs, n, _ = x.size()
        x = torch.cat([x, e.reshape(bs, n, -1)], dim=-1)
        # x = torch.cat([x,e], dim=-1)
        # self.x_embedder = nn.Linear(x.size(-1), self.hidden_size, bias=False)
        x = self.x_embedder(x)
        return  x

    def forward(self, x, e, node_mask):
        x_in, e_in = x, e
        z=self.process(x,e)


        for i, block in enumerate(self.encoders):
            z = block(z, node_mask)
        # z=z*node_mask.unsqueeze(-1)
        return zutils.zPlaceHolder(X=x_in, E=e_in, z=z).mask(node_mask)
        # return z



class DiffDecoder(nn.Module):
    def __init__(
            self,
            max_n_nodes,
            hidden_size=384,
            depth=12,
            num_heads=8,
            mlp_ratio=4.0,
            drop_condition=0.1,
            Xdim=118,
            Edim=5,
            ydim=3,
            task_type='regression',
    ):
        super().__init__()
        self.num_heads = num_heads
        self.ydim = ydim
        self.Xdim=Xdim
        self.Edim=Edim
        self.hidden_size=hidden_size
        self.x_embedder = nn.Linear(hidden_size,Xdim + max_n_nodes*Edim, bias=True)
        self.norm_final = nn.LayerNorm(Xdim + max_n_nodes*Edim, elementwise_affine=False)
        self.decoder = Decoder(
            max_n_nodes=max_n_nodes,
            hidden_size=hidden_size,
            atom_type=Xdim,
            bond_type=Edim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
        )

        # self.initialize_weights()

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
        _constant_init(self.decoder.adaLN_modulation[0], 0)



    def forward(self,z, node_mask):


        X, E = self.decoder(z, node_mask)



        return zutils.PlaceHolder(X=X, E=E).mask(node_mask)


        # X, E, y = self.decoder(z, x_in, e_in, node_mask)
        # return X,E


class EncoderLayer(nn.Module):
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

    def forward(self, x, node_mask):

        x = x + self.norm1(self.attn(x, node_mask=node_mask))
        x = x + self.norm2(self.mlp(x))


        return x




def make_cdist_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """Make mask for coordinate pairwise distance from padding mask.

    Args:
        padding_mask (torch.Tensor): Padding mask for the batched input sequences with shape (b, l).

    Returns:
        torch.Tensor: Mask for coordinate pairwise distance with shape (b, l, l).
    """
    padding_mask = padding_mask.unsqueeze(-1)
    mask = padding_mask * padding_mask.transpose(-1, -2)
    return mask
def compute_distance_residual_bias(cdist: torch.Tensor, cdist_mask: torch.Tensor, raw_max: bool = True) -> torch.Tensor:
    b, l, _ = cdist.shape
    D = cdist * cdist_mask
    if not raw_max:
        D_max, _ = torch.max(D.view(b, -1), dim=-1)  # max value of each sample
        D = D_max.view(b, 1, 1) - D  # sample-max value subtract every value
    else:
        D_max, _ = torch.max(D, dim=-1)  # max value of every raw in each sample
        D = D_max.view(b, l, 1) - D  # raw-max value subtract every raw
    D.diagonal(dim1=-2, dim2=-1)[:] = 0  # set diagonal to 0
    D = D * cdist_mask
    return D
class NDRSA_block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.multi_attention = MultiHeadAttention(
            d_q=hidden_size,
            d_k=hidden_size,
            d_v=hidden_size,
            d_model=hidden_size,
            n_head=num_heads,
            qkv_bias=True,
            attn_drop=0.1,

        )
        self.add_norm01 = AddNorm(norm_shape=hidden_size, dropout=0.1,
                                  pre_ln=True)
        self.position_wise_ffn = PositionWiseFFN(
            d_in=hidden_size,
            d_hidden=hidden_size*4,
            d_out=hidden_size,
            dropout=0.1,
        )
        self.add_norm02 = AddNorm(norm_shape=hidden_size, dropout=0.1,
                                  pre_ln=True)

    def forward(self, x, node_mask):
        D, D_M = torch.cdist(x, x), make_cdist_mask(node_mask)
        D = compute_distance_residual_bias(cdist=D, cdist_mask=D_M)  # masked and row-max subtracted distance matrix
        # D = D * D_M  # for ablation study

        attn_out = self.multi_attention(x, x, x, distance_matrix=D)
        Y, attn_weight = attn_out["out"], attn_out["attn_weight"]
        X = self.add_norm01(x, Y)
        Y = self.position_wise_ffn(X)
        X = self.add_norm02(X, Y)


        return X

class Decoder(nn.Module):
    # Structure Decoder
    def __init__(self, max_n_nodes, hidden_size, atom_type, bond_type, mlp_ratio, num_heads=None):
        super().__init__()
        self.atom_type = atom_type
        self.hidden_size=hidden_size
        self.bond_type = bond_type
        final_size = atom_type + max_n_nodes * bond_type
        self.xedecoder = Mlp(in_features=hidden_size,hidden_features=1024,
                             out_features=512, drop=0)
        self.decoders = nn.ModuleList(
            [
                NDRSA_block(512, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(2)

            ]
        )
        self.ndout = Mlp(in_features=512, hidden_features=1024,
                             out_features=hidden_size, drop=0)


        self.atomdecoder=Mlp(in_features=hidden_size//2,hidden_features=1024,
                             out_features=atom_type, drop=0)
        self.bonddecoder = Mlp(in_features=hidden_size // 2,hidden_features=1024,
                               out_features=max_n_nodes*bond_type, drop=0)
        self.atom_norm_final = nn.LayerNorm(atom_type, elementwise_affine=False)
        self.bond_norm_final = nn.LayerNorm(max_n_nodes * bond_type, elementwise_affine=False)
        self.norm_final = nn.LayerNorm(final_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * final_size, bias=True)
        )

    def forward(self, x , node_mask):
        x_all = self.xedecoder(x)
        for i, block in enumerate(self.decoders):
            x_all = block(x_all, node_mask)
        B, N, D = x_all.size()
        x_out=self.ndout(x_all)
        # x_all=self.norm_final(x_all)

        atom_embedding=self.atomdecoder(x_out[:,:,:self.hidden_size//2])
        atom_out=self.atom_norm_final(atom_embedding)
        bond_embedding = self.bonddecoder(x_out[:, :, self.hidden_size // 2:])
        bond_out=self.bond_norm_final(bond_embedding).reshape(B, N, N, self.bond_type)
        # bond_out=bond_embedding



        # bond_out = x_all[:, :, self.atom_type:].reshape(B, N, N, self.bond_type)


        #### standardize adj_out
        edge_mask = (~node_mask)[:, :, None] & (~node_mask)[:, None, :]
        diag_mask = (
            torch.eye(N, dtype=torch.bool)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .type_as(edge_mask)
        )
        bond_out.masked_fill_(edge_mask[:, :, :, None], 0)
        bond_out.masked_fill_(diag_mask[:, :, :, None], 0)

        bond_out = 1 / 2 * (bond_out + torch.transpose(bond_out, 1, 2))

        # bond_out=mask_adjs(bond_out,node_mask)
        # atom_out=mask_x(atom_out,node_mask)
        return atom_out, bond_out
