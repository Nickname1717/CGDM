"""
    This module contains different attention modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.model import utils


class SelfAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop_out = nn.Dropout(dropout)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attention_mask: torch.Tensor = None, attention_bias: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute (multi-head) Self-Attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

        Args:
            - Q (torch.Tensor): Query, shape: (b, l, d) or (b, h, l, d)
            - K (torch.Tensor): Key, shape: (b, l, d) or (b, h, l, d)
            - V (torch.Tensor): Value, shape: (b, l, d) or (b, h, l, d)
            - attention_mask (torch.Tensor, optional): Attention mask, shape: (b, l) or (b, l, l), 1 for valid, 0 for invalid. Defaults to None.
            - attention_bias (torch.Tensor, optional): Attention bias, shape: (b, l, l) or (b, h, l, l). Defaults to None.

        Returns:
            torch.Tensor: Weighted sum of value, shape: (b, l, d) or (b, h, l, d)
        """
        scale = Q.shape[-1] ** 0.5
        # Q @ K.mT <==> torch.matmul(Q, K.mT)
        attention_score = (Q @ K.mT) / torch.tensor(scale)  # (b, l, l) | (b, h, l, l)
        if attention_bias is not None:
            attention_score += attention_bias
        if attention_mask is not None:
            attention_score = utils.mask_attention_score(attention_score, attention_mask)
        attention_weight = F.softmax(attention_score, dim=-1)  # (b, l, l) | (b, h, l, l)
        return self.drop_out(attention_weight) @ V  # (b, l, d) | (b, h, l, d)


class MSRSA(nn.Module):
    def __init__(self, num_heads: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads


        self.drop_out = nn.Dropout(dropout)
        self.weight_D = None, None


        self.weight_D = torch.randn(num_heads).view(1, num_heads, 1, 1)
        self.weight_D = nn.Parameter(self.weight_D, requires_grad=True)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor = None,
        row_subtracted_distance_matrix: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute (multi-head) Self-Attention || b: batch_size, l: seq_len, d: hidden_dims, h: num_heads

        Args:
            - Q (torch.Tensor): Query, shape: (b, l, d) or (b, h, l, d)
            - K (torch.Tensor): Key, shape: (b, l, d) or (b, h, l, d)
            - V (torch.Tensor): Value, shape: (b, l, d) or (b, h, l, d)
            - attention_mask (torch.Tensor): Attention mask, shape: (b, l) or (b, l, l), 1 for valid, 0 for invalid
            - adjacency_matrix (torch.Tensor): Adjacency matrix, shape: (b, l, l)
            - row_subtracted_distance_matrix (torch.Tensor): Subtracted distance matrix (every raw is subtracted by raw-max value), shape: (b, l, l)

        Returns:
            torch.Tensor: Weighted sum of value, shape: (b, l, d) or (b, h, l, d)
        """
        M, D_s = attention_mask, row_subtracted_distance_matrix


        D_s = D_s.unsqueeze(1)
        scale = Q.shape[-1] ** 0.5
        attn_score = Q @ K.mT  # (b, l, l) | (b, h, l, l)
        attn_score = utils.mask_attention_score(attn_score, M, 0.0) if M is not None else attn_score

        B_D = attn_score * (D_s * self.weight_D)

        attn_score = attn_score + B_D if B_D is not None else attn_score
        attn_score = attn_score / torch.tensor(scale)  # (b, h, l, l) scaled by sqrt(d) after adding residual terms
        attention_weight = F.softmax(attn_score, dim=-1)  # (b, l, l) | (b, h, l, l)
        return {
            "out": self.drop_out(attention_weight) @ V,  # (b, l, d) | (b, h, l, d)
            "attn_weight": attention_weight.detach(),  # (b, l, l) | (b, h, l, l)
        }
