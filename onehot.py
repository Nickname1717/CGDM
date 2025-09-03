import torch


import torch
from rdkit import Chem


def one_hot_encode_adj(adj_matrix, n):
    # Adjust the adjacency matrix by subtracting 1
    # adj_matrix_adj = adj_matrix - 1

    # Batch size
    batch_size = adj_matrix.size(0)

    # Initialize the one-hot encoded tensor
    one_hot_adj = torch.zeros(batch_size, adj_matrix.size(1), adj_matrix.size(2), n, device=adj_matrix.device)

    # One-hot encode each edge type, now starting from 0
    for edge_type in range(n):
        one_hot_adj[:, :, :, edge_type] = (adj_matrix == edge_type).float()

    return one_hot_adj


def convert_to_onehot(x):
    """
    将输入张量转换为 one-hot 编码，最大的值为 1，其余为 0。

    参数:
    x (torch.Tensor): 输入张量，其形状为 (batch_size, num_nodes, num_features)。

    返回:
    torch.Tensor: one-hot 编码后的张量，其形状为 (batch_size, num_nodes, num_features)。
    """
    # 找到每个位置上最大值的索引
    max_indices = torch.argmax(x, dim=-1, keepdim=True)
    # 创建一个全零张量，与 x 形状相同
    onehot = torch.zeros_like(x)
    # 将最大值的索引位置设为 1
    onehot.scatter_(-1, max_indices, 1)

    return onehot


def validate_smiles(smiles_list):
    valid_smiles = []
    invalid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
        else:
            invalid_smiles.append(smi)

    return valid_smiles, invalid_smiles
def decode_one_hot(one_hot_tensor):
    # Get the indices of the maximum values along the third dimension
    adj_matrix = torch.argmax(one_hot_tensor, dim=2).float()

    # Adjust the matrix by setting the all-zero entries to -1
    zero_mask = torch.all(one_hot_tensor == 0, dim=2)
    adj_matrix[zero_mask] = -1

    adj_matrix += 1

    return adj_matrix


def create_node_mask(x):
    """
    创建一个布尔掩码，指示每个节点是否不等于 [0, 0, 0, 0].

    参数:
    x (torch.Tensor): 输入张量，其形状为 (..., 4).

    返回:
    torch.Tensor: 一个布尔掩码，其形状为与 x 的前 n-1 维相同.
    """
    # 创建一个与 x 形状相同的全零数组
    zero_feature = torch.zeros_like(x)
    # 比较 x 是否等于 zero_feature，得到一个布尔值掩码
    mask = torch.all(x == zero_feature, dim=-1)
    # 取反，得到每个节点是否不等于 [0, 0, 0, 0] 的布尔值
    node_mask = ~mask

    return node_mask


def update_adj_matrix(adj, node_mask):
    """
    更新邻接矩阵，将与 node_mask 为 False 的节点相关的行和列的值设为 -1。

    参数:
    adj (torch.Tensor): 邻接矩阵，其形状为 (batch_size, num_nodes, num_nodes)。
    node_mask (torch.Tensor): 节点掩码，其形状为 (batch_size, num_nodes)，布尔值表示每个节点是否有效。

    返回:
    torch.Tensor: 更新后的邻接矩阵，其形状为 (batch_size, num_nodes, num_nodes)。
    """
    # 创建一个与 adj 形状相同的布尔掩码矩阵，初始值为 True
    adj_mask = torch.ones_like(adj, dtype=torch.bool)

    # 将 node_mask 为 False 的行和列置为 False
    # 广播 node_mask 并进行掩码操作
    node_mask_expanded = node_mask.unsqueeze(2)  # shape: (batch_size, num_nodes, 1)
    adj_mask &= node_mask_expanded
    adj_mask &= node_mask_expanded.transpose(1, 2)
    adj_mask &= ~torch.diag_embed(torch.ones_like(node_mask, dtype=torch.bool))
    # 更新邻接矩阵，将掩码矩阵中为 False 的位置的值设为 -1
    adj_update = adj.masked_fill(~adj_mask, -1)

    return adj_update

