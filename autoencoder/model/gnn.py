import torch
import torch.nn.functional as F

from models.layers import DenseGCNConv, MLP
from utils.graph_utils import mask_x, pow_tensor
from models.attention import  AttentionLayer

from utils.graph_utils import mask_adjs
class GNNencoder(torch.nn.Module):

    def __init__(self, max_feat_num, depth, nhid):

        super(GNNencoder, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1024,
                            use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):

        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        # x=xs.view(*out_shape)
        x = mask_x(x, flags)

        return x

class GNNDecoder(torch.nn.Module):

    def __init__(self,nfeat, nhid,ain, output_feat_num, num_layers):
        super(GNNDecoder, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.output_feat_num = output_feat_num
        self.ain=ain
        self.num_layers = num_layers




        # Initialize GCN layers
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        # MLP to process predicted adjacency matrix A
        self.adj_mlp = MLP(num_layers=3, input_dim=self.ain, hidden_dim=512, output_dim=self.ain,
                           use_bn=False, activate_func=F.elu)

        # Final MLP to predict node features X
        self.final_mlp = MLP(num_layers=3, input_dim=self.nhid, hidden_dim=512, output_dim=self.output_feat_num,
                             use_bn=False, activate_func=F.elu)

        # Mask used for adjacency matrix
        self.mask = torch.ones([self.output_feat_num, self.output_feat_num]) - torch.eye(self.output_feat_num)
        self.mask.unsqueeze_(0)  # Add batch dimension

    def forward(self, z, flags=None):
        # Predict adjacency matrix A from z using matrix multiplication
        adj_pred = torch.matmul(z, z.transpose(-1, -2))  # B x N x N

        # Pass adj_pred through the MLP to refine A
        adj_shape = adj_pred.shape
        adj_pred_flat = adj_pred.view(adj_shape[0], -1)  # B x (N*N)

        # Pass flattened adjacency matrix through MLP
        adj_pred_flat = self.adj_mlp(adj_pred_flat)  # B x (N*N)           # Pass through MLP
        adj_pred = adj_pred_flat.view(adj_shape)          # Reshape back to B x N x N

        # Mask the adjacency matrix (optional depending on flags)
        adj_pred = mask_adjs(adj_pred, flags)

        # Pass z through GCN layers to refine node features
        x = z
        for layer in self.layers:
            x = layer(x, adj_pred)  # Use refined adj_pred
            x = F.elu(x)

        # Predict node features X
        x_pred = mask_x(self.final_mlp(x), flags)


        return  x_pred,adj_pred

class ScoreNetworkX_GMH(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
        super().__init__()

        self.depth = depth
        self.c_init = c_init

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init,
                                                  c_hid, num_heads, conv))
            elif _ == self.depth - 1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid,
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid,
                                                  c_hid, num_heads, conv))

        fdim = max_feat_num + depth * nhid
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2*fdim, output_dim=max_feat_num,
                         use_bn=False, activate_func=F.elu)

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        x_list = [x]
        for _ in range(self.depth):
            x, adjc = self.layers[_](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)
        x = mask_x(x, flags)

        return x
