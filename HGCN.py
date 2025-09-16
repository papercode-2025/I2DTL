import torch.nn as nn
import torch
from torch_geometric.nn.conv import MessagePassing,HypergraphConv
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F

class hgnn(nn.Module):
    def __init__(self,args,n_dim):
        super(hgnn, self).__init__()
        self.args=args
        self.hyperedge_weight = nn.Parameter(torch.ones(10000))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(n_dim))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(n_dim))
        self.hyperedge_attr3 = nn.Parameter(torch.rand(n_dim))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperconv = HyConv(n_dim,n_dim)

    def forward(self, features, hyperedge_index, hyperedge_type, bi_weight=None):
        self.hyperedge_attr1.data = self.hyperedge_attr1.data.to(features.device)
        self.hyperedge_attr2.data = self.hyperedge_attr2.data.to(features.device)
        self.hyperedge_attr3.data = self.hyperedge_attr3.data.to(features.device)
        self.hyperedge_weight.data = self.hyperedge_weight.data.to(features.device)
        self.EW_weight.data = self.EW_weight.data.to(features.device)

        hyperedge_attr = torch.stack([
            self.hyperedge_attr1, 
            self.hyperedge_attr2, 
            self.hyperedge_attr3  
        ], dim=0)[hyperedge_type.long()]
        hyperedge_weight = self.hyperedge_weight[0:hyperedge_index[1].max().item() + 1]
        norm = self.EW_weight[0:hyperedge_index.size(1)]
        if bi_weight is not None:
            norm = bi_weight
        out, hy = self.hyperconv(features, hyperedge_index, hyperedge_weight, hyperedge_attr, norm)
        return out, hy



class HyConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HyConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        
        self.heads = 1
        self.concat = True
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edgeweight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edgefc = torch.nn.Linear(in_channels, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edgeweight)
        zeros(self.bias)

    def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None, EW_weight=None ,dia_len=None):

        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
        alpha = None

        D = scatter_add(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=num_nodes)  
        D = 1.0 / D  
        D[D == float("inf")] = 0

        if EW_weight is None:
            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)      
        else:
            B = scatter_add(EW_weight[hyperedge_index[0]],
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,  
                             size=(num_nodes, num_edges))  
        out = out.view(num_edges, -1)
        hyperedge_attr=out

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_nodes, num_edges))  
        if self.concat is True and out.size(1) == 1:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias.to(out.device))
            out = out + self.bias

        return torch.nn.LeakyReLU()(out) ,torch.nn.LeakyReLU()(hyperedge_attr.view(num_edges, -1)) #

    def message(self, x_j, norm_i, alpha):
        H, F = self.heads, self.out_channels

        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        else:
            out = norm_i.view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

