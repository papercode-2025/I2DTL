import torch
import torch.nn as nn
import torch.nn.functional as F
from HGCN import hgnn
from utils import contrastive_loss

class cross_modal_hypergraph(nn.Module):
    def __init__(self, args):
        super(cross_modal_hypergraph, self).__init__()
        self.args = args
        self.act_fn = nn.ReLU()
        self.args = args
        self.fc=nn.Linear(args.dim, args.graph_dim)
        self.hgcn = hgnn(args, args.graph_dim)
        self.memory = nn.Embedding(self.args.mem_dim, self.args.graph_dim)

    def forward(self, t, a, v, dia_len):
        num_dia = len(dia_len)
        hyperedge_index, hyperedge_type, features = self.create_hypergraph(t, a, v, dia_len, self.args.top_k)
        features = self.fc(features)
        out, _ = self.hgcn(features, hyperedge_index, hyperedge_type)
        out, loss_cl = self.cross_modal_contrast(out, dia_len)
        return out, loss_cl   
    
    def create_hypergraph(self, t, a, v, dia_len, param_top_k):
        node_count = 0
        edge_count = 0
        index1 = []
        index2 = []
        index_temp = []
        hyperedge_type = []
        for i in dia_len:
            top_k = int(i * param_top_k)
            len_temp = []
            if node_count == 0:
                t_temp = t[:i]
                a_temp = a[:i]
                v_temp = v[:i]
                feature_temp = torch.cat((t_temp, a_temp, v_temp), dim=0)
                features = torch.cat((t_temp, a_temp, v_temp), dim=0)
                temp = 0 + i
            else:
                t_temp = t[temp:temp + i]
                a_temp = a[temp:temp + i]
                v_temp = v[temp:temp + i]
                feature_temp = torch.cat((t_temp, a_temp, v_temp), dim=0)
                features = torch.cat((features, feature_temp), dim=0)
                temp = temp + i
            
            distance = torch.cdist(feature_temp, feature_temp) 
            _, tk_idx = distance.topk(top_k, dim=1, largest=False) 
            tk_idx = tk_idx + node_count

            nodes = list(range(i * 3))
            nodes = [j + node_count for j in nodes] 
            nodes_t = nodes[0:i * 3 // 3]  
            nodes_a = nodes[i * 3 // 3:i * 3 * 2 // 3] 
            nodes_v = nodes[i * 3 * 2 // 3:]

            for j in range(3 * i):
                index_temp = list(set(tk_idx[j].tolist() + [nodes_t[j//3]] + [nodes_a[j//3]] + [nodes_v[j//3]]))
                len_temp.append(len(index_temp))
                index1 = index1 + index_temp
            for k in range(3 * i):
                index2 = index2 + [edge_count] * len_temp[k]
                edge_count = edge_count + 1
            hyperedge_type = hyperedge_type + [0] * i + [1] * i + [2] * i
            node_count = node_count + 3 * i
        index1 = torch.LongTensor(index1).view(1, -1)
        index2 = torch.LongTensor(index2).view(1, -1)
        hyperedge_index = torch.cat([index1, index2], dim=0).cuda()
        hyperedge_type = torch.LongTensor(hyperedge_type).view(-1, 1).cuda()
        return hyperedge_index, hyperedge_type, features

    def cross_modal_contrast(self, feature, dia_len):
        t, a, v = self.reverse_features(dia_len, feature)
        out = torch.cat((t, a, v), dim=1)
        loss = 0
        loss = loss + contrastive_loss(t, a)
        loss = loss + contrastive_loss(t, v)
        loss = loss + contrastive_loss(a, v)
        loss = loss / 3
        return out, loss
    
    def reverse_features(self, dia_len, features):
        t = []
        a = []
        v = []
        for i in dia_len:
            tt = features[0:1 * i]
            aa = features[1 * i:2 * i]
            vv = features[2 * i:3 * i]
            features = features[3 * i:]
            t.append(tt)
            a.append(aa)
            v.append(vv)
        tmpt = torch.cat(t, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        return tmpt, tmpa, tmpv
