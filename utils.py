import torch,numpy,random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, -1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, log_prob, labels):
       
        class_weights_expanded = self.class_weights.view(1, -1).expand(labels.size(0), -1)
        
        loss = -torch.sum(class_weights_expanded * labels * log_prob) / log_prob.size(0)  
        return loss
    
def simple_batch_graphify(features, lengths):
    node_features = []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
    node_features = torch.cat(node_features, dim=0)
    node_features = node_features.cuda()
    return node_features

seed = 42
def seed_everything(seed=seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    numpy.random.seed(int(seed) + worker_id)

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, T):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / T)
    indices = np.arange(0, num_nodes)
    np.random.shuffle(indices)
    i = 0
    mask = indices[i * batch_size:(i + 1) * batch_size]
    refl_sim = f(sim(z1[mask], z1))  # [B, N]
    between_sim = f(sim(z1[mask], z2))  # [B, N]
    loss = -torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                      / (refl_sim.sum(1) + between_sim.sum(1)
                         - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag()))

    return loss
def com_semi_loss(z1: torch.Tensor, z2: torch.Tensor, T, com_nodes1, com_nodes2):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim[com_nodes1, com_nodes2] / (
                refl_sim.sum(1)[com_nodes1] + between_sim.sum(1)[com_nodes1] - refl_sim.diag()[com_nodes1]))

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, T):
    f = lambda x: torch.exp(x / T)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def contrastive_loss(x1, x2, tau=0.5, top_k=5, num_hard_neg=3, num_neighbors=5):

    x1 = F.normalize(x1, p=2, dim=1)
    x2 = F.normalize(x2, p=2, dim=1)

    sim_matrix = torch.mm(x1, x2.T) / tau

    sim_matrix_exp = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True).values)

    pos_sim = torch.diag(sim_matrix_exp)

    neg_sim_matrix = sim_matrix_exp * (1 - torch.eye(sim_matrix.shape[0], device=sim_matrix.device))

    topk_neg_sim, _ = torch.topk(neg_sim_matrix, k=top_k, dim=1, largest=True)
    topk_neg_sim = topk_neg_sim.mean(dim=1) 

    hard_neg_sim, _ = torch.topk(neg_sim_matrix, k=num_hard_neg, dim=1, largest=False)
    hard_neg_sim = hard_neg_sim.mean(dim=1) 

    loss = -torch.log(pos_sim / (0.5 * topk_neg_sim + 0.5 * hard_neg_sim + 1e-8))

    return loss.mean()

def minmax_norm(x):
    x_min = x.min()
    x_max = x.max()
    if torch.isclose(x_max, x_min):  
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min + 1e-8)