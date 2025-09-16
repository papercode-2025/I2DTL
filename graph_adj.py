import torch
import math

class dynamic_regional_graph(torch.nn.Module):
    def __init__(self, window_size=15):
        super(dynamic_regional_graph, self).__init__()
        self.window_size = window_size
    def forward(self, x, dia_len, qmask):
        batch_size, seq_len, _ = x.shape
        adj = torch.zeros((batch_size, seq_len, seq_len), device=x.device)

        for b, len_ in enumerate(dia_len):

            speakers = qmask[:len_, b].argmax(dim=1)
            for i in range(len_):
                start = max(0, i - self.window_size)
                end = min(len_, i + self.window_size + 1)
                window_indices = torch.arange(start, end, device=x.device)
                same_speaker_mask = (speakers == speakers[i])
                in_window_mask = torch.zeros(len_, dtype=torch.bool, device=x.device)
                in_window_mask[window_indices] = True
                speaker_indices = torch.where(same_speaker_mask & in_window_mask)[0]
                
                sim1 = self.atom_calculate_edge_weight(x[b][i, :].detach(), x[b][start: end, :].detach())
                adj[b, i, window_indices] = 1 - torch.acos(sim1) / torch.pi

                if len(speaker_indices) > 1:
                    sim2 = self.atom_calculate_edge_weight(x[b][i, :].detach(), x[b][speaker_indices, :].detach())
                    adj[b, i, speaker_indices] = adj[b, i, speaker_indices] + 1 - torch.acos(sim2) / torch.pi

            degrees = adj[b].sum(1)
            degrees[degrees == 0] = 1 
            D = torch.diag(torch.pow(degrees, -0.5))
            adj[b] = torch.matmul(torch.matmul(D, adj[b]), D)

        return adj
    
    @staticmethod
    def atom_calculate_edge_weight(x1, x2):
        f = torch.nn.functional.cosine_similarity(x1, x2, dim=1)
        f = torch.clamp(f, -1, 1)  
        return f
