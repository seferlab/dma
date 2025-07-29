import torch
import torch.nn as nn
import torch.nn.functional as F

class WGAANLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=0.5):
        super(WGAANLayer, self).__init__()
        self.fc = nn.Linear(2 * in_features, out_features)
        self.beta = beta

    def forward(self, h, adj, amount, count):
        # Edge weight computation
        amount_norm = (amount - amount.min()) / (amount.max() - amount.min() + 1e-8)
        count_norm = (count - count.min()) / (count.max() - count.min() + 1e-8)
        edge_weight = self.beta * amount_norm + (1 - self.beta) * count_norm

        # Attention
        h_i, h_j = h[adj[0]], h[adj[1]]
        h_cat = torch.cat([h_i, h_j], dim=1)
        score = F.leaky_relu(self.fc(h_cat)) * edge_weight.unsqueeze(1)
        attention = torch.softmax(score, dim=0)

        agg = torch.zeros_like(h)
        agg.index_add_(0, adj[0], attention * h_j)

        return agg
