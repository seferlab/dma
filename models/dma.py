import torch
import torch.nn as nn
from models.wgaan import WGAANLayer

class DMAModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, beta=0.5):
        super(DMAModel, self).__init__()
        self.gnn1 = WGAANLayer(in_dim, hidden_dim, beta)
        self.gnn2 = WGAANLayer(hidden_dim, out_dim, beta)
        self.decoder = nn.Bilinear(out_dim, out_dim, 1)

    def forward(self, x, adj, amount, count, pairs):
        h = self.gnn1(x, adj, amount, count)
        h = self.gnn2(h, adj, amount, count)
        z_i, z_j = h[pairs[:, 0]], h[pairs[:, 1]]
        return torch.sigmoid(self.decoder(z_i, z_j)).squeeze()
