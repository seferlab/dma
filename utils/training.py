import torch
import torch.nn as nn
import torch.optim as optim
from models.dma import DMAModel

def train_dma(G, features, labels, epochs=100, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMAModel(features.shape[1], 64, 32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # Dummy adjacency list, edge info
    adj = torch.randint(0, features.shape[0], (2, len(labels)))
    amount = torch.rand(len(labels))
    count = torch.randint(1, 10, (len(labels),))
    pairs = adj.T

    features, amount, count, labels = features.to(device), amount.to(device), count.to(device), labels.to(device)

    for epoch in range(epochs):
        model.train()
        preds = model(features, adj, amount, count, pairs)
        loss = loss_fn(preds, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
