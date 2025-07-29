import torch
import pandas as pd
import networkx as nx

def load_transaction_graph(path):
    df = pd.read_csv(path)
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['from'], row['to'], amount=row['value'], timestamp=row['timestamp'])

    # Dummy data for now
    features = torch.randn(len(G), 28)
    labels = torch.randint(0, 2, (len(G.edges),))

    return G, features, labels
