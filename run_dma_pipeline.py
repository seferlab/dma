from utils.training import train_dma
from utils.data_loader import load_transaction_graph

if __name__ == "__main__":
    G, features, labels = load_transaction_graph("data/ethereum_sample.csv")
    train_dma(G, features, labels)
