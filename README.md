# DMA Ethereum Transaction Tracking

This repository contains the implementation of the DMA (Directed Multigraph Attention) model for Solana transaction tracking.

## Structure
- `run_dma_pipeline.py`: Entry point for training the model.
- `models/`: Contains model components (`WGAANLayer`, `DMAModel`).
- `utils/`: Data loader and training utilities.
- `data/`: Folder for Ethereum transaction data (e.g. CSV with 'from', 'to', 'value', 'timestamp').

## Usage
```bash
cd src
python run_dma_pipeline.py
```

## Requirements
- PyTorch
- NetworkX
- Pandas
```
