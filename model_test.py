import torch
import torch.nn as nn
from dataset import TextDataset
from torch.utils.data import DataLoader

# Load data
dataset = TextDataset("data.txt")
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Dummy model
class DummyModel(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)      # (B, T, D)
        x = self.linear(x)         # (B, T, V)
        return x

# Initialize model
model = DummyModel()

# Get one batch
for X_batch, Y_batch in dataloader:
    output = model(X_batch)

    print("Input shape:", X_batch.shape)
    print("Output shape:", output.shape)
    break
