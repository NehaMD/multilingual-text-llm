import torch
import torch.nn as nn
from dataset import TextDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = TextDataset("data.txt")
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Dummy model
class DummyModel(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

# Initialize
model = DummyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop (few batches only)
for i, (X_batch, Y_batch) in enumerate(dataloader):
    optimizer.zero_grad()

    outputs = model(X_batch)

    outputs = outputs.view(-1, outputs.size(-1))
    Y_batch = Y_batch.view(-1)

    loss = criterion(outputs, Y_batch)

    loss.backward()
    optimizer.step()

    print(f"Step {i}, Loss: {loss.item()}")

    # stop after few steps (as per plan)
    if i >= 10:
        break
