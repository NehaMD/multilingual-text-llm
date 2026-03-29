import torch
import torch.nn as nn
from dataset import TextDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = TextDataset("data.txt")
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Get ONE batch
X_batch, Y_batch = next(iter(dataloader))

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)  # higher LR for overfitting

# Train on SAME batch repeatedly
for step in range(200):
    optimizer.zero_grad()

    outputs = model(X_batch)

    outputs = outputs.view(-1, outputs.size(-1))
    targets = Y_batch.view(-1)

    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
