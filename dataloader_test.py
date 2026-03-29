from dataset import TextDataset
from torch.utils.data import DataLoader

dataset = TextDataset("data.txt")
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

for X_batch, Y_batch in dataloader:
    print("X batch shape:", X_batch.shape)
    print("Y batch shape:", Y_batch.shape)
    
    # Check dimensions
    print("Batch size:", X_batch.shape[0])
    print("Sequence length:", X_batch.shape[1])
    
    break