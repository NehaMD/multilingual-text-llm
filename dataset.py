from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=128):
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file("tokenizer.json")

        # Load data
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor(encoded.ids, dtype=torch.long)

        # Chunking
        num_chunks = len(input_ids) // seq_length
        input_ids = input_ids[:num_chunks * seq_length]
        chunks = input_ids.view(num_chunks, seq_length)

        # Create X and Y
        self.X = chunks[:, :-1]
        self.Y = chunks[:, 1:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Test dataset
dataset = TextDataset("data.txt")

print("Dataset size:", len(dataset))

x, y = dataset[0]
print("Sample X shape:", x.shape)
print("Sample Y shape:", y.shape)
