from tokenizers import Tokenizer
import torch

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Read data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize entire text
encoded = tokenizer.encode(text)

# Convert to tensor
input_ids = torch.tensor(encoded.ids, dtype=torch.long)

print("Total tokens:", len(input_ids))
print("Tensor shape:", input_ids.shape)

# Print first 20 tokens
print("First 20 token IDs:", input_ids[:20])
