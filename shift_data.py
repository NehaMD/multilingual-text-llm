from tokenizers import Tokenizer
import torch

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Load data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize
encoded = tokenizer.encode(text)
input_ids = torch.tensor(encoded.ids, dtype=torch.long)

# Chunking
seq_length = 128
num_chunks = len(input_ids) // seq_length
input_ids = input_ids[:num_chunks * seq_length]
chunks = input_ids.view(num_chunks, seq_length)

# Create X and Y
X = chunks[:, :-1]   # all except last token
Y = chunks[:, 1:]    # all except first token

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Print first example
print("\nFirst X:", X[0])
print("\nFirst Y:", Y[0])
