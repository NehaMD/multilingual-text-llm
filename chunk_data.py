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

# Define sequence length
seq_length = 128

# Number of full chunks
num_chunks = len(input_ids) // seq_length

# Trim extra tokens
input_ids = input_ids[:num_chunks * seq_length]

# Reshape into chunks
chunks = input_ids.view(num_chunks, seq_length)

print("Number of chunks:", num_chunks)
print("Chunk shape:", chunks.shape)

# Print first chunk
print("First chunk:", chunks[0])
