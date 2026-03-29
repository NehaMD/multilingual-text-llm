import torch
import torch.nn as nn
from tokenizers import Tokenizer
import torch.nn.functional as F

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Dummy model (same as training)
class DummyModel(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

# Initialize model
model = DummyModel()

# ⚠️ IMPORTANT: load trained weights (if saved later)
# For now, we use current model (as per plan stage)

model.eval()

# Prompt
prompt = "The meaning of life is"

# Encode prompt
input_ids = tokenizer.encode(prompt).ids
input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, T)

# Generate tokens
max_new_tokens = 20

for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(input_ids)  # (1, T, V)
    
    next_token_logits = outputs[0, -1, :]
    temperature = 0.8

    # Apply temperature
    scaled_logits = next_token_logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample from distribution
    next_token_id = torch.multinomial(probs, num_samples=1).item()

    # Append token
    input_ids = torch.cat(
        [input_ids, torch.tensor([[next_token_id]])], dim=1
    )

# Decode
generated_ids = input_ids[0].tolist()
generated_text = tokenizer.decode(generated_ids)

print("Generated text:")
print(generated_text)
