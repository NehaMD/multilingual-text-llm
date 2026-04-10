import torch
import torch.nn.functional as F
from model import LanguageModel
import sentencepiece as spm

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# HYPERPARAMETERS (MUST match training)
# -----------------------------
block_size = 128
# BUG FIX: Vocab Size Mismatch
# Updated to match the tokenizer's trained vocab size to prevent out-of-bounds generation errors.
vocab_size = 16000   # must match tokenizer vocab size
embed_dim = 512
num_heads = 8
num_layers = 8

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# -----------------------------
# ENCODE / DECODE
# -----------------------------
def encode(text):
    # BUG FIX: Missing BOS Token
    # The tokenizer adds BOS and EOS during training data creation. By explicitly prepending
    # the BOS token during sampling, we prevent Out-Of-Distribution prompt sequences.
    return sp.encode(text, add_bos=True)

def decode(tokens):
    return sp.decode(tokens)

# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model = LanguageModel(
        vocab_size,
        block_size,
        embed_dim,
        num_heads,
        num_layers
    )

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    return model

# -----------------------------
# GENERATION FUNCTION
# -----------------------------
def generate(model, start_tokens, max_new_tokens=100, temperature=1.0, top_k=40):
    tokens = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -block_size:]

        logits = model(tokens_cond)
        logits = logits[:, -1, :] / temperature

        # BUG FIX: Weak Decoding Strategy
        # Added Top-K sampling. Previously, raw multinomial sampling could easily select
        # completely incorrect 'long-tail' words and result in gibberish feedback loops.
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat((tokens, next_token), dim=1)

    return tokens[0].tolist()

# -----------------------------
# GENERATE TEXT
# -----------------------------
def generate_text(prompt, max_new_tokens=100, temperature=0.8):
    model = load_model()

    input_tokens = encode(prompt)
    output_tokens = generate(model, input_tokens, max_new_tokens, temperature)

    return decode(output_tokens)

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    print("\n--- TEXT GENERATION ---\n")

    print(sp.get_piece_size())

    prompts = {
        "English": "Once upon a time",
        "Hindi": "एक समय की बात है",
    }

    model = load_model()

    for lang, prompt in prompts.items():
        tokens = encode(prompt)
        output_tokens = generate(model, tokens, max_new_tokens=100, temperature=0.8)
        text = decode(output_tokens)

        print(f"\n[{lang}]")
        print(text)