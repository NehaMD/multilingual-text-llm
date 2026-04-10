import torch
import torch.nn.functional as F
import numpy as np
from model import LanguageModel

# -----------------------------
# LOG FILE
# -----------------------------
log_file = open("evaluation_log.txt", "a")

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# HYPERPARAMETERS (same as training)
# -----------------------------
batch_size = 32
block_size = 128
# BUG FIX: Vocab Size Mismatch
# Previously this was set to 32000, which mismatched the tokenizer's 16000 VOCAB_SIZE.
vocab_size = 16000
embed_dim = 512
num_heads = 8
num_layers = 8

eval_iters = 100

# -----------------------------
# LOAD DATA
# -----------------------------
# BUG FIX: Data Leakage & Bloat
# Previously, we loaded overlapping X.npy and Y.npy. We now slice directly from the 1D token stream.
tokens = np.fromfile("tokenization/tokens.bin", dtype=np.int32)
data = torch.tensor(tokens, dtype=torch.long)

# -----------------------------
# TRAIN / VAL SPLIT (same as training)
# -----------------------------
split = int(0.9 * len(data))
val_data = data[split:]

# -----------------------------
# BATCH FUNCTION
# -----------------------------
def get_batch():
    ix = torch.randint(0, len(val_data) - block_size - 1, (batch_size,))
    x = torch.stack([val_data[i : i + block_size] for i in ix])
    y = torch.stack([val_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------
# LOAD MODEL
# -----------------------------
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

print("Model loaded successfully!")

# -----------------------------
# EVALUATION
# -----------------------------
@torch.no_grad()
def evaluate():
    total_loss = 0
    total_acc = 0
    total_top5_acc = 0

    for _ in range(eval_iters):
        x, y = get_batch()
        logits = model(x)

        # LOSS
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )
        total_loss += loss.item()

        # TOKEN ACCURACY
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        total_acc += acc.item()

        # TOP-5 ACCURACY
        top5 = torch.topk(logits, k=5, dim=-1).indices
        correct_top5 = (top5 == y.unsqueeze(-1)).any(dim=-1).float().mean()
        total_top5_acc += correct_top5.item()

    avg_loss = total_loss / eval_iters
    avg_acc = total_acc / eval_iters
    avg_top5 = total_top5_acc / eval_iters
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity, avg_acc, avg_top5

# -----------------------------
# RUN EVALUATION
# -----------------------------
loss, ppl, acc, top5 = evaluate()

# -----------------------------
# PRINT + LOG
# -----------------------------
log_line = (
    f"Validation Loss: {loss:.4f}\n"
    f"Perplexity: {ppl:.4f}\n"
    f"Token Accuracy: {acc:.4f}\n"
    f"Top-5 Accuracy: {top5:.4f}\n"
)


print("\n===== Evaluation Results =====")
print(log_line)

log_file.write(log_line)
log_file.close()

print("Metrics saved to evaluation_log.txt")