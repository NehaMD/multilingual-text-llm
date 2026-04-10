import torch
import torch.nn.functional as F
import numpy as np
import os
import math

from model import LanguageModel

log_file = open("training_log.txt", "a")

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 10
learning_rate = 2e-4
min_lr = 2e-5        # Learning rate decays down to this value
warmup_iters = 200   # How many steps to spend warming up
checkpoint_interval = 1000

# BUG FIX: Vocab Size Mismatch
# Previously this was set to 32000, but the tokenizer was trained with 16000.
# This caused the model to predict across 16000 'dead' tokens, leading to gibberish.
vocab_size = 16000  
embed_dim = 512
num_heads = 8
num_layers = 8

# -----------------------------
# LOAD TOKENS
# -----------------------------
# BUG FIX: Data Leakage & Bloat
# Previously, the data was loaded from X.npy and Y.npy which were overlapping sliding windows.
# This caused the validation set to share up to 127 identical tokens with the training set!
# Instead, we now load the raw 1D token stream and slice dynamically.
tokens = np.fromfile("tokenization/tokens.bin", dtype=np.int32)
data = torch.tensor(tokens, dtype=torch.long)

# -----------------------------
# TRAIN / VAL SPLIT
# -----------------------------
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# -----------------------------
# BATCH FUNCTION
# -----------------------------
def get_batch(split):
    # Select the appropriate split
    split_data = train_data if split == "train" else val_data

    # Randomly pick starting indices for the batch, leaving room for block_size + 1 (for Y targets)
    ix = torch.randint(0, len(split_data) - block_size - 1, (batch_size,))
    
    # Dynamically slice X (input) and Y (target which is X shifted right by 1)
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

print("Total tokens loaded:", len(data))
print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))

# -----------------------------
# MODEL INIT
# -----------------------------
model = LanguageModel(
    vocab_size,
    block_size,   
    embed_dim,
    num_heads,
    num_layers
)
model = model.to(device)

# -----------------------------
# OPTIMIZER
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------------------
# LR SCHEDULER (Cosine Decay)
# -----------------------------
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------
# LOAD CHECKPOINT (if exists)
# -----------------------------
start_step = 0
checkpoint_path = "latest_checkpoint.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint["step"]
    print(f"Resumed from step {start_step}")

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        total_loss = 0
        eval_iters = 50

        for _ in range(eval_iters):
            x, y = get_batch(split)
            if _ % 10 == 0:
                print(f"  Eval step {_}/{eval_iters} for {split}")
            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1)
            )
            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()
    return losses

# -----------------------------
# TRAINING LOOP
# -----------------------------
for step in range(start_step, max_iters):

    # ---- EVALUATION ----
    if step % eval_interval == 0:
        losses = estimate_loss()
        log_line = f"Step {step}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    # ---- CHECKPOINT ----
    if step % checkpoint_interval == 0 and step > 0:
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step
        }

        torch.save(checkpoint, f"checkpoint_step_{step}.pth")

        # also save a "latest" checkpoint (for easy resume)
        torch.save(checkpoint, "latest_checkpoint.pth")

        print(f"Checkpoint saved at step {step}")

    # ---- TRAIN STEP ----
    # Determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_group:
        param_group['lr'] = lr

    x, y = get_batch("train")

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------------
# FINAL SAVE
# -----------------------------
torch.save(model.state_dict(), "model.pth")

log_file.close()

print("Training complete. Model saved as model.pth")