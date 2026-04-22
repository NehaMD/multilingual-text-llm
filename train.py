import torch
import torch.nn.functional as F
import numpy as np
import os
import math

from model import LanguageModel

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# -----------------------------
# HYPERPARAMETERS (fixed ones)
# -----------------------------
batch_size = 32
block_size = 128
max_iters = 10000
eval_interval = 500      
learning_rate = 1e-4
min_lr = 1e-5
warmup_iters = 2000
checkpoint_interval = 2000

vocab_size = 16000
num_heads = 4

# -----------------------------
# LOAD TOKENS (once)
# -----------------------------
tokens = np.fromfile("tokenization/tokens.bin", dtype=np.int32)
data = torch.tensor(tokens, dtype=torch.long)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

print("Total tokens:", len(data))

# -----------------------------
# BATCH FUNCTION
# -----------------------------
def get_batch(split):
    split_data = train_data if split == "train" else val_data

    ix = torch.randint(0, len(split_data) - block_size - 1, (batch_size,))
    x = torch.stack([split_data[i:i+block_size] for i in ix])
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

# -----------------------------
# LR SCHEDULER
# -----------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        total_loss = 0
        eval_iters = 20   # ⬅️ faster eval

        for _ in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1)
            )
            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()
    return losses

# =========================================================
#  MAIN TRAIN FUNCTION
# =========================================================
def run_training(embed_dim, num_layers):

    print(f"\nStarting: embed={embed_dim}, layers={num_layers}")

    # ---- model ----
    model = LanguageModel(
        vocab_size,
        block_size,
        embed_dim,
        num_heads,
        num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ---- unique log file ----
    log_file = open(f"log_e{embed_dim}_l{num_layers}.csv", "w")
    log_file.write("step,train_loss,val_loss\n")

    # ---- unique checkpoint ----
    checkpoint_path = f"ckpt_e{embed_dim}_l{num_layers}.pth"

    start_step = 0

    # optional resume (per config)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    best_val_loss = float("inf")

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for step in range(start_step, max_iters):

        # ---- evaluation ----
        if step % eval_interval == 0:
            losses = estimate_loss(model)

            train_loss = losses["train"]
            val_loss = losses["val"]

            print(f"[e={embed_dim}, l={num_layers}] Step {step} | Train {train_loss:.4f} | Val {val_loss:.4f}")

            # CSV logging
            log_file.write(f"{step},{train_loss},{val_loss}\n")
            log_file.flush()

            # SAVE BEST MODEL
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save({
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "step": step
                }, f"best_model_e{embed_dim}_l{num_layers}.pth")

                print(f"New best model saved (val_loss={val_loss:.4f})")

        # ---- checkpoint ----
        if step % checkpoint_interval == 0 and step > 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step
            }, checkpoint_path)

        # ---- training step ----
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
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

    # ---- final save ----
    torch.save(model.state_dict(), f"model_e{embed_dim}_l{num_layers}.pth")

    log_file.close()

    # ---- return best val loss ----
    final_losses = estimate_loss(model)
    return best_val_loss