from train import run_training
import csv

EMBED_DIMS = 256
NUM_LAYERS = [1, 2, 4, 8]

results = []

for e in EMBED_DIMS:
    for l in NUM_LAYERS:
        val_loss = run_training(e, l)

        results.append({
            "embed_dim": e,
            "num_layers": l,
            "val_loss": val_loss
        })

# save summary
with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["embed_dim", "num_layers", "val_loss"])
    writer.writeheader()
    writer.writerows(results)