import torch
import torch.nn.functional as F
import numpy as np
from model import LanguageModel
import sentencepiece as spm
import csv

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
# HYPERPARAMETERS
# -----------------------------
batch_size = 32
block_size = 128
vocab_size = 16000
embed_dim = 1024
num_heads = 4
num_layers = 4

eval_iters = 100

# -----------------------------
# LOAD DATA
# -----------------------------
tokens = np.fromfile("tokenization/tokens.bin", dtype=np.int32)
data = torch.tensor(tokens, dtype=torch.long)

split = int(0.9 * len(data))
val_data = data[split:]

# -----------------------------
# BATCH FUNCTION
# -----------------------------
def get_batch():
    ix = torch.randint(0, len(val_data) - block_size - 1, (batch_size,))
    x = torch.stack([val_data[i: i + block_size] for i in ix])
    y = torch.stack([val_data[i + 1: i + block_size + 1] for i in ix])
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

checkpoint = torch.load("best_model_e1024_l4.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# -----------------------------
# TOKENIZER
# -----------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

def encode(text):
    return sp.encode(text, out_type=int)

def decode(tokens):
    return sp.decode(tokens)

# -----------------------------
# PERPLEXITY
# -----------------------------
@torch.no_grad()
def evaluate_perplexity():
    total_loss = 0

    for _ in range(eval_iters):
        x, y = get_batch()
        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )
        total_loss += loss.item()

    avg_loss = total_loss / eval_iters
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity

# -----------------------------
# GENERATION FUNCTION
# -----------------------------
def generate(model, start_tokens, max_new_tokens=50, temperature=1.0):
    x = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :]

        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

    return x[0].cpu().numpy().tolist()

# -----------------------------
# LANGUAGE DETECTION
# -----------------------------
def is_hindi(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def is_english(text):
    return any('a' <= ch.lower() <= 'z' for ch in text)

# -----------------------------
# METRICS
# -----------------------------
def distinct_n(tokens, n=1):
    if len(tokens) < n:
        return 0
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return len(set(ngrams)) / len(tokens)

def overlap_score(prompt, generated):
    p = set(prompt.lower().split())
    g = set(generated.lower().split())
    return len(p & g) / len(p) if len(p) > 0 else 0

def repetition_rate(tokens):
    return 1 - (len(set(tokens)) / len(tokens)) if len(tokens) > 0 else 0

# -----------------------------
# TEST PROMPTS
# -----------------------------
test_prompts = [
    "The future of AI is",
    "India is a country that",
    "Artificial intelligence can",
    "भारत एक ऐसा देश है जहाँ",
    "मुझे लगता है कि",
]

# -----------------------------
# GENERATION EVALUATION
# -----------------------------
@torch.no_grad()
def evaluate_generation(temperatures, perplexity):

    print("\n===== GENERATION EVALUATION =====\n")

    temp_results = []

    # Detailed CSV
    with open("generate_e1024_l4.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "Temperature",
            "Prompt",
            "Generated_Text",
            "Language_Correct",
            "Distinct_1",
            "Distinct_2",
            "Overlap",
            "Repetition",
            "Perplexity"
        ])

        for temp in temperatures:
            print(f"\n===== Temperature: {temp} =====")

            lang_correct = 0
            total = len(test_prompts)

            total_d1 = total_d2 = total_overlap = total_rep = 0

            for prompt in test_prompts:
                input_tokens = encode(prompt)

                output_tokens = generate(
                    model,
                    input_tokens,
                    max_new_tokens=50,
                    temperature=temp
                )

                generated_part = output_tokens[len(input_tokens):]
                output_text = decode(generated_part)

                print(f"\nPrompt: {prompt}")
                print(f"Generated: {output_text}")

                # Language accuracy
                if is_hindi(prompt):
                    correct = is_hindi(output_text)
                else:
                    correct = is_english(output_text)

                lang_correct += int(correct)

                # Metrics
                d1 = distinct_n(generated_part, 1)
                d2 = distinct_n(generated_part, 2)
                overlap = overlap_score(prompt, output_text)
                rep = repetition_rate(generated_part)

                total_d1 += d1
                total_d2 += d2
                total_overlap += overlap
                total_rep += rep

                print(f"Language Correct: {correct}")
                print(f"D1: {d1:.4f}, D2: {d2:.4f}, Overlap: {overlap:.4f}, Rep: {rep:.4f}")

                writer.writerow([
                    temp, prompt, output_text, correct,
                    round(d1,4), round(d2,4),
                    round(overlap,4), round(rep,4),
                    round(perplexity,4)
                ])

            # Averages
            avg_lang = lang_correct / total
            avg_d1 = total_d1 / total
            avg_d2 = total_d2 / total
            avg_overlap = total_overlap / total
            avg_rep = total_rep / total

            print("\n--- Summary ---")
            print(f"Lang Acc: {avg_lang:.4f}")
            print(f"D1: {avg_d1:.4f}, D2: {avg_d2:.4f}")
            print(f"Overlap: {avg_overlap:.4f}, Rep: {avg_rep:.4f}")

            temp_results.append([
                temp, avg_lang, avg_d1, avg_d2, avg_overlap, avg_rep, perplexity
            ])

    # Summary CSV
    with open("eval_e1024_l4.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Temperature",
            "Language_Accuracy",
            "Distinct_1",
            "Distinct_2",
            "Overlap",
            "Repetition",
            "Perplexity"
        ])

        writer.writerows(temp_results)

    print("\n✅ Results saved to generation_results.csv and temperature_summary.csv")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    loss, ppl = evaluate_perplexity()

    print("\n===== Perplexity =====")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")

    temperatures = [0.5, 0.8, 1.0, 1.2]

    evaluate_generation(temperatures, ppl)