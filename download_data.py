from datasets import load_dataset

# Load public dataset (no auth required)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

with open("data.txt", "w", encoding="utf-8") as f:
    count = 0
    
    for sample in dataset:
        text = sample["text"].strip()
        
        if len(text) > 0:
            f.write(text + "\n")
            count += 1
        
        if count >= 10000:
            break

print("Saved 10,000 lines to data.txt")