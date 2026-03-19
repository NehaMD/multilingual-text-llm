from datasets import load_dataset
import os
from itertools import islice

dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    data_files="data/hi-1.txt", 
    split="train",
    streaming=True
)

os.makedirs("./data", exist_ok=True)

with open("./data/hindi_dataset.txt", "w", encoding="utf-8") as f:
    for sample in islice(dataset, 50000):
        text = sample["text"].strip()   

        if text:
            f.write(text + "\n")

print("Saved lines!")