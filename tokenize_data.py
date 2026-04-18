import sentencepiece as spm
import numpy as np
import torch
import os
from torch.utils.data import Dataset

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_DATA = "./data/final_dataset.txt"
OUTPUT_DIR = "./tokenization"
MODEL_PREFIX = "tokenizer"
VOCAB_SIZE = 16000
SEQ_LEN = 256

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_tokenizer():
    print(f"Training tokenizer on {INPUT_DATA}...")
    spm.SentencePieceTrainer.train(
        input=INPUT_DATA,
        model_prefix=MODEL_PREFIX,     # output files: tokenizer.model, tokenizer.vocab
        vocab_size=VOCAB_SIZE,            
        model_type="bpe",             
        character_coverage=0.9995,   
        user_defined_symbols=["[EN]", "[HI]"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print("Tokenizer trained successfully!")

def process_tokens():
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")
    
    bin_file = os.path.join(OUTPUT_DIR, "tokens.bin")
    
    print(f"Tokenizing {INPUT_DATA} into {bin_file}...")
    with open(bin_file, "wb") as f_out:
        with open(INPUT_DATA, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                # Encode line with BOS and EOS
                ids = sp.encode(line, add_bos=True, add_eos=True)
                np.array(ids, dtype=np.int32).tofile(f_out)

    print("Tokens saved to tokens.bin")

def create_datasets():
    bin_file = os.path.join(OUTPUT_DIR, "tokens.bin")
    tokens = np.fromfile(bin_file, dtype=np.int32)
    print(f"Total tokens loaded: {len(tokens)}")

    num_sequences = len(tokens) - SEQ_LEN
    if num_sequences <= 0:
        print("Error: Not enough tokens to create even one sequence!")
        return

    # BUG FIX: Data Leakage & Bloat 
    # Previously, this section created massive overlapping arrays (X.npy and Y.npy) 
    # using a sliding window. This caused Train/Val data leakage and wasted disk space.
    # We now skip this step because `train.py` dynamically slices directly from `tokens.bin`.
    print("Skipping X.npy and Y.npy creation. `train.py` now loads `tokens.bin` dynamically!")

if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists(INPUT_DATA):
        print(f"Error: Could not find '{INPUT_DATA}'. Ensure your data folder is present.")
    else:
        train_tokenizer()
        process_tokens()
        create_datasets()
        print("\nPreprocessing complete. You can now run train.py!")
