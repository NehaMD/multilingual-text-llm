import sentencepiece as spm
import matplotlib.pyplot as plt
from collections import Counter
import os

# Load the tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# Path to data
data_path = "./data/final_dataset.txt"
# Set to a positive integer to sample only a subset of lines for a faster plot.
# Set to None to process the entire file.
SAMPLE_LINES = None


def count_tokens_by_language():
    english_counts = Counter()
    hindi_counts = Counter()
    english_lines = 0
    hindi_lines = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_index, raw_line in enumerate(f):
            if SAMPLE_LINES is not None and line_index >= SAMPLE_LINES:
                break

            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[EN]"):
                lang = "en"
                text = line[4:].strip()
            elif line.startswith("[HI]"):
                lang = "hi"
                text = line[4:].strip()
            else:
                continue

            if not text:
                continue

            token_ids = sp.encode(text, add_bos=False, add_eos=False)
            if lang == "en":
                english_counts.update(token_ids)
                english_lines += 1
            else:
                hindi_counts.update(token_ids)
                hindi_lines += 1

    return english_counts, hindi_counts, english_lines, hindi_lines


def plot_distribution(token_counts, lang_name, n_lines):
    frequencies = list(token_counts.values())
    plt.figure(figsize=(10, 6))
    plt.hist(frequencies, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Token Frequency Distribution for {lang_name} ({n_lines} lines)')
    plt.xlabel('Frequency')
    plt.ylabel('Number of Tokens')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    out_file = f'{lang_name}_token_distribution.png'
    plt.savefig(out_file)
    print(f"Saved plot: {out_file}")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please ensure the data is available.")
        exit(1)

    english_counts, hindi_counts, english_lines, hindi_lines = count_tokens_by_language()

    print(f"English lines: {english_lines}")
    print(f"Hindi lines: {hindi_lines}")
    print(f"English token types: {len(english_counts)}")
    print(f"Hindi token types: {len(hindi_counts)}")
    print(f"English total token occurrences: {sum(english_counts.values())}")
    print(f"Hindi total token occurrences: {sum(hindi_counts.values())}")

    if english_counts:
        plot_distribution(english_counts, "English", english_lines)
    else:
        print("No English token data found.")

    if hindi_counts:
        plot_distribution(hindi_counts, "Hindi", hindi_lines)
    else:
        print("No Hindi token data found.")