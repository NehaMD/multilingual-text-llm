import sentencepiece as spm
from collections import Counter
import numpy as np

sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

eng_counts = Counter()
hi_counts = Counter()
eng_lengths = []
hi_lengths = []

with open('data/final_dataset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('[EN]'):
            text = line[4:].strip()
            toks = sp.encode(text, add_bos=False, add_eos=False)
            eng_counts.update(toks)
            eng_lengths.append(len(toks))
        elif line.startswith('[HI]'):
            text = line[4:].strip()
            toks = sp.encode(text, add_bos=False, add_eos=False)
            hi_counts.update(toks)
            hi_lengths.append(len(toks))

print('english lines', len(eng_lengths), 'hindi lines', len(hi_lengths))
print('avg eng len', np.mean(eng_lengths), 'avg hi len', np.mean(hi_lengths))
print('median eng', np.median(eng_lengths), 'median hi', np.median(hi_lengths))
print('max eng', max(eng_lengths), 'max hi', max(hi_lengths))
print('min eng', min(eng_lengths), 'min hi', min(hi_lengths))

eng_types = set(eng_counts.keys())
hi_types = set(hi_counts.keys())
print('english types', len(eng_types), 'hindi types', len(hi_types))
print('shared types', len(eng_types & hi_types))
print('eng-only types', len(eng_types - hi_types), 'hi-only types', len(hi_types - eng_types))
print('english total tokens', sum(eng_counts.values()), 'hindi total tokens', sum(hi_counts.values()))
print('top-10 eng', eng_counts.most_common(10))
print('top-10 hi', hi_counts.most_common(10))
