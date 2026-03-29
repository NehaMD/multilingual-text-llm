from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize tokenizer with BPE model
tokenizer = Tokenizer(models.BPE())

# Use whitespace pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer
trainer = trainers.BpeTrainer(
    vocab_size=5000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train tokenizer
tokenizer.train(files=["data.txt"], trainer=trainer)

# Save tokenizer
tokenizer.save("tokenizer.json")

print("Tokenizer trained and saved!")
