from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Sample sentence
text = "Hello this is a test sentence"

# Encode
encoded = tokenizer.encode(text)

print("Original Text:")
print(text)

print("\nEncoded Tokens:")
print(encoded.tokens)

print("\nToken IDs:")
print(encoded.ids)

# Decode
decoded = tokenizer.decode(encoded.ids)

print("\nDecoded Text:")
print(decoded)
