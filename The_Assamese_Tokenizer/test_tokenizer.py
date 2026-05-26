"""
My tokenizer correctly returns integer token ID lists with automatic BOS/EOS tokens, meaning it is fully ready for uint16 token shard generation. it also have special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
"""

from tokenizers import Tokenizer
# Load tokenizer
tokenizer = Tokenizer.from_file(
    "The_Assamese_Tokenizer/assamese_tokenizer/tokenizer.json"
)

# Demo Assamese text
text = "মই অসম ভাল পাওঁ"

# Encode
output = tokenizer.encode(text)

# Check output type
print("Output object type:")
print(type(output))

print("\nToken IDs:")
print(output.ids)

print("\nType of token IDs:")
print(type(output.ids))

print("\nFirst token type:")
print(type(output.ids[0]))

print("\nTokens:")
print(output.tokens)