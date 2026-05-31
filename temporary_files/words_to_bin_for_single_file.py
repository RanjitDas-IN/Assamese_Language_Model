
# ------------------------------------words to tokens shards------------------------------------------
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


# ===== HARD CODED PATHS =====
# INPUT_FILE = "data/personal_cards.txt"
TOKENIZER_JSON = "The_Assamese_Tokenizer/assamese_tokenizer/tokenizer.json"
# OUTPUT_FILE = "token_shards/train/train_003.bin"


# Load tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_JSON)

# Read text
text = Path(INPUT_FILE).read_text(encoding="utf-8")

# Convert text to token ids
ids = tokenizer.encode(text).ids

# Save tokens as binary
np.array(ids, dtype=np.uint16).tofile(OUTPUT_FILE)

print(f"Saved {len(ids)} tokens -> {OUTPUT_FILE}")