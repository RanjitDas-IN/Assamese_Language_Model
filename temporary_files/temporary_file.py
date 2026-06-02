#-------------------------find all .txt files inside nested folders-----------------------------
# from pathlib import Path
# root_dir = Path("data")


# txt_files = [file.name for file in root_dir.rglob("*.txt")]
# print("[")
# for files in txt_files:
#     print(f'"{files}",')
# print("]")


#-----------------------shards (.bin) testing (converting to original words)-------------------------------
# import numpy as np
# from tokenizers import Tokenizer

# # Load tokenizer
# tokenizer = Tokenizer.from_file(
#     "The_Assamese_Tokenizer/assamese_tokenizer/tokenizer.json"
# )

# # Load token IDs
# tokens = np.fromfile(
#     "token_shards/test/test_001.bin",
#     dtype=np.uint16
# )
# print(tokens[:100])
# # Save token_id : decoded_token
# with open("decoded_tokens.txt", "w", encoding="utf-8") as f:
#     for token_id in tokens[:100]:
#         token_text = tokenizer.decode([int(token_id)])
#         f.write(f"{token_id}: {token_text}\n")

# #-----------------------marge 2 .bin files-------------------------------
# import numpy as np
# from pathlib import Path

# # ─────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────
# INPUT_2 = "token_shards/train/train_003.bin"
# INPUT_1 = "token_shards/train/train_002.bin"

# OUTPUT  = "token_shards/train/marged_002.bin"

# UINT16_DTYPE = np.dtype(np.uint16)
# UINT16_MAX   = np.iinfo(np.uint16).max
# UINT16_BYTES = UINT16_DTYPE.itemsize

# # ─────────────────────────────────────────────
# # Total
# # ─────────────────────────────────────────────

# size_1 = Path(INPUT_1).stat().st_size
# size_2 = Path(INPUT_2).stat().st_size

# print(f"size of {INPUT_1} : {size_1 / (1024**2):.2f} MB")
# print(f"size of {INPUT_2} : {size_2 / (1024**2):.2f} MB")

# TOTAL_SIZE = size_1 + size_2

# print(f"Expected total file size : {TOTAL_SIZE / (1024**2):.2f} MB")
# print("=" * 50)


# # ─────────────────────────────────────────────
# # Validation
# # ─────────────────────────────────────────────
# for p in [INPUT_1, INPUT_2]:
#     if not Path(p).exists():
#         raise FileNotFoundError(f"Missing file: {p}")

#     file_size = Path(p).stat().st_size

#     if file_size % UINT16_BYTES != 0:
#         raise ValueError(
#             f"{p} is corrupted or not uint16-aligned "
#             f"(size={file_size} bytes)"
#         )

# # ─────────────────────────────────────────────
# # Load as uint16 token IDs
# # ─────────────────────────────────────────────
# tokens_1 = np.fromfile(INPUT_1, dtype=UINT16_DTYPE)
# tokens_2 = np.fromfile(INPUT_2, dtype=UINT16_DTYPE)

# # ─────────────────────────────────────────────
# # Extra safety check
# # ─────────────────────────────────────────────
# if tokens_1.max() > UINT16_MAX:
#     raise ValueError(f"{INPUT_1} contains invalid token IDs")

# if tokens_2.max() > UINT16_MAX:
#     raise ValueError(f"{INPUT_2} contains invalid token IDs")

# # ─────────────────────────────────────────────
# # Merge
# # ─────────────────────────────────────────────
# merged = np.concatenate([tokens_1, tokens_2])

# # ─────────────────────────────────────────────
# # Save
# # ─────────────────────────────────────────────
# Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

# merged.astype(UINT16_DTYPE).tofile(OUTPUT)

# # ─────────────────────────────────────────────
# # Info
# # ─────────────────────────────────────────────
# print("=" * 50)
# print("MERGE COMPLETE")
# print("=" * 50)

# print(f"Input 1 tokens : {len(tokens_1):,}")
# print(f"Input 2 tokens : {len(tokens_2):,}")
# print(f"Total tokens   : {len(merged):,}")

# print()
# print(f"Saved to       : {OUTPUT}")
# print(f"Output size    : {Path(OUTPUT).stat().st_size / (1024**2):.2f} MB")
# print("=" * 50)

