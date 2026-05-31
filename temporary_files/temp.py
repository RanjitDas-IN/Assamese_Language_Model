## -------------------------Write the first 100 lins of a txt to another txt---------------------------
# import os
# target_file= "data/1B_assamese_Tokens_Quwn3/cleanned_1B_Quwn_tokens.txt"
# output_file = "first_100_lines_of_" + os.path.basename(target_file)
# # output_file = "100_demo_lines_from_my_dataset.txt"
# # print(output_file)
# with open(target_file, encoding="utf-8") as f, \
#      open(output_file, "w", encoding="utf-8") as out:
#     for i in range(500):
#         line = f.readline()
#         if not line:
#             break
#         out.write(line)
# # print("\n\nDone")


#-------------------------print what is inside the txt file-----------------------------
# with open("cleanned_1B_as_tokens_unfiltered.txt", encoding="utf-8") as f:
#     for _ in range(20):
#         print(f.readline(), end="")



#-------------------------parquet to txt for single file-----------------------------
# import pandas as pd

# df = pd.read_parquet(
#     r"/home/ranjit/Downloads/wiki_asm_Beng_0000_of_0063.parquet",
#     columns=["text"]
# )
# # print(df)
# with open("data/ai4bharat_sangraha_dataset/synthetic/wiki_asm_Beng_0000_of_0063.txt", "w", encoding="utf-8") as f:
#     for row in df["text"]:
#         f.write(str(row) + "\n")



#-------------------------parquet to txt for multiple files-----------------------------

# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm

# # Folder containing parquet files
# input_dir = Path("/home/ranjit/Downloads/sangrah_parquet")

# # Output folder
# output_dir = Path("data/ai4bharat_sangraha_dataset/synthetic2")
# output_dir.mkdir(parents=True, exist_ok=True)

# # Find all matching parquet files
# parquet_files = sorted(input_dir.glob("wiki_asm_Beng_*_of_0063.parquet"))

# for parquet_file in tqdm(parquet_files, desc="Converting parquet -> txt"):
    
#     # Read only "text" column
#     df = pd.read_parquet(parquet_file, columns=["text"])

#     # Create output txt filename with same name
#     output_txt = output_dir / f"{parquet_file.stem}.txt"

#     # Save text data
#     with open(output_txt, "w", encoding="utf-8") as f:
#         for row in tqdm(
#             df["text"],
#             desc=f"Writing {parquet_file.stem}",
#             leave=False
#         ):
#             f.write(str(row) + "\n")

# print("✅ All parquet files converted to txt.")



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
        
        
#-----------------------count(.bin) the num of tokens-------------------------------
# import numpy as np
# from pathlib import Path

# base = Path("token_shards")

# total_tokens = 0

# for split in ["train", "test", "val"]:
#     split_path = base / split

#     if not split_path.exists():
#         continue

#     print(f"\n[{split.upper()}]")

#     split_total = 0

#     for bin_file in sorted(split_path.glob("*.bin")):
#         tokens = np.fromfile(bin_file, dtype=np.uint16)

#         count = len(tokens)
#         split_total += count
#         total_tokens += count

#         print(f"{bin_file.name}: {count:,} tokens")

#     print(f"Total {split}: {split_total:,} tokens")

# print(f"\nGRAND TOTAL: {total_tokens:,} tokens")



# ------------------------------------words to tokens shards------------------------------------------

# from pathlib import Path
# import numpy as np
# from tokenizers import Tokenizer


# # ===== HARD CODED PATHS =====
# INPUT_FILE = "data/personal_cards.txt"
# TOKENIZER_JSON = "The_Assamese_Tokenizer/assamese_tokenizer/tokenizer.json"
# OUTPUT_FILE = "token_shards/train/train_003.bin"


# # Load tokenizer
# tokenizer = Tokenizer.from_file(TOKENIZER_JSON)

# # Read text
# text = Path(INPUT_FILE).read_text(encoding="utf-8")

# # Convert text to token ids
# ids = tokenizer.encode(text).ids

# # Save tokens as binary
# np.array(ids, dtype=np.uint16).tofile(OUTPUT_FILE)

# print(f"Saved {len(ids)} tokens -> {OUTPUT_FILE}")