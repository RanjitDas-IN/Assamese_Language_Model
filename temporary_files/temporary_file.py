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

# ---------------------------------------------------------Add [BOS], [EOS] Token at tha start and end of the lines of a txt-------------------------------------------------------------------

# input_file = "data/personal_cards.txt"
# output_file = "data/personal_cards_bos_eos.txt"

# with open(input_file, "r", encoding="utf-8") as fin, \
#      open(output_file, "w", encoding="utf-8") as fout:

#     for line in fin:
#         print("EOS, BOS")
#         line = line.strip()

#         if not line:
#             continue

#         fout.write(f"[BOS]{line}[EOS]\n")

# print("Done:", output_file)



# # --------------------------------------------------demo only--------------------------------------------------------------------

# from pathlib import Path

# input_file = "data/1B_assamese_Tokens_Quwn3/ultra_cleanned_1B_Quwn_tokens.txt"
# output_file = "data/1B_assamese_Tokens_Quwn3/ultra_cleanned_1B_Quwn_tokens_bos_eos.txt"

# with open(input_file, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# docs = []
# current_doc = []

# for line in lines:
#     if not line.strip():
#         continue

#     # New article = line does NOT start with a space
#     if not line.startswith(" ") and current_doc:
#         docs.append("".join(current_doc).strip())
#         current_doc = []

#     current_doc.append(line)

# if current_doc:
#     docs.append("".join(current_doc).strip())

# with open(output_file, "w", encoding="utf-8") as f:
#     for doc in docs:
#         f.write("[BOS]\n")
#         f.write(doc)
#         f.write("\n[EOS]\n\n")

# print(f"Documents found: {len(docs)}")



# ---------------------------------------This is for ai4bharat_sangraha_dataset only WALK-IN TO ALL TXT--------------------------------------

# from pathlib import Path

# base_dir = Path("data/ai4bharat_sangraha_dataset/synthetic2")

# for input_file in sorted(base_dir.glob("ultra_cleanned_wiki_asm_Beng_*_of_0063.txt")):

#     # Skip already processed files
#     if input_file.stem.endswith("_bos_eos"):
#         continue

#     output_file = input_file.with_name(
#         f"{input_file.stem}_bos_eos.txt"
#     )

#     with open(input_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     docs = []
#     current_doc = []

#     for line in lines:
#         if not line.strip():
#             continue

#         # New article = line does NOT start with a space
#         if not line.startswith(" ") and current_doc:
#             docs.append("".join(current_doc).strip())
#             current_doc = []

#         current_doc.append(line)

#     if current_doc:
#         docs.append("".join(current_doc).strip())

#     with open(output_file, "w", encoding="utf-8") as f:
#         for doc in docs:
#             f.write("[BOS]\n")
#             f.write(doc)
#             f.write("\n[EOS]\n\n")

#     print(
#         f"Processed: {input_file.name} -> {output_file.name} | Documents found: {len(docs)}"
#     )

# print("\nDone.")

# ---------------------------------------This is for data/AsRED only--------------------------------------

# import re

# INPUT_FILE = "data/rahular_varta_DailyHuntDataset/cleanned_val_as_shard_01.txt"
# OUTPUT_FILE = "data/rahular_varta_DailyHuntDataset/Ultra_cleanned_val_as_shard_01.txt"

# # Precompile regexes
# START_HASH_RE = re.compile(r"^#")
# PAREN_RE = re.compile(r"\([^)]*\)")
# HASHTAG_WORD_RE = re.compile(r"\S*#\S*")
# EN_NUM_RE = re.compile(r"[A-Za-z0-9]+")
# MULTISPACE_RE = re.compile(r" +")
# ASSAMESE_WORD_RE = re.compile(r"[\u0980-\u09FF]+")

# with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
#      open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

#     for line in fin:

#         # 1. Remove lines starting with #
#         if START_HASH_RE.match(line):
#             print("Clkeanning Processing")
#             continue

#         # 2. Apply cleaning
#         line = PAREN_RE.sub("", line)
#         line = HASHTAG_WORD_RE.sub("", line)
#         line = EN_NUM_RE.sub("", line)
#         line = line.replace("#", "")
#         line = MULTISPACE_RE.sub(" ", line)
#         line = line.strip()

#         if not line:
#             continue

#         # 3. Keep only lines with >=25 Assamese/Bengali words
#         words = ASSAMESE_WORD_RE.findall(line)

#         if len(words) < 25:
#             continue

#         fout.write(line + "\n")

# ---------------------------------------This is for indicaqa_as_json only--------------------------------------

# import json

# input_path = r"FT_Data/indicaqa_as_json/indicqa.as.json"
# output_path = r"FT_Data/indicaqa_as_json/indicqa_as_sft.jsonl"

# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # IMPORTANT
# articles = data["data"]

# count = 0

# with open(output_path, "w", encoding="utf-8") as fout:
#     for article in articles:
#         for paragraph in article["paragraphs"]:
#             context = paragraph["context"].strip()

#             for qa in paragraph["qas"]:
#                 if qa["category"] != "SHORT":
#                     continue

#                 if len(qa["answers"]) == 0:
#                     continue

#                 answer = qa["answers"][0]["text"].strip()

#                 if answer == "":
#                     continue

#                 question = qa["question"].strip()

#                 sample = {
#                     "messages": [
#                         {
#                             "role": "user",
#                             "content": f"[BOS]প্ৰশ্ন: {question}[EOS]"
#                         },
#                         {
#                             "role": "assistant",
#                             "content": f"[BOS]{answer}[EOS]"
#                         }
#                     ]
#                 }

#                 fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
#                 count += 1

# print("Examples written:", count)
# ---------------------------------------Find duplicate--------------------------------------
# import json
# from collections import defaultdict

# input_path = "FT_Data/indicaqa_as_json/indicqa.as.json"
# output_path = "FT_Data/indic_align_instruct/OASST.jsonl"

# # First pass: collect occurrences
# occurrences = defaultdict(list)

# with open(input_path, "r", encoding="utf-8") as fin:
#     for line_num, line in enumerate(fin, 1):
#         obj = json.loads(line)

#         key = (
#             obj["messages"][0]["content"],
#             obj["messages"][1]["content"]
#         )

#         occurrences[key].append((line_num, line))

# # Print duplicates
# duplicate_count = 0
# for key, lines in occurrences.items():
#     if len(lines) > 1:
#         duplicate_count += len(lines) - 1

#         # print("\n" + "="*80)
#         # print(f"Found {len(lines)} identical copies:")
#         # print(key[0])  # user
#         # print(key[1])  # assistant
#         # print("Line numbers:", [x[0] for x in lines])

# print("\nDuplicate lines:", duplicate_count)

# # Ask user
# choice = input("\nRemove duplicates? (y/n): ").strip().lower()

# if choice == "y":
#     seen = set()

#     with open(input_path, "r", encoding="utf-8") as fin, \
#          open(output_path, "w", encoding="utf-8") as fout:

#         for line in fin:
#             obj = json.loads(line)

#             key = (
#                 obj["messages"][0]["content"],
#                 obj["messages"][1]["content"]
#             )

#             if key not in seen:
#                 seen.add(key)
#                 fout.write(line)

#     print("Duplicates removed.")
#     print("Saved to:", output_path)

# else:
#     print("No changes made.")