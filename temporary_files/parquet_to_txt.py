# import pandas as pd
# import json
# import ast

# input_path = r"FT_Data/wiki_chat/wiki_chat_0000_of_0031.parquet"

# df = pd.read_parquet(input_path)
# # print(df)
# # print(df.columns)
# d = df.head()
# d.to_csv(f"Dolly.csv")

# --------------------------------Remove duplicate--------------------------------------
# import json
# from collections import defaultdict

# input_path = "FT_Data/Updesh_beta/creative_writing.jsonl"
# output_path = "FT_Data/Updesh_beta/Cleaned_Multihop_reasoning.jsonl"

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
# --------------------------------Updesh_beta--------------------------------------

# import pandas as pd
# import json
# import ast

# input_path = r"FT_Data/Updesh_beta/Multihop_reasoning.parquet"
# output_path = r"FT_Data/Updesh_beta/Multihop_reasoning.jsonl"

# df = pd.read_parquet(input_path)

# count = 0

# with open(output_path, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         messages = row["messages"]

#         # Convert string to list if necessary
#         if isinstance(messages, str):
#             messages = ast.literal_eval(messages)

#         converted = []

#         for msg in messages:
#             if msg["role"] in ("user", "assistant"):
#                 converted.append({
#                     "role": msg["role"],
#                     "content": f"[BOS]{msg['content']}[EOS]"
#                 })

#         if converted:
#             f.write(json.dumps({"messages": converted}, ensure_ascii=False) + "\n")
#             count += 1

# print("Examples written:", count)


# # --------------------------------indic_align_instruct--------------------------------------
# from pathlib import Path
# import json
# import pyarrow.parquet as pq

# input_dir = Path("FT_Data/wiki_chat")

# for parquet_path in sorted(input_dir.glob("*.parquet")):
#     output_path = parquet_path.with_suffix(".jsonl")
#     count = 0

#     pf = pq.ParquetFile(parquet_path)

#     with output_path.open("w", encoding="utf-8") as f:
#         for batch in pf.iter_batches(columns=["asm_Beng"], batch_size=1000):
#             column = batch.column(0)  # only one selected column

#             for cell in column.to_pylist():
#                 try:
#                     if cell is None:
#                         continue

#                     # Expected shape: [[question, answer]]
#                     qa = cell[0] if isinstance(cell, (list, tuple)) and len(cell) > 0 else cell

#                     user_text = str(qa[0]).strip()
#                     assistant_text = str(qa[1]).strip()

#                     sample = {
#                         "messages": [
#                             {
#                                 "role": "user",
#                                 "content": f"[BOS]{user_text}[EOS]"
#                             },
#                             {
#                                 "role": "assistant",
#                                 "content": f"[BOS]{assistant_text}[EOS]"
#                             }
#                         ]
#                     }

#                     f.write(json.dumps(sample, ensure_ascii=False) + "\n")
#                     count += 1

#                 except Exception as e:
#                     print(f"Skipping row in {parquet_path.name}:", e)

#     print(f"{parquet_path.name} -> {output_path.name} | Written examples: {count}")