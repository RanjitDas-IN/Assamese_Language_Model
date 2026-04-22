# from datasets import load_dataset
# import sys

# # Load dataset in streaming mode (NO RAM issue)
# dataset = load_dataset("rahular/varta", split="train", streaming=True)

# output_file = "scrapped_text.txt"

# count = 0

# with open(output_file, "a", encoding="utf-8") as f:
#     for row in dataset:
#         text = row.get("text")

#         if text:
#             f.write(text.strip() + "\n")

#         count += 1

#         # progress indicator (important for long runs)
#         if count % 10000 == 0:
#             print(f"Processed {count} rows...", flush=True)

# print("Done.")

