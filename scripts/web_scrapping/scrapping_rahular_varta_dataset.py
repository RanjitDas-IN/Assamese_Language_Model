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

# -------------------------------------------------------------------------------------------------------------------
"""
the above script take too much time,
so download the file from the huggingface,
specially `as` files, which are in JSON
and it has much more keys, but i need only the `Headline` and `Text` part
so Below script did that
"""

import json
import sys

INPUT_FILE = "val_as_shard_01.json"
OUTPUT_FILE = "data/rahular_varta_DailyHuntDataset/val_as_shard_01.txt"

count = 0

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line_number, line in enumerate(infile, start=1):

            line = line.strip()

            if not line:
                continue

            # Validate each JSON line
            try:
                item = json.loads(line)

            except json.JSONDecodeError as e:
                print("\n❌ BROKEN JSON DETECTED")
                print(f"Line Number: {line_number}")
                print(f"Error: {e}")
                sys.exit(1)

            headline = str(item.get("headline", "")).strip()
            text = str(item.get("text", "")).strip()

            if not headline and not text:
                continue

            outfile.write(f"{headline}: {text}\n\n")
            count += 1

except FileNotFoundError:
    print(f"\n❌ File not found: {INPUT_FILE}")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Unexpected error:")
    print(e)
    sys.exit(1)

print("\n✅ All JSON lines are valid.")
print(f"📄 Total entries written: {count}")
print(f"💾 Saved to: {OUTPUT_FILE}")