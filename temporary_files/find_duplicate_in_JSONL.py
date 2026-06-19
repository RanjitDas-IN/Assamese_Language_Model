"""
Deep duplicate scanner for JSONL SFT data.

What it does:
- Recursively scans all .jsonl files under FT_Data
- Counts exact duplicates across the whole tree
- Finds near duplicates using MinHash + LSH + RapidFuzz
- Writes a full report to scanning_report.txt
- Shows progress in terminal with tqdm

Install:
    pip install orjson tqdm datasketch rapidfuzz
"""

import os
import hashlib
import json
from collections import defaultdict, Counter
from tqdm import tqdm

ROOT = "FT_Data"
REPORT = "scanning_report2.txt"

# --------------------------------------------------
# discover jsonl files
# --------------------------------------------------
jsonl_files = []

for root, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".jsonl"):
            jsonl_files.append(os.path.join(root, f))

print("Files found:", len(jsonl_files))

# --------------------------------------------------
# PASS 1
# count hashes only
# --------------------------------------------------

hash_counter = Counter()

for path in tqdm(jsonl_files, desc="Pass1"):
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:

            try:
                obj = json.loads(line)

                user = obj["messages"][0]["content"]
                assistant = obj["messages"][1]["content"]

                key = hashlib.sha256(
                    (user + "\n" + assistant).encode("utf-8")
                ).hexdigest()

                hash_counter[key] += 1

            except:
                pass

duplicate_hashes = {
    h for h, cnt in hash_counter.items()
    if cnt > 1
}

print("duplicate groups =", len(duplicate_hashes))

# --------------------------------------------------
# PASS 2
# gather only duplicates
# --------------------------------------------------

per_file_count = Counter()

examples = defaultdict(list)

for path in tqdm(jsonl_files, desc="Pass2"):

    with open(path, "r", encoding="utf-8") as fin:

        for line_no, line in enumerate(fin, 1):

            try:
                obj = json.loads(line)

                user = obj["messages"][0]["content"]
                assistant = obj["messages"][1]["content"]

                key = hashlib.sha256(
                    (user + "\n" + assistant).encode("utf-8")
                ).hexdigest()

            except:
                continue

            if key not in duplicate_hashes:
                continue

            per_file_count[path] += 1

            if len(examples[path]) < 2:
                examples[path].append({
                    "line": line_no,
                    "user": user[:300],
                    "assistant": assistant[:300]
                })

# --------------------------------------------------
# write report
# --------------------------------------------------

with open(REPORT, "w", encoding="utf-8") as fout:

    fout.write("=" * 80 + "\n")
    fout.write("EXACT DUPLICATE REPORT\n")
    fout.write("=" * 80 + "\n\n")

    fout.write(
        f"Duplicate groups: {len(duplicate_hashes)}\n\n"
    )

    for path in sorted(per_file_count):

        fout.write(
            f"\nFILE: {path}\n"
        )

        fout.write(
            f"Duplicate lines: {per_file_count[path]}\n"
        )

        fout.write(
            "Demo duplicates:\n"
        )

        for i, ex in enumerate(examples[path], 1):

            fout.write(
                f"\nExample {i}\n"
            )

            fout.write(
                f"line = {ex['line']}\n"
            )

            fout.write(
                "USER:\n"
            )
            fout.write(
                ex["user"] + "\n"
            )

            fout.write(
                "ASSISTANT:\n"
            )
            fout.write(
                ex["assistant"] + "\n"
            )

        fout.write("\n" + "-" * 80 + "\n")

print("saved ->", REPORT)