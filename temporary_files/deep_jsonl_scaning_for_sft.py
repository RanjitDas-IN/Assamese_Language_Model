
"""
Deep JSONL dataset scanner for SFT datasets.

Checks:
1. Every line is valid JSON.
2. Top-level object must contain only "messages".
3. messages must be a list of exactly length 2.
4. roles must be ["user", "assistant"].
5. content must be strings.
6. content must not be empty.
7. content must start with [BOS] and end with [EOS].
8. Exact duplicate samples across ALL files.
9. Produces scanning_report.txt.
10. Saves random samples for manual inspection.

pip install orjson
"""

import os
import random
import hashlib
from collections import defaultdict
import orjson

ROOT_DIR = "FT_Data"
REPORT_FILE = "scanning_report.txt"
SAMPLE_FILE = "manual_samples.txt"
N_RANDOM_SAMPLES = 3000

issues = []
duplicates = defaultdict(list)
samples = []

total_files = 0
total_lines = 0
total_examples = 0


def report(msg):
    issues.append(msg)


for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if not file.lower().endswith(".jsonl"):
            continue

        total_files += 1
        path = os.path.join(root, file)

        with open(path, "rb") as f:
            for line_no, line in enumerate(f, start=1):

                total_lines += 1

                try:
                    obj = orjson.loads(line)
                except Exception as e:
                    report(f"{path}:{line_no} -> Invalid JSON ({e})")
                    continue

                if not isinstance(obj, dict):
                    report(f"{path}:{line_no} -> Root object not dict")
                    continue

                if set(obj.keys()) != {"messages"}:
                    report(
                        f"{path}:{line_no} -> top-level keys={list(obj.keys())}"
                    )

                messages = obj.get("messages")

                if not isinstance(messages, list):
                    report(f"{path}:{line_no} -> messages is not list")
                    continue

                if len(messages) != 2:
                    report(
                        f"{path}:{line_no} -> messages length={len(messages)}"
                    )
                    continue

                expected_roles = ["user", "assistant"]

                contents = []

                for i, msg in enumerate(messages):

                    if not isinstance(msg, dict):
                        report(f"{path}:{line_no} -> message[{i}] not dict")
                        continue

                    if set(msg.keys()) != {"role", "content"}:
                        report(
                            f"{path}:{line_no} -> message[{i}] keys={list(msg.keys())}"
                        )

                    role = msg.get("role")
                    content = msg.get("content")

                    if role != expected_roles[i]:
                        report(
                            f"{path}:{line_no} -> message[{i}] role='{role}' expected='{expected_roles[i]}'"
                        )

                    if not isinstance(content, str):
                        report(
                            f"{path}:{line_no} -> message[{i}] content type={type(content).__name__}"
                        )
                        continue

                    if len(content.strip()) == 0:
                        report(
                            f"{path}:{line_no} -> message[{i}] empty content"
                        )

                    if not content.startswith("[BOS]"):
                        report(
                            f"{path}:{line_no} -> message[{i}] missing BOS"
                        )

                    if not content.endswith("[EOS]"):
                        report(
                            f"{path}:{line_no} -> message[{i}] missing EOS"
                        )

                    contents.append(content)

                if len(contents) == 2:

                    key = hashlib.sha256(
                        (contents[0] + "\n" + contents[1]).encode("utf-8")
                    ).hexdigest()

                    duplicates[key].append(f"{path}:{line_no}")

                    total_examples += 1

                    if len(samples) < N_RANDOM_SAMPLES:
                        samples.append(line.decode("utf-8"))
                    else:
                        j = random.randint(0, total_examples - 1)
                        if j < N_RANDOM_SAMPLES:
                            samples[j] = line.decode("utf-8")


duplicate_groups = 0
duplicate_examples = 0

for _, locs in duplicates.items():
    if len(locs) > 1:
        duplicate_groups += 1
        duplicate_examples += len(locs)

        report(
            "\nDuplicate sample:\n"
            + "\n".join(locs[:20])
        )

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("SFT DATASET SCAN REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Files scanned      : {total_files}\n")
    f.write(f"Lines scanned      : {total_lines}\n")
    f.write(f"Examples scanned   : {total_examples}\n")
    f.write(f"Duplicate groups   : {duplicate_groups}\n")
    f.write(f"Duplicate examples : {duplicate_examples}\n")
    f.write("\n")

    if issues:
        f.write("ISSUES FOUND\n")
        f.write("-" * 80 + "\n")

        for x in issues:
            f.write(x + "\n")
    else:
        f.write("NO STRUCTURAL ISSUES FOUND\n")

with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(s)

print("Done.")
print("Report:", REPORT_FILE)
print("Samples:", SAMPLE_FILE)