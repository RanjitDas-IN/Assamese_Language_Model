# import pandas as pd
# df = pd.read_csv("FT_Data/who_is_nisha.psv", sep = "|")
# print(df.shape)
# print(df.columns)

import os
import json
import math
import random
from tqdm import tqdm

ROOT_DIR = "FT_Data"
NISHA_PATH = "data/who_is_nisha.jsonl"
OUTPUT_PREFIX = "RN_"
INSERT_FRACTION = 0.90
SEED = 42

random.seed(SEED)


def load_nisha_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                # Keep the full JSON line exactly as-is
                json.loads(raw)
                examples.append(raw)
            except Exception:
                continue
    return examples


def iter_jsonl_files(root_dir):
    nisha_abs = os.path.abspath(NISHA_PATH)
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.lower().endswith(".jsonl"):
                continue
            if name.startswith(OUTPUT_PREFIX):
                continue
            full_path = os.path.abspath(os.path.join(dirpath, name))
            if full_path == nisha_abs:
                continue
            yield full_path


def count_lines_fast(path):
    total = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            total += block.count(b"\n")
    return total


def make_insert_positions(total_lines, insert_count):
    """
    Spread inserts across the file with small randomness.
    Positions mean: insert before original line index `pos`.
    """
    if insert_count <= 0:
        return []

    if total_lines <= 0:
        return [0] * insert_count

    # Evenly spaced base positions
    base_positions = [
        int(round(i * total_lines / insert_count))
        for i in range(insert_count)
    ]

    # Add small jitter so it is not perfectly regular
    jitter_window = max(1, total_lines // max(insert_count * 10, 1))
    positions = []
    for p in base_positions:
        jitter = random.randint(-jitter_window, jitter_window)
        pos = max(0, min(total_lines, p + jitter))
        positions.append(pos)

    positions.sort()
    return positions


def process_file(input_path, nisha_examples):
    output_path = os.path.join(
        os.path.dirname(input_path),
        OUTPUT_PREFIX + os.path.basename(input_path)
    )

    total_lines = count_lines_fast(input_path)
    insert_count = math.ceil(len(nisha_examples) * INSERT_FRACTION)

    # Randomly choose at least 90% of Nisha examples for this file
    chosen = random.sample(nisha_examples, k=insert_count)
    random.shuffle(chosen)

    insert_positions = make_insert_positions(total_lines, insert_count)

    inserted = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        pbar = tqdm(total=total_lines, desc=os.path.relpath(input_path, ROOT_DIR), unit="lines")

        current_line_index = 0
        insert_idx = 0

        for line in fin:
            # Insert all scheduled Nisha lines before this original line
            while insert_idx < len(insert_positions) and insert_positions[insert_idx] <= current_line_index:
                fout.write(chosen[insert_idx] + "\n")
                inserted += 1
                insert_idx += 1

            fout.write(line)
            kept += 1
            current_line_index += 1
            pbar.update(1)

        # Insert anything remaining at the end
        while insert_idx < len(insert_positions):
            fout.write(chosen[insert_idx] + "\n")
            inserted += 1
            insert_idx += 1

        pbar.close()

    return output_path, kept, inserted


def main():
    nisha_examples = load_nisha_examples(NISHA_PATH)
    if not nisha_examples:
        print(f"No valid examples found in {NISHA_PATH}")
        return

    print(f"Loaded Nisha examples into RAM: {len(nisha_examples)}")

    jsonl_files = list(iter_jsonl_files(ROOT_DIR))
    if not jsonl_files:
        print(f"No .jsonl files found under {ROOT_DIR}")
        return

    summary = []

    for path in tqdm(jsonl_files, desc="Files", unit="file"):
        out_path, kept, inserted = process_file(path, nisha_examples)
        summary.append((path, out_path, kept, inserted))

    print("\nDone.")
    for src, dst, kept, inserted in summary:
        print(f"{src} -> {dst} | kept={kept} inserted={inserted}")


if __name__ == "__main__":
    main()