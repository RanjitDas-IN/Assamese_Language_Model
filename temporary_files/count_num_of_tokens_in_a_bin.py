#-----------------------count(.bin) the num of tokens-------------------------------
import numpy as np
from pathlib import Path

base = Path("token_shards")

total_tokens = 0

for split in ["train", "test", "val"]:
    split_path = base / split

    if not split_path.exists():
        continue

    print(f"\n[{split.upper()}]")

    split_total = 0

    for bin_file in sorted(split_path.glob("*.bin")):
        tokens = np.fromfile(bin_file, dtype=np.uint16)

        count = len(tokens)
        split_total += count
        total_tokens += count

        print(f"{bin_file.name}: {count:,} tokens")

    print(f"Total {split}: {split_total:,} tokens")

print(f"\nGRAND TOTAL: {total_tokens:,} tokens")
