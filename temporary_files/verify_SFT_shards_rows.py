
#!/usr/bin/env python3
"""
Verify that every *_input_ids.bin shard has a matching *_labels.bin shard
and that both contain the same number of rows.

Folder layout:
FT_token_shards/
    train_000000_input_ids.bin
    train_000000_labels.bin
    ...
"""

from pathlib import Path
import numpy as np

SHARD_DIR = Path("FT_token_shards")
CONTEXT_LENGTH = 1024

INPUT_DTYPE = np.uint16
LABEL_DTYPE = np.int32


def rows(path: Path, dtype):
    item_size = np.dtype(dtype).itemsize
    return path.stat().st_size // (CONTEXT_LENGTH * item_size)


input_files = sorted(SHARD_DIR.glob("*_input_ids.bin"))

if not input_files:
    raise FileNotFoundError(f"No input shards found in '{SHARD_DIR}'")

bad = False

for iid in input_files:
    lbl = iid.parent / iid.name.replace("_input_ids.bin", "_labels.bin")

    if not lbl.exists():
        print(f"[MISSING LABEL] {iid.name}")
        bad = True
        continue

    input_rows = rows(iid, INPUT_DTYPE)
    label_rows = rows(lbl, LABEL_DTYPE)

    if input_rows != label_rows:
        print(
            f"[ROW MISMATCH]\n"
            f"  {iid.name}: {input_rows:,} rows\n"
            f"  {lbl.name}: {label_rows:,} rows\n"
        )
        bad = True
    else:
        print(f"[OK] {iid.name} -> {input_rows:,} rows")

if bad:
    print("\n❌ Some shard pairs are invalid.")
else:
    print("\n✅ All shard pairs are valid.")