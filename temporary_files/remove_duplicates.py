import os
import sqlite3
import hashlib
from itertools import islice

import orjson
from tqdm import tqdm

ROOT_DIR = "FT_Data"
CHUNK_SIZE = 5000


def iter_jsonl_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".jsonl") and not name.startswith("RD_"):
                yield os.path.join(dirpath, name)


def count_lines_fast(path):
    total = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            total += block.count(b"\n")
    return total


def dedupe_file(input_path):
    folder = os.path.dirname(input_path)
    base = os.path.basename(input_path)
    output_path = os.path.join(folder, f"RD_{base}")
    db_path = output_path + ".seen.sqlite"

    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(db_path):
        os.remove(db_path)

    total_lines = count_lines_fast(input_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)")
    conn.commit()

    kept = 0
    removed = 0
    malformed_kept = 0

    try:
        with open(input_path, "rb") as fin, open(output_path, "w", encoding="utf-8") as fout:
            pbar = tqdm(total=total_lines, desc=os.path.relpath(input_path, ROOT_DIR), unit="lines")
            while True:
                chunk = list(islice(fin, CHUNK_SIZE))
                if not chunk:
                    break

                for raw in chunk:
                    pbar.update(1)

                    stripped = raw.strip()
                    if not stripped:
                        malformed_kept += 1
                        fout.write(raw.decode("utf-8", errors="replace"))
                        kept += 1
                        continue

                    try:
                        obj = orjson.loads(stripped)

                        if not isinstance(obj, dict):
                            raise ValueError("root is not dict")

                        msgs = obj["messages"]
                        if not isinstance(msgs, list) or len(msgs) != 2:
                            raise ValueError("messages is not a 2-item list")

                        user = msgs[0]["content"]
                        assistant = msgs[1]["content"]

                        if not isinstance(user, str) or not isinstance(assistant, str):
                            raise ValueError("content is not string")

                        key = hashlib.sha256(
                            (user + "\n" + assistant).encode("utf-8")
                        ).hexdigest()

                        cur.execute("INSERT OR IGNORE INTO seen (h) VALUES (?)", (key,))
                        if cur.rowcount == 1:
                            fout.write(raw.decode("utf-8", errors="replace"))
                            kept += 1
                        else:
                            removed += 1

                    except Exception:
                        # Keep malformed lines unchanged so data is not lost.
                        malformed_kept += 1
                        fout.write(raw.decode("utf-8", errors="replace"))
                        kept += 1

                conn.commit()

            pbar.close()

    finally:
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)

    return output_path, kept, removed, malformed_kept


def main():
    files = list(iter_jsonl_files(ROOT_DIR))

    if not files:
        print(f"No .jsonl files found under: {ROOT_DIR}")
        return

    total_kept = 0
    total_removed = 0
    total_malformed = 0

    for path in tqdm(files, desc="Files", unit="file"):
        output_path, kept, removed, malformed_kept = dedupe_file(path)

        total_kept += kept
        total_removed += removed
        total_malformed += malformed_kept

        print(
            f"\nDONE: {path}\n"
            f"  saved : {output_path}\n"
            f"  kept  : {kept}\n"
            f"  removed duplicates : {removed}\n"
            f"  malformed kept     : {malformed_kept}\n"
        )

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total kept             : {total_kept}")
    print(f"Total duplicates removed: {total_removed}")
    print(f"Total malformed kept    : {total_malformed}")


if __name__ == "__main__":
    main()