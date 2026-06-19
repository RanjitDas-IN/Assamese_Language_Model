
import os
import codecs
from tqdm import tqdm
from pathlib import Path


# ---------------------------- CONFIG ----------------------------

root_dir = Path("data")

# Find all .txt files recursively
# FILE_PATHS = list(root_dir.rglob("*.txt"))
FILE_PATHS = list(root_dir.rglob("*_bos_eos.txt"))
print(f"Total Files: {len(FILE_PATHS)}")


CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


# ---------------------------- WORD COUNTER ----------------------------

def count_words(path):
    total_bytes = os.path.getsize(path)

    word_count = 0
    carry = ""

    decoder = codecs.getincrementaldecoder("utf-8")()

    with open(path, "rb") as f, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"Counting {path.name}"
    ) as pbar:

        while True:
            chunk = f.read(CHUNK_SIZE)

            if not chunk:
                break

            pbar.update(len(chunk))

            text = decoder.decode(chunk)

            if not text:
                continue

            text = carry + text

            # Handle partial words across chunks
            if text and not text[-1].isspace():
                parts = text.split()

                if parts:
                    carry = parts.pop()
                else:
                    carry = text
                    continue
            else:
                parts = text.split()
                carry = ""

            word_count += len(parts)

        # Flush decoder
        tail = decoder.decode(b"", final=True)

        if tail:
            text = carry + tail
        else:
            text = carry

        if text.strip():
            word_count += len(text.split())

    return word_count


# ---------------------------- HUMAN READABLE ----------------------------

def making_tokens_human_readable(n):
    if n >= 1_000_000_000_000_000:
        return f"{n / 1_000_000_000_000_000:.1f}P"

    elif n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f}T"

    elif n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"

    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"

    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"

    else:
        return str(n)


# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":

    if not FILE_PATHS:
        print("No .txt files found.")
        exit()

    grand_total = 0

    for path in FILE_PATHS:

        total = count_words(path)

        grand_total += total

        print(
            f"{path} → "
            f"{making_tokens_human_readable(total)} "
            f"({total:,} Tokens)"
        )

    print("-" * 50)

    print(
        f"TOTAL → "
        f"{making_tokens_human_readable(grand_total)} "
        f"({grand_total:,} Tokens)"
    )