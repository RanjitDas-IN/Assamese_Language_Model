"""
A fast, memory-efficient word counter for large text corpora that streams files in chunks, shows real-time progress, and outputs human-readable statistics ideal for analyzing massive datasets.
"""


import os
import codecs
from tqdm import tqdm

FILE_PATH = "Ranjit_Data/real_data/IndicCorpV2_AIBharat/as.txt"
FILE_PATHS = ["as.txt", "as2.txt", "as3.txt"]
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

def count_words(path):
    total_bytes = os.path.getsize(path)
    word_count = 0
    carry = ""

    decoder = codecs.getincrementaldecoder("utf-8")()

    with open(path, "rb") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Counting Tokens") as pbar:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            pbar.update(len(chunk))
            text = decoder.decode(chunk)

            if not text:
                continue

            text = carry + text

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

        # flush decoder
        tail = decoder.decode(b"", final=True)
        if tail:
            text = carry + tail
        else:
            text = carry

        if text.strip():
            word_count += len(text.split())

    return word_count


def human_readable(n):
    if n >= 1_000_000_000_000_000:
        return f"{n/1_000_000_000_000_000:.1f}P"
    elif n >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.1f}T"
    elif n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(n)
    
    
    
#---------------------------------- For single file ---------------------------------------------------
if __name__ == "__main__":
    total = count_words(FILE_PATH)
    print(f"Total Tokens: {human_readable(total)} ({total:,})")
    
    
    
#---------------------------------- For multiple files ---------------------------------------------------

# if __name__ == "__main__":
#     grand_total = 0

#     for path in FILE_PATHS:
#         total = count_words(path)
#         grand_total += total

        # print(f"{path} → {human_readable(total)} ({total:,} Tokens)")

#     print("-" * 40)
#     print(f"TOTAL → {human_readable(grand_total)} ({grand_total:,} Tokens)")