"""
A fast, memory-efficient word counter for large text corpora that streams files in chunks, shows real-time progress, and outputs human-readable statistics ideal for analyzing massive datasets.
"""


import os
import codecs
from tqdm import tqdm

FILE_PATH = "asm_wikipedia_2021_10K-sentences.txt"

# FILE_PATHS = ["Ranjit_Data/real_data/DailyHunt_Assamese_News_Dataset/cleaned.txt", 
#               "Ranjit_Data/real_data/DailyHunt_Assamese_News_Dataset/poem.txt",
#               "Ranjit_Data/real_data/DailyHunt_Assamese_News_Dataset/day2_data.txt",
#               "Ranjit_Data/real_data/DailyHunt_Assamese_News_Dataset/day3_data.txt",
            #   "Ranjit_Data/real_data/rahular_varta_DailyHuntDataset/scrapped_text.txt",
            #   "Ranjit_Data/real_data/MWire-Labs-assamese-monolingual-corpus/assamese_monolingual_sentences_final_cleaned.csv",
            #   "Ranjit_Data/real_data/IndicCorpV2_AIBharat/as.txt",
            #   "Ranjit_Data/real_data/cc-100/filtered_cc-100_assamese_text_corpora.txt",
            #   "Ranjit_Data/real_data/1B_assamese_Tokens_Quwn3/cleanned_1B_Quwn_tokens.txt"             
            #   ]
            
            
            
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


def making_tokens_human_readable(n):
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
    print(f"Total Tokens: {making_tokens_human_readable(total)} ({total:,})")
    
    
    
#---------------------------------- For multiple files ---------------------------------------------------

# if __name__ == "__main__":
#     grand_total = 0

#     for path in FILE_PATHS:
#         total = count_words(path)
#         grand_total += total

#         print(f"{path} → {making_tokens_human_readable(total)} ({total:,} Tokens)")

#     print("-" * 40)
#     print(f"\nTOTAL → {making_tokens_human_readable(grand_total)} ({grand_total:,} Tokens)")
    
    
    
    
    
# Ranjit_Data/real_data/poem.txt → 79.6K (79,629 Tokens)
# Ranjit_Data/real_data/cleaned.txt → 1.9M (1,918,169 Tokens)
# Ranjit_Data/real_data/day2_data.txt → 2.6M (2,593,961 Tokens)
# Ranjit_Data/real_data/day3_day3.txt → 1.7M (1,710,903 Tokens)
# Ranjit_Data/real_data/IndicCorpV2_AIBharat/as.txt → 67.3M (67,269,308 Tokens)
# (venv) [ranjit@arch Laguage_Model]$ python Ranjit_Data/scripts/token_counter.py 
# Ranjit_Data/real_data/cc-100/filtered_cc-100_assamese_text_corpora.txt → 3.9M (3,923,910 Tokens)
# Ranjit_Data/real_data/rahular_varta_DailyHuntDataset/scrapped_text.txt → 184.2M (184,172,812 Tokens)
# Ranjit_Data/real_data/1B_assamese_Tokens_Quwn3/cleanned_1B_Quwn_tokens.txt → 124.7M (124,658,768 Tokens)
# Ranjit_Data/real_data/MWire-Labs-assamese-monolingual-corpus/assamese_monolingual_sentences_final_cleaned.csv → 38.7M (38,670,822 Tokens)
# ----------------------------------------

# TOTAL → 425.0M (424,998,282 Tokens)





# human_readable(1_200)                # 1.2K
# human_readable(5_000_000)            # 5.0M
# human_readable(3_200_000_000)        # 3.2B
# human_readable(7_000_000_000_000)    # 7.0T
# human_readable(9_000_000_000_000_000)# 9.0P