# #-------------------------------------------------------intellagent cleaning Script----------------------------------------------------------------------------------
# # Quick intuition:

# # if te output:
# # Cleaning dataset: 51419it [00:25, 2004.35it/s]
# # Processed → 51k lines
# # Time → 25 seconds
# # Speed → ~2000 lines/sec

# # | Term   | Meaning                    |
# # | ------ | -------------------------- |
# # | `it`   | how many items done        |
# # | `it/s` | how fast you're processing |



# import unicodedata,re
# from tqdm import tqdm
# INPUT_FILE = r"data/ai4bharat_sangraha_dataset/synthetic2/wiki_asm_Beng_0017_of_0063.txt"
# OUTPUT_FILE = r"data/ai4bharat_sangraha_dataset/synthetic2/cleanned_wiki_asm_Beng_0017_of_0063.txt"

# # Assamese/Bengali Unicode block
# def is_assamese(char):
#     return 0x0980 <= ord(char) <= 0x09FF

# def is_valid_char(char):
#     # Ignore spaces and punctuation
#     return char.isalpha()

# def assamese_ratio(line):
#     total = 0
#     assamese_count = 0

#     for ch in line:
#         if is_valid_char(ch):
#             total += 1
#             if is_assamese(ch):
#                 assamese_count += 1

#     if total == 0:
#         return 0

#     return assamese_count / total

# THRESHOLD = 0.9  # adjust: 0.6–0.8 depending on strictness

# # Count total lines (for progress bar)
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     total_lines = sum(1 for _ in f)


# with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
#      open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

#     kept = 0
#     removed = 0

#     # for line in tqdm(infile, desc="Cleaning dataset"): # use this line if you have too large file; like 10GB+
#     for line in tqdm(infile, total=total_lines, desc="Cleaning dataset"):
#         line = line.replace("\\n", " ")  # remove literal '\n'
#         line = re.sub(r"[‎]", "", line)
#         line = re.sub(r"\s*\(\s*\)\s*", " ", line)
#         line = re.sub(r"[ \t]{2,}", " ", line)
#         line = re.sub(r"\s*\.{2,}\s*", " ", line)
#         ratio = assamese_ratio(line)

#         if ratio >= THRESHOLD:
#             outfile.write(line)
#             kept += 1
#         else:
#             removed += 1

# print(f"Done. Kept: {kept}, Removed: {removed}")
# print(f"cleanned --> {INPUT_FILE}\nto --> {OUTPUT_FILE}")


# #-------------------------------------------------------blind cleaning Script----------------------------------------------------------------------------------

import re

input_path = r"data/1B_assamese_Tokens_Quwn3/cleanned_1B_Quwn_tokens.txt"
output_path = r"data/1B_assamese_Tokens_Quwn3/ultra_cleanned_1B_Quwn_tokens.txt"


def clean_text(text: str) -> str:
    # Remove wiki templates (non-nested)
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.S)

    # Remove image/file markup
    text = re.sub(
        r"\[\[(?:চিত্ৰ|ফাইল|file|image):[^\]]+\]\]",
        " ",
        text,
        flags=re.I,
    )

    # Convert wiki links:
    # [[A]] -> A
    # [[A|B]] -> B
    text = re.sub(
        r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]",
        r"\1",
        text,
    )

    # Remove external links but keep label text
    # [https://site.com label] -> label
    text = re.sub(
        r"\[(?:https?://|www\.)[^\s\]]+\s([^\]]+)\]",
        r"\1",
        text,
        flags=re.I,
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove English / Latin tokens
    text = re.sub(
        r"\b[A-Za-z]+(?:[-_][A-Za-z]+)*\b",
        " ",
        text,
    )

    # Remove Arabic-script text
    text = re.sub(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+",
        " ",
        text,
    )

    # Keep Assamese/Bengali chars, digits, and selected punctuation
    text = re.sub(
        r"[^\u0980-\u09FF0-9\s\.\,\;\:\!\?\-\(\)\'\"“”‘’/]",
        " ",
        text,
    )

    # Cleanup
    text = re.sub(r"\s*\(\s*\)\s*", " ", text)
    text = re.sub(r"\(\)", " ", text)
    text = re.sub(r'"\s*"', " ", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\(\s+", "(", text)

    # Normalize spaces while preserving newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


try:
    line_count = 0
    written_count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line_count += 1

            cleaned = clean_text(line)

            if cleaned:
                fout.write(cleaned + "\n")
                written_count += 1

    print(
        f"\nCleaning completed successfully."
        f"\nInput lines   : {line_count}"
        f"\nOutput lines  : {written_count}"
        f"\nEdited lines  : {line_count-written_count}"
        f"\nSaved to      : {output_path}"
    )

except FileNotFoundError as e:
    print(f"\nFile not found:\n{e}")

except PermissionError as e:
    print(f"\nPermission error:\n{e}")

except Exception as e:
    print(f"\nUnexpected error:\n{e}")



# #-------------------------------------------------------blind cleaning Script for all txt under a folder----------------------------------------------------------------------------------

# import re
# from pathlib import Path

# BASE_DIR = Path("data/ai4bharat_sangraha_dataset/synthetic2")


# def clean_text(text: str) -> str:
#     text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.S)

#     text = re.sub(
#         r"\[\[(?:চিত্ৰ|ফাইল|file|image):[^\]]+\]\]",
#         " ",
#         text,
#         flags=re.I,
#     )

#     text = re.sub(
#         r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]",
#         r"\1",
#         text,
#     )

#     text = re.sub(
#         r"\[(?:https?://|www\.)[^\s\]]+\s([^\]]+)\]",
#         r"\1",
#         text,
#         flags=re.I,
#     )

#     text = re.sub(r"<[^>]+>", " ", text)

#     text = re.sub(
#         r"\b[A-Za-z]+(?:[-_][A-Za-z]+)*\b",
#         " ",
#         text,
#     )

#     text = re.sub(
#         r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+",
#         " ",
#         text,
#     )

#     text = re.sub(
#         r"[^\u0980-\u09FF0-9\s\.\,\;\:\!\?\-\(\)\'\"“”‘’/]",
#         " ",
#         text,
#     )

#     text = re.sub(r"\s*\(\s*\)\s*", " ", text)
#     text = re.sub(r"\(\)", " ", text)
#     text = re.sub(r'"\s*"', " ", text)
#     text = re.sub(r"\s+\)", ")", text)
#     text = re.sub(r"\(\s+", "(", text)

#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     return text.strip()


# for idx in range(5, 18):  # 0005 -> 0017

#     input_file = (
#         BASE_DIR /
#         f"cleanned_wiki_asm_Beng_{idx:04d}_of_0063.txt"
#     )

#     output_file = (
#         BASE_DIR /
#         f"ultra_cleanned_wiki_asm_Beng_{idx:04d}_of_0063.txt"
#     )

#     if not input_file.exists():
#         print(f"SKIP: {input_file.name} not found")
#         continue

#     line_count = 0
#     written_count = 0

#     with open(input_file, "r", encoding="utf-8") as fin, \
#          open(output_file, "w", encoding="utf-8") as fout:

#         for line in fin:
#             line_count += 1

#             cleaned = clean_text(line)

#             if cleaned:
#                 fout.write(cleaned + "\n")
#                 written_count += 1

#     print(
#         f"✓ {input_file.name}\n"
#         f"  Input  : {line_count}\n"
#         f"  Output : {written_count}\n"
#         f"  Saved  : {output_file.name}\n"
#     )

# print("\nAll files processed.")