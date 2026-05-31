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


# import re

# input_path = r"first_100_lines_of_cleanned_1B_Quwn_tokens.txt"
# output_path = r"Cleanned_1B_Quwn_tokens.txt"

# try:
#     with open(input_path,"r", encoding="utf-8") as f:
#         text = f.read()
#         # print(fff)
# except:
#     print("\n\nChange the file name        - Ranjit Das\n\n")
    



# try:
#     # remove English letters, digits, and symbols
#     cleaned = re.sub(r"[A-Za-z0-9…@#$+%':\-]", "", text)
    
#     # Remove all invisible unicode characters
#     cleaned = re.sub(r"[‎]", "", cleaned)
#     # remove empty parentheses like (), ( ), (   ) remove empty brackets and any space before them
#     cleaned = re.sub(r"\s*\(\s*\)\s*", " ", cleaned)
#     # optionally, also remove space before closing bracket ') ' → ')'
#     cleaned = re.sub(r"\s+\)", ")", cleaned)
#     # replace multiple spaces with a single space
#     cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
#     # replace space + two dots + space with nothing
#     cleaned = re.sub(r"\s\.\.\s", "", cleaned)
#     # remove 2 or more dots with optional spaces around them
#     cleaned = re.sub(r"\s*\.{2,}\s*", " ", cleaned)
#     # remove empty quotes like "   "
#     cleaned = re.sub(r'"\s*"', "", cleaned)
#     # replace '( ' with '('
#     cleaned = re.sub(r"\(\s+", "(", cleaned)
#     # replace '()' with ''
#     cleaned = re.sub(r"\(\)", "", cleaned)
#     cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
#     # remove leading/trailing spaces per line + empty lines
#     lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
#     final_text = "\n".join(lines)

# except:
#     print("\nHello Honny, Name error in the first cleaning.       -Ranjit Das\n")




# try:
#     # think append mode or write mode 
#     with open(output_path,"w",encoding="utf-8") as f:
#         f.write(final_text)
#     print(f"\nCleaning process done\nAnd saved to \"{output_path}\"")
# except:
#     print("\nChange File name.       -Ranjit Das\n")




