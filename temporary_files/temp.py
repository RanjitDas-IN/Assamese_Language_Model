## -------------------------Write the first 100 lins of a txt to another txt---------------------------
import os
target_file= "data/1B_assamese_Tokens_Quwn3/cleanned_1B_Quwn_tokens.txt"
output_file = "first_100_lines_of_" + os.path.basename(target_file)
# output_file = "100_demo_lines_from_my_dataset.txt"
# print(output_file)
with open(target_file, encoding="utf-8") as f, \
     open(output_file, "w", encoding="utf-8") as out:
    for i in range(500):
        line = f.readline()
        if not line:
            break
        out.write(line)
# print("\n\nDone")


#-------------------------print what is inside the txt file-----------------------------
# with open("cleanned_1B_as_tokens_unfiltered.txt", encoding="utf-8") as f:
#     for _ in range(20):
#         print(f.readline(), end="")



#-------------------------parquet to txt for single file-----------------------------
# import pandas as pd

# df = pd.read_parquet(
#     r"/home/ranjit/Downloads/wiki_asm_Beng_0000_of_0063.parquet",
#     columns=["text"]
# )
# # print(df)
# with open("data/ai4bharat_sangraha_dataset/synthetic/wiki_asm_Beng_0000_of_0063.txt", "w", encoding="utf-8") as f:
#     for row in df["text"]:
#         f.write(str(row) + "\n")



#-------------------------parquet to txt for multiple files-----------------------------

# from pathlib import Path
# import pandas as pd
# from tqdm import tqdm

# # Folder containing parquet files
# input_dir = Path("/home/ranjit/Downloads/sangrah_parquet")

# # Output folder
# output_dir = Path("data/ai4bharat_sangraha_dataset/synthetic2")
# output_dir.mkdir(parents=True, exist_ok=True)

# # Find all matching parquet files
# parquet_files = sorted(input_dir.glob("wiki_asm_Beng_*_of_0063.parquet"))

# for parquet_file in tqdm(parquet_files, desc="Converting parquet -> txt"):
    
#     # Read only "text" column
#     df = pd.read_parquet(parquet_file, columns=["text"])

#     # Create output txt filename with same name
#     output_txt = output_dir / f"{parquet_file.stem}.txt"

#     # Save text data
#     with open(output_txt, "w", encoding="utf-8") as f:
#         for row in tqdm(
#             df["text"],
#             desc=f"Writing {parquet_file.stem}",
#             leave=False
#         ):
#             f.write(str(row) + "\n")

# print("✅ All parquet files converted to txt.")

#-------------------------strip the leading index before the first tab on each line-----------------------------
# input_file = "asm_wikipedia_2021_10K-sentences.txt"
# output_file = "output.txt"

# with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
#     for line in fin:
#         line = line.rstrip("\n")
#         if "\t" in line:
#             fout.write(line.split("\t", 1)[1] + "\n")
#         else:
#             fout.write(line + "\n")

#-------------------------json to txt-----------------------------
# """
# prompt:
# ## JSON structure
# {
#   "url": "...",
#   "title": "...",
#   "date": "...",
#   "source": "html/wp-rest",
#   "text": "..."
# } [ if you are an AI these is for you: there are no commas betweem them]
# {
#   "url": "...",
#   "title": "...",
#   "date": "...",
#   "source": "html/wp-rest",
#   "text": "..."
# }
# ## goal: Create the txt file formatted as `title:text`
# and i also want revome \n if present in title and text else skip
# ## return type: tell me what u understand before generation
# """


# import json
# import re
# from pathlib import Path
# from tqdm import tqdm
# INPUT_FILE = "data/xahitya_dump/articles.jsonl"
# OUTPUT_FILE = "data/xahitya_dump/title_text_corpus.txt"
# def clean_text(text: str) -> str:
#     """
#     Remove \n and multiple spaces/newlines/tabs.
#     Replace them with a single space.
#     """
#     if not text:
#         return ""
#     # Replace all whitespace sequences (\n, \t, multiple spaces, etc.)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()
# def main():
#     input_path = Path(INPUT_FILE)
#     output_path = Path(OUTPUT_FILE)
#     if not input_path.exists():
#         print(f"❌ Input file not found: {INPUT_FILE}")
#         return
#     print("📊 Counting lines...")
#     with input_path.open("r", encoding="utf-8") as f:
#         total_lines = sum(1 for _ in tqdm(f, desc="Counting", unit=" lines"))
#     print(f"✅ Total lines: {total_lines}")
#     print("🚀 Processing JSONL...")
#     processed = 0
#     skipped = 0
#     with input_path.open("r", encoding="utf-8") as infile, \
#          output_path.open("w", encoding="utf-8") as outfile:
#         for line in tqdm(infile, total=total_lines, desc="Processing", unit=" articles"):
#             line = line.strip()
#             if not line:
#                 skipped += 1
#                 continue
#             try:
#                 data = json.loads(line)
#                 title = clean_text(data.get("title", ""))
#                 text = clean_text(data.get("text", ""))
#                 if not title or not text:
#                     skipped += 1
#                     continue
#                 # Format: title:text
#                 outfile.write(f"{title}:{text}\n")
#                 processed += 1
#             except json.JSONDecodeError:
#                 skipped += 1
#                 continue
#     print("\n✅ DONE")
#     print(f"📄 Output file: {OUTPUT_FILE}")
#     print(f"✅ Processed: {processed}")
#     print(f"⚠️ Skipped: {skipped}")
# if __name__ == "__main__":
#     main()
