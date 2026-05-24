## -------------------------Write the first 100 lins of a txt to another txt---------------------------
# import os
# target_file= "/home/ranjit/Desktop/projects/Laguage_Model/data/ai4bharat_sangraha_dataset/synthetic/wiki_asm_Beng_0002_of_0063.txt"
# output_file = "first_100_lines_of_" + os.path.basename(target_file)
# # print(output_file)
# with open(target_file, encoding="utf-8") as f, \
#      open(output_file, "w", encoding="utf-8") as out:
#     for i in range(500):
#         line = f.readline()
#         if not line:
#             break
#         out.write(line)
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
