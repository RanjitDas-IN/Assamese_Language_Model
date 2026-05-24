## -------------------------Write the first 100 lins of a txt to another txt---------------------------
target_file= "scrapped_text.txt"
output_file= "first_100_lines_of_" + target_file
# print(output_file)
with open(target_file, encoding="utf-8") as f, \
     open(output_file, "w", encoding="utf-8") as out:
    for i in range(500):
        line = f.readline()
        if not line:
            break
        out.write(line)
print("\n\nDone")



#-------------------------print what is inside the txt file-----------------------------
# with open("cleanned_1B_as_tokens_unfiltered.txt", encoding="utf-8") as f:
#     for _ in range(20):
#         print(f.readline(), end="")



#-------------------------parquet to txt-----------------------------
# import pandas as pd

# df = pd.read_parquet("as-00000-of-00001.parquet")

# with open("nuralnets_multingual_tinystories.txt", "w", encoding="utf-8") as f:
#     for row in df["text"]:
#         f.write(str(row) + "\n")


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