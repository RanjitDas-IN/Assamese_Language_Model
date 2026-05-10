#-------------------------Write the first 100 lins of a txt to another txt---------------------------
# with open("data/rahular_varta_DailyHuntDataset/scrapped_text.txt", encoding="utf-8") as f, \
#      open("first_100_lines.txt", "w", encoding="utf-8") as out:
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



#-------------------------parquet to txt-----------------------------
# import pandas as pd

# df = pd.read_parquet("as-00000-of-00001.parquet")

# with open("nuralnets_multingual_tinystories.txt", "w", encoding="utf-8") as f:
#     for row in df["text"]:
#         f.write(str(row) + "\n")