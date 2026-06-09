# -------------------------Write the first 100 lins of a txt to another txt---------------------------
import os
target_file= "data/AsRED/cleanned_social_media_post.txt"
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
print("\n\nDone")
