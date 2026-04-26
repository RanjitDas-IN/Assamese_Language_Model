#-------------------------------------------------------Write the first 100 lins of a txt to another txt----------------------------------------------------------------------------------


with open("Ranjit_Data/real_data/MWire-Labs-assamese-monolingual-corpus/assamese_monolingual_sentences_final_cleaned.csv", encoding="utf-8") as f, \
     open("first_100_lines.txt", "w", encoding="utf-8") as out:
    for i in range(500):
        line = f.readline()
        if not line:
            break
        out.write(line)
print("\n\nDone")

#-------------------------------------------------------print what is inside the txt file----------------------------------------------------------------------------------
# with open("Ranjit_Data/real_data/1B_assamese_Tokens_Quwn3/cleanned_1B_as_tokens_unfiltered.txt", encoding="utf-8") as f:
#     for _ in range(20):
#         print(f.readline(), end="")





