import re

input_path = r"Ranjit_Data/messy_data.txt"
output_path = r"Ranjit_Data/cleaned.txt"


try:
    with open(input_path,"r", encoding="utf-8") as f:
        text = f.read()
        # print(fff)
except:
    print("\n\nChange the file name        - Ranjit Das\n\n")
    



    # remove English letters, digits, and symbols
try:
    cleaned = re.sub(r"[A-Za-z0-9@#$%':\-]", "", text)
    # remove empty parentheses like (), ( ), (   )
    # remove empty brackets and any space before them
    cleaned = re.sub(r"\s*\(\s*\)\s*", " ", cleaned)
    # optionally, also remove space before closing bracket ') ' → ')'
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    # replace multiple spaces with a single space
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # replace space + two dots + space with nothing
    cleaned = re.sub(r"\s\.\.\s", "", cleaned)
    # remove 2 or more dots with optional spaces around them
    cleaned = re.sub(r"\s*\.{2,}\s*", " ", cleaned)
    # remove empty quotes like "   "
    cleaned = re.sub(r'"\s*"', "", cleaned)
    # replace '( ' with '('
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    # replace '()' with ''
    cleaned = re.sub(r"\(\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # remove leading/trailing spaces per line + empty lines
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    final_text = "\n".join(lines)

except:
    print("\nHello Honny, Name error in the first cleaning.       -Ranjit Das\n")




try:
    with open(output_path,"a",encoding="utf-8") as f:
        f.write(final_text)
    print(f"\nCleaning process done\nAnd saved to {output_path}")
except:
    print("\nChange File name.       -Ranjit Das\n")
