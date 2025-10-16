import re
import os
import sys
import traceback

input_path = r"Sosanka_Data\Song\raw_song.txt"
output_path = r"Sosanka_Data\Song\After_cleaning_Song.txt"

# --- Read input file safely ---
if not os.path.isfile(input_path):
    print(f"Input file not found: {input_path}")
    sys.exit(1)

try:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
except Exception as e:
    print("Failed to read input file. Error:")
    traceback.print_exc()
    sys.exit(1)

print("Read input file successfully. Preview (first 300 chars):")
print("-" * 40)
print(text[:300].replace("\n", "\\n"))  # show \n as characters so preview is readable
print("-" * 40)

# --- Cleaning ---
try:
    # remove English letters, digits, and selected symbols
    cleaned = re.sub(r"[A-Za-z0-9@#$%':]", "", text)

    # remove empty parentheses like (), ( ), (   )
    cleaned = re.sub(r"\s*\(\s*\)\s*", " ", cleaned)

    # remove stray spaces before closing parentheses and after opening
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)

    # replace multiple spaces with a single space
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # replace patterns of multiple dots (.., ..., etc.) with single space
    cleaned = re.sub(r"\.{2,}", " ", cleaned)

    # remove empty quotes like "   "
    cleaned = re.sub(r'"\s*"', "", cleaned)

    # strip leading/trailing spaces on each line and remove empty lines
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # OPTION A (most likely what you want): ensure each line ends with a newline character '\n'
    final_text = "\n".join(lines) + "\n"

    # OPTION B (if you really wanted the literal characters "/n" at end of each line):
    # final_text = "\n".join(line + "/n" for line in lines) + "\n"

except Exception as e:
    print("Error during cleaning:")
    traceback.print_exc()
    sys.exit(1)

# --- Write output file safely ---
try:
    # create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\nCleaning process done. Saved to: {output_path}")
except Exception as e:
    print("Failed to write output file. Error:")
    traceback.print_exc()
    sys.exit(1)
