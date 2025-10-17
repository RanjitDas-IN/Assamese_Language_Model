import re
import os

# Input and output file paths
input_path = r"Sosanka_Data\News\raw_news.txt"
output_path = r"Sosanka_Data\News\After_cleaning_news.txt"

try:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print(f"\nFile not found! Please check the file name or path:\n{input_path}\n")
    exit()

# --- Cleaning process ---
try:
    cleaned = text

    # Remove unwanted digits and symbols (you can add more if needed)
    cleaned = re.sub(r"[A-Za-z0-9@#$%':]", "", cleaned)

    # Remove empty parentheses like (), ( ), (   )
    cleaned = re.sub(r"\s*\(\s*\)\s*", " ", cleaned)

    # Fix spaces before or after brackets
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)

    # Replace multiple dots or spaces with a single space
    cleaned = re.sub(r"\.{2,}", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)


     # replace multiple spaces with a single space
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    
    # Remove empty quotes like "   "
    cleaned = re.sub(r'"\s*"', "", cleaned)

    # Remove leading/trailing spaces per line + empty lines
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # Add literal '/n' at the end of each line
    final_text = "\n".join(line + "\n" for line in lines)

    # Add two blank lines, then the separator
    final_text += "\n----------------------------------------------------------------------------------------------------\n\n"

except Exception as e:
    print(f"\nError during cleaning: {e}\n")
    exit()

# --- Save cleaned story ---
try:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\nStory cleaning completed successfully!\nSaved to: {output_path}\n")
except Exception as e:
    print(f"\nError while saving file: {e}\n")
