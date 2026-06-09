

import json
from pathlib import Path

INPUT_DIR = Path("extracted_aswikisource")
OUTPUT_DIR = Path("data/aswiki_encyclopedia")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import re

def clean_text(text: str) -> str:
    # Remove templates
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.S)

    # Remove file/image markup
    text = re.sub(r"\[\[(?:চিত্ৰ|File|ফাইল):[^\]]+\]\]", " ", text, flags=re.I)

    # Keep display text from wiki links:
    # [[রাজাবাই ক্লক টাওৱাৰ]] -> ৰাজাবাই ক্লক টাওৱাৰ
    # [[A|B]] -> B
    text = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", text)

    # Remove external links but keep label text
    text = re.sub(r"\[(?:https?://|www\.)[^\s\]]+\s([^\]]+)\]", r"\1", text, flags=re.I)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove English / Latin words
    text = re.sub(r"[A-Za-z]+(?:[-’'][A-Za-z]+)*", " ", text)

    # Remove Arabic-script text like سلامی جمہوریۂ پاکستان
    text = re.sub(r"[\u0600-\u06FF]+", " ", text)

    # Remove leftover junk symbols, keep Assamese/Bengali block, digits, punctuation, spaces
    text = re.sub(r"[^\u0980-\u09FF0-9\s\.\,\;\:\!\?\-\(\)\'\"“”‘’/]", " ", text)

    # Collapse whitespace
    return " ".join(text.split())

processed_files = 0
processed_articles = 0

for input_file in INPUT_DIR.rglob("wiki_*"):

    rel_path = input_file.relative_to(INPUT_DIR)
    output_file = OUTPUT_DIR / rel_path.with_suffix(".txt")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
                
                text = obj.get("text", "").strip()
                text = " ".join(text.split())
                text = clean_text(text)
                text = re.sub(r"\s*\(\s*\)\s*", " ", text)
                text = re.sub(r"\s+\)", ")", text)
                text = re.sub(r"[ \t]{2,}", " ", text)
                text = re.sub(r"\s\.\.\s", "", text)
                text = re.sub(r'"\s*"', "", text)
                text = re.sub(r"\(\)", "", text)
                

                if not text:
                    continue

                fout.write("[BOS]\n")
                fout.write(text)
                fout.write("\n[EOS]\n\n")

                processed_articles += 1

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in {input_file}")

    processed_files += 1

print(f"Processed files: {processed_files}")
print(f"Processed articles: {processed_articles}")
print(f"Saved to: {OUTPUT_DIR}")