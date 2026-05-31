# -------------------------parquet to txt for single file-----------------------------
import pandas as pd

df = pd.read_parquet(
    r"wiki_asm_Beng_0000_of_0063.parquet",
    columns=["text"]
)
# print(df)
with open("data/ai4bharat_sangraha_dataset/synthetic/wiki_asm_Beng_0000_of_0063.txt", "w", encoding="utf-8") as f:
    for row in df["text"]:
        f.write(str(row) + "\n")



# -------------------------parquet to txt for multiple files-----------------------------

from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Folder containing parquet files
input_dir = Path("sangrah_parquet")

# Output folder
output_dir = Path("data/ai4bharat_sangraha_dataset/synthetic2")
output_dir.mkdir(parents=True, exist_ok=True)

# Find all matching parquet files
parquet_files = sorted(input_dir.glob("wiki_asm_Beng_*_of_0063.parquet"))

for parquet_file in tqdm(parquet_files, desc="Converting parquet -> txt"):
    
    # Read only "text" column
    df = pd.read_parquet(parquet_file, columns=["text"])

    # Create output txt filename with same name
    output_txt = output_dir / f"{parquet_file.stem}.txt"

    # Save text data
    with open(output_txt, "w", encoding="utf-8") as f:
        for row in tqdm(
            df["text"],
            desc=f"Writing {parquet_file.stem}",
            leave=False
        ):
            f.write(str(row) + "\n")

print("✅ All parquet files converted to txt.")
