import sqlite3
import sys
import os
from tqdm import tqdm

_Table_Name = "raw_token"
_DB_File: str = "LLM_sqlite3.db"

def txt_to_sql(txt_file: str):
    """
    Reads a .txt file and stores each non-empty line as a row in a SQLite database.
    Args:
        txt_file:   Path to the input .txt file.
    """
    if not os.path.exists(txt_file):
        print(f"Error: file '{txt_file}' not found.")
        sys.exit(1)

    conn = sqlite3.connect(_DB_File)
    cursor = conn.cursor()

    # Create table with a single text column
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {_Table_Name} (
            text TEXT NOT NULL
        )
    """)

    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]  # skip blank lines

    rows = [(line,) for line in tqdm(lines, desc="Inserting rows", unit="row")]

    cursor.executemany(
        f"INSERT INTO {_Table_Name} (text) VALUES (?)",
        rows,
    )
    conn.commit()
    inserted = cursor.rowcount
    conn.close()
    print(f"Done! Inserted {inserted} row(s) into '{_Table_Name}' in '{_DB_File}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python txt_to_sql.py <input.txt> [output.db] [table_name]")
        print("  input.txt   – required: path to the text file")
        sys.exit(1)

    txt_file = sys.argv[1]
    txt_to_sql(txt_file)