#-------------------------find all .txt files inside nested folders-----------------------------
# from pathlib import Path
# root_dir = Path("data")


# txt_files = [file.name for file in root_dir.rglob("*.txt")]
# print("[")
# for files in txt_files:
#     print(f'"{files}",')
# print("]")


#-----------------------shards (.bin) testing (converting to original words)-------------------------------
# import numpy as np
# from tokenizers import Tokenizer

# # Load tokenizer
# tokenizer = Tokenizer.from_file(
#     "The_Assamese_Tokenizer/assamese_tokenizer/tokenizer.json"
# )

# # Load token IDs
# tokens = np.fromfile(
#     "token_shards/test/test_001.bin",
#     dtype=np.uint16
# )
# print(tokens[:100])
# # Save token_id : decoded_token
# with open("decoded_tokens.txt", "w", encoding="utf-8") as f:
#     for token_id in tokens[:100]:
#         token_text = tokenizer.decode([int(token_id)])
#         f.write(f"{token_id}: {token_text}\n")
