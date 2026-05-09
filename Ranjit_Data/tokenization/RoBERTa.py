import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer


print("Happening....")
# Load dataset
# df = pd.read_csv(csv_path, sep='|')
# assert 'utterance' in df.columns, "CSV must contain an 'utterance' column."

# Initialize RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

encoded = tokenizer(
        "hello dear",
        padding= True,
        # padding= 'longest',
        truncation=True,
        max_length=80,
        return_tensors='pt'
    )


# Prepare token tensors
tokens = {
    'input_ids': encoded['input_ids'],
    'attention_mask': encoded['attention_mask']
}

# Save tokenized data
save_path = os.path.expanduser(r"roberta_tokens.pt")
torch.save(tokens, save_path)
print(f"Tokenization complete. Saved tensors to {save_path}")