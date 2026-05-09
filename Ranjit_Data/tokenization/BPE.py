# hf_tokenizer_pipeline.py
# My Own
import os
import re
import ast
import sys
import spacy
import logging
import pandas as pd
from tqdm import tqdm       #shows a ProgressBar
from ast import literal_eval
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# === CONFIGURATION ===
DATA_PATH = r"Ranjit_Data/real_data/cc-100/filtered_cc-100_assamese_text_corpora.txt"
SEP = '|'                                # Dataframe separator
SPACY_MODEL = "en_core_web_trf"         # transformer-based model
# PHRASAL_VERBS = ["shut", "turn", "log"]
# PREPOSITIONS = ["down", "off", "out"]
SLANG_PATH = r"3_SLANG_ABBREVIATIONS.txt"
TOKENIZER_PATH = r"REPORTS/hf_tokenizer.json"
CORPUS_PATH = r"hf_tokenizer_corpus.txt"
VOCAB_SIZE = 50000

# === SETUP LOGGING ===
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# === LOAD SLANG MAPPING ===
def load_slang(path):
    """
    Loads a dict from a slang file that defines SLANG_ABBREVIATIONS = {...} [ eg: "u": "you" | "ur": "your" | "r": "are" | "b4": "before"]
    Safely strips comments and extracts the dictionary block.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Remove comment lines
        no_comments = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith('#')
        )

        # Extract dictionary using regex
        match = re.search(r'=\s*(\{.*\})', no_comments, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find dictionary definition in {path!r}")

        dict_block = match.group(1)
        slang_dict = literal_eval(dict_block)

        logging.info(f"Loaded {len(slang_dict)} slang mappings from {path}")
        return slang_dict

    except Exception as e:
        logging.error(f"Failed to load slang mappings from {path}: {e}")
        return {}




SLANG_ABBREVIATIONS = load_slang(SLANG_PATH)
# === SLANG NORMALIZATION ===
def normalize_slang(text):              #   normalize_slang("please close yt") | normalize_slang("u": "you","ur": "your","r": "are","b4": "before",) 
    return " ".join(SLANG_ABBREVIATIONS.get(word.lower(), word) for word in text.split())
from token_embedding import RobertaEmbeddingExtractor
model = RobertaEmbeddingExtractor()

model.run()
    # === LOAD DATA ===
def load_data():    
    try:
        df = pd.read_csv(DATA_PATH, sep=SEP, names=["utterance", "intent"], header=None, dtype=str)
        df = df.dropna(subset=["utterance", "intent"]).reset_index(drop=True)
        logging.info(f"Loaded {len(df)} records from {DATA_PATH}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # === INITIALIZE SPACY ===
    try:
        nlp = spacy.load(SPACY_MODEL)
        logging.info(f"Loaded spaCy model: {SPACY_MODEL}")
    except Exception as e:
        logging.error(f"Could not load spaCy model {SPACY_MODEL}: {e}")
        sys.exit(1)

    # === PREPROCESS & BUILD CORPUS (only once) ===
    if not os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, 'w', encoding='utf-8') as corpus_file:       #r"/home/ranjit/Desktop/Decision_Making_Model/hf_tokenizer_corpus.txt"
            for utterance in tqdm(df['utterance'], desc="Preprocessing"):   # Shows a ProgressBar  32%|██████████▍                      | 8.89G/27.9G [00:42<01:31, 223MB/s]
                normalized = normalize_slang(utterance)             #   normalize_slang("please close yt")
                doc = nlp(normalized)
                tokens = []
                for token in doc:
                    tokens.append(token.text)
                corpus_file.write(" ".join(tokens) + "\n")
        logging.info(f"Built corpus at {CORPUS_PATH}")
    else:
        logging.info(f"Corpus already exists at {CORPUS_PATH}, skipping preprocessing.")

    # === TRAIN HUGGING FACE TOKENIZER ===
    if os.path.exists(TOKENIZER_PATH):
        logging.info(f"Tokenizer already exists at {TOKENIZER_PATH}. Skipping training.")
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
            show_progress=True,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        )
        tokenizer.train([CORPUS_PATH], trainer)
        tokenizer.save(TOKENIZER_PATH)
        logging.info(f"Saved tokenizer to {TOKENIZER_PATH}")

# === CLEANUP (optional) ===
# os.remove(CORPUS_PATH)
# logging.info(f"Removed temporary corpus file: {CORPUS_PATH}")
