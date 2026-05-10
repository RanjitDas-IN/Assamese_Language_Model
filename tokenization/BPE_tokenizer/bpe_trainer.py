"""
train_assamese_tokenizer.py
============================
BPE tokenizer training pipeline for Assamese GPT-style LM.
Streams directly from SQLite — no temp files.
"""

import logging
import os
import sqlite3
import time

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH       = "LLM_sqlite3.db"
TABLE_NAME    = "raw_token"
COLUMN_NAME   = "text"
OUTPUT_DIR    = "./assamese_tokenizer_output"

VOCAB_SIZE    = 50_000
MIN_FREQUENCY = 2
MAX_LENGTH    = 2048
BATCH_SIZE    = 50_000
MAX_ROWS      = None   # None = full dataset | 10_000 / 50_000 / 100_000 for testing

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite streaming generator
# ---------------------------------------------------------------------------

def iter_texts():
    """
    Streams text rows from SQLite in BATCH_SIZE chunks.
    Stops early if MAX_ROWS is set.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur  = conn.cursor()

    cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    total = cur.fetchone()[0]
    limit = MAX_ROWS if MAX_ROWS else total
    log.info("Total rows: %d | Training on: %d", total, limit)

    cur.execute(f"SELECT {COLUMN_NAME} FROM {TABLE_NAME}")

    yielded = 0
    while yielded < limit:
        rows = cur.fetchmany(BATCH_SIZE)
        if not rows:
            break
        for row in rows:
            if yielded >= limit:
                break
            text = row[0]
            if text and text.strip():
                yield text.strip()
                yielded += 1
        log.info("  … %d / %d rows streamed", yielded, limit)

    conn.close()

# ---------------------------------------------------------------------------
# Build tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Strip(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder       = decoders.BPEDecoder(suffix="</w>")

    return tokenizer

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=special_tokens,
        show_progress=True,
        end_of_word_suffix="</w>",
    )

    log.info("Starting BPE training (vocab_size=%d, min_frequency=%d) …", VOCAB_SIZE, MIN_FREQUENCY)
    t0 = time.time()
    tokenizer.train_from_iterator(iter_texts(), trainer=trainer)
    log.info("Training done in %.1f seconds.", time.time() - t0)
    log.info("Final vocab size: %d", tokenizer.get_vocab_size())

    return tokenizer

# ---------------------------------------------------------------------------
# Post-processor
# ---------------------------------------------------------------------------

def attach_post_processor(tokenizer: Tokenizer) -> Tokenizer:
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
    return tokenizer

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_tokenizer(tokenizer: Tokenizer) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    core_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer.save(core_path)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=core_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=MAX_LENGTH,
    )
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    log.info("Tokenizer saved to: %s", OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------

SAMPLE_SENTENCES = [
    "অসমীয়া ভাষা এটি সুন্দৰ ভাষা।",
    "ভাৰতৰ উত্তৰ-পূৱ অঞ্চলত অসম ৰাজ্য অৱস্থিত।",
    "ব্ৰহ্মপুত্ৰ নদীৰ পাৰত অসমৰ সভ্যতা গঢ়ি উঠিছে।",
]

def run_tests() -> None:
    log.info("Running sanity tests …")
    fast_tok = PreTrainedTokenizerFast.from_pretrained(OUTPUT_DIR)
    fast_tok.model_max_length = MAX_LENGTH

    for sent in SAMPLE_SENTENCES:
        enc     = fast_tok(sent, truncation=True, max_length=MAX_LENGTH)
        ids     = enc["input_ids"]
        tokens  = fast_tok.convert_ids_to_tokens(ids)
        decoded = fast_tok.decode(ids, skip_special_tokens=True)
        ok      = decoded.strip() == sent.strip()

        print("\n" + "─" * 56)
        print(f"  {'✅' if ok else '⚠️ MISMATCH'}")
        print(f"  Input   : {sent}")
        print(f"  Tokens  : {tokens}")
        print(f"  Decoded : {decoded}")

    print("\n" + "─" * 56)
    print(f"  Vocab size : {fast_tok.vocab_size}")
    print(f"  BOS        : {fast_tok.bos_token!r} (id={fast_tok.bos_token_id})")
    print(f"  EOS        : {fast_tok.eos_token!r} (id={fast_tok.eos_token_id})")
    print(f"  PAD        : {fast_tok.pad_token!r} (id={fast_tok.pad_token_id})")
    print(f"  UNK        : {fast_tok.unk_token!r} (id={fast_tok.unk_token_id})")
    print("─" * 56)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("DB: %s | Table: %s | MAX_ROWS: %s", DB_PATH, TABLE_NAME, MAX_ROWS or "ALL")

    tokenizer = build_tokenizer()
    tokenizer = train_tokenizer(tokenizer)
    tokenizer = attach_post_processor(tokenizer)
    save_tokenizer(tokenizer)
    run_tests()

    log.info("Done.")

if __name__ == "__main__":
    main()