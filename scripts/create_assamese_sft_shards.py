import os
import re
import json
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# ============================================================
# Config
# ============================================================
ROOT_DIR = Path("FT_Data")
OUTPUT_DIR = Path("FT_token_shards")
TOKENIZER_PATH = Path("The_Assamese_Tokenizer/assamese_tokenizer/tokenizer.json")

CONTEXT_LENGTH = 1024
SHARD_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB
STATE_VERSION = 5

UINT16_DTYPE = np.uint16
LABEL_DTYPE = np.int32

# Fallbacks if tokenizer metadata is incomplete
PAD_FALLBACK = 1
BOS_FALLBACK = 2
EOS_FALLBACK = 3
UNK_FALLBACK = 0


# ============================================================
# Helpers
# ============================================================
def iter_jsonl_files(root_dir: Path):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".jsonl"):
                yield Path(dirpath) / name


def strip_bos_eos(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\[BOS\]\s*", "", text)
    text = re.sub(r"\s*\[EOS\]$", "", text)
    return text.strip()


def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH),
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )

    bos_id = tok.bos_token_id if tok.bos_token_id is not None else BOS_FALLBACK
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else EOS_FALLBACK
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else PAD_FALLBACK
    unk_id = tok.unk_token_id if tok.unk_token_id is not None else UNK_FALLBACK

    return tok, bos_id, eos_id, pad_id, unk_id


def fit_question_answer(q_ids, a_ids, max_len, bos_id, eos_id, pad_id):
    """
    Build: [BOS] question [EOS] answer [EOS]
    Labels: answer-only loss with -100 on prompt/padding.
    """
    budget = max_len - 3  # BOS + EOS + EOS
    if budget <= 0:
        raise ValueError(f"context_length={max_len} too small")

    # Keep as much question as possible; if too long, keep its tail and
    # preserve at least some answer tokens.
    if len(q_ids) + len(a_ids) > budget:
        if len(q_ids) >= budget:
            q_keep = max(1, budget // 2)
            a_keep = max(0, budget - q_keep)
            q_ids = q_ids[-q_keep:]
            a_ids = a_ids[:a_keep]
        else:
            a_ids = a_ids[: budget - len(q_ids)]

    seq = [bos_id] + q_ids + [eos_id] + a_ids + [eos_id]

    # answer-only labels
    labels = [-100] * (1 + len(q_ids) + 1) + a_ids + [eos_id]

    if len(seq) < max_len:
        pad_n = max_len - len(seq)
        seq += [pad_id] * pad_n
        labels += [-100] * pad_n
    else:
        seq = seq[:max_len]
        labels = labels[:max_len]

    return np.asarray(seq, dtype=UINT16_DTYPE), np.asarray(labels, dtype=LABEL_DTYPE)


def parse_record(obj):
    """Expect a 2-message JSONL sample: user then assistant."""
    if not isinstance(obj, dict):
        return None

    messages = obj.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return None

    try:
        user_msg = messages[0]
        assistant_msg = messages[1]

        if user_msg.get("role") != "user":
            return None
        if assistant_msg.get("role") != "assistant":
            return None

        q = user_msg.get("content", "")
        a = assistant_msg.get("content", "")
        if not isinstance(q, str) or not isinstance(a, str):
            return None

        q = strip_bos_eos(q)
        a = strip_bos_eos(a)
        return q, a
    except Exception:
        return None


def flush_shard(shard_idx, input_rows, label_rows, stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = OUTPUT_DIR / f"train_{shard_idx:06d}_input_ids.bin"
    label_path = OUTPUT_DIR / f"train_{shard_idx:06d}_labels.bin"

    input_arr = np.asarray(input_rows, dtype=UINT16_DTYPE)
    label_arr = np.asarray(label_rows, dtype=LABEL_DTYPE)

    input_arr.tofile(input_path)
    label_arr.tofile(label_path)

    shard_info = {
        "shard_index": shard_idx,
        "input_file": input_path.name,
        "label_file": label_path.name,
        "num_examples": int(input_arr.shape[0]),
        "input_bytes": int(input_arr.nbytes),
        "label_bytes": int(label_arr.nbytes),
        "total_bytes": int(input_arr.nbytes + label_arr.nbytes),
    }
    stats["shards"].append(shard_info)
    return shard_info


# ============================================================
# Main
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer, bos_id, eos_id, pad_id, unk_id = load_tokenizer()
    vocab_size = len(tokenizer)

    print("Tokenizer loaded")
    print("  vocab_size:", vocab_size)
    print("  bos_token_id:", bos_id)
    print("  eos_token_id:", eos_id)
    print("  pad_token_id:", pad_id)
    print("  unk_token_id:", unk_id)

    jsonl_files = sorted(list(iter_jsonl_files(ROOT_DIR)))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under {ROOT_DIR}")

    stats = {
        "state_version": STATE_VERSION,
        "context_length": CONTEXT_LENGTH,
        "shard_size_bytes": SHARD_SIZE_BYTES,
        "input_dtype": "uint16",
        "label_dtype": "int32",
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_len": vocab_size,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "files": [],
        "shards": [],
        "total_examples": 0,
        "skipped_examples": 0,
        "total_tokenized_chars": 0,
    }

    current_input_rows = []
    current_label_rows = []
    current_bytes = 0
    shard_idx = 0

    pbar_files = tqdm(jsonl_files, desc="Files", unit="file")

    for file_path in pbar_files:
        file_info = {
            "path": str(file_path),
            "examples": 0,
            "skipped": 0,
        }
        stats["files"].append(file_info)

        with open(file_path, "rb") as fin:
            line_bar = tqdm(desc=file_path.name, unit="line", leave=False)
            for raw in fin:
                line_bar.update(1)

                raw = raw.strip()
                if not raw:
                    file_info["skipped"] += 1
                    stats["skipped_examples"] += 1
                    continue

                try:
                    obj = orjson.loads(raw)
                except Exception:
                    file_info["skipped"] += 1
                    stats["skipped_examples"] += 1
                    continue

                parsed = parse_record(obj)
                if parsed is None:
                    file_info["skipped"] += 1
                    stats["skipped_examples"] += 1
                    continue

                question, answer = parsed
                if not question or not answer:
                    file_info["skipped"] += 1
                    stats["skipped_examples"] += 1
                    continue

                # Tokenize without specials; we add BOS/EOS manually.
                q_ids = tokenizer.encode(question, add_special_tokens=False)
                a_ids = tokenizer.encode(answer, add_special_tokens=False)

                if not q_ids or not a_ids:
                    file_info["skipped"] += 1
                    stats["skipped_examples"] += 1
                    continue

                input_ids, labels = fit_question_answer(
                    q_ids=q_ids,
                    a_ids=a_ids,
                    max_len=CONTEXT_LENGTH,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )

                current_input_rows.append(input_ids)
                current_label_rows.append(labels)
                current_bytes += input_ids.nbytes + labels.nbytes

                file_info["examples"] += 1
                stats["total_examples"] += 1
                stats["total_tokenized_chars"] += len(question) + len(answer)

                # Automatic flush at ~1GB.
                if current_bytes >= SHARD_SIZE_BYTES:
                    flush_shard(shard_idx, current_input_rows, current_label_rows, stats)
                    print(
                        f"\nSaved shard {shard_idx:06d} | "
                        f"examples={len(current_input_rows)} | "
                        f"bytes={current_bytes:,}"
                    )
                    shard_idx += 1
                    current_input_rows = []
                    current_label_rows = []
                    current_bytes = 0

            line_bar.close()

    # Final flush
    if current_input_rows:
        flush_shard(shard_idx, current_input_rows, current_label_rows, stats)
        print(
            f"\nSaved shard {shard_idx:06d} | "
            f"examples={len(current_input_rows)} | "
            f"bytes={current_bytes:,}"
        )
        shard_idx += 1

    stats["num_shards"] = shard_idx

    manifest = {
        "state_version": stats["state_version"],
        "context_length": stats["context_length"],
        "shard_size_bytes": stats["shard_size_bytes"],
        "input_dtype": stats["input_dtype"],
        "label_dtype": stats["label_dtype"],
        "tokenizer_path": stats["tokenizer_path"],
        "tokenizer_len": stats["tokenizer_len"],
        "bos_token_id": stats["bos_token_id"],
        "eos_token_id": stats["eos_token_id"],
        "pad_token_id": stats["pad_token_id"],
        "unk_token_id": stats["unk_token_id"],
        "num_shards": stats["num_shards"],
        "total_examples": stats["total_examples"],
        "skipped_examples": stats["skipped_examples"],
        "total_tokenized_chars": stats["total_tokenized_chars"],
        "files": stats["files"],
        "shards": stats["shards"],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Manifest:", manifest_path)
    print("Output directory:", OUTPUT_DIR.resolve())
    print("Shards created:", shard_idx)
    print("Total examples:", stats["total_examples"])
    print("Skipped examples:", stats["skipped_examples"])


if __name__ == "__main__":
    main()
