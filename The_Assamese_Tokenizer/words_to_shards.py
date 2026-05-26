#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, logging, os, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import numpy as np
from tokenizers import Tokenizer

SHARD_SIZE_BYTES = 1 * 1024 * 1024 * 1024
UINT16_DTYPE = np.dtype(np.uint16)
UINT16_MAX = np.iinfo(np.uint16).max
UINT16_BYTES = UINT16_DTYPE.itemsize
DEFAULT_BUFFER_MB = 32
DEFAULT_CHARS_PER_CHUNK = 1_000_000
DEFAULT_LARGE_FILE_THRESHOLD = 500 * 1024 * 1024
STATE_VERSION = 4

@dataclass
class ShardRecord:
    index: int
    filename: str
    split: str
    tokens: int = 0
    bytes_written: int = 0
    contributing_source_files: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class StateManager:
    def __init__(self, path: Path): self.path = path
    def load(self) -> dict:
        if not self.path.exists():
            return {"version": STATE_VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "completed_files": {}, "writers": {}, "stats": {"files_processed": 0, "tokens_written": 0}, "in_progress": None}
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION: raise RuntimeError("State version mismatch; use a fresh state file")
        return state
    def save(self, state: dict) -> None:
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(self.path)

class ShardWriter:
    def __init__(self, split: str, output_dir: Path, state: dict, rolling_buffer_bytes: int):
        self.split, self.output_dir, self.state = split, output_dir, state
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_buffer_capacity_tokens = max(1, rolling_buffer_bytes // UINT16_BYTES)
        self.rolling_buffer = np.empty(self.rolling_buffer_capacity_tokens, dtype=UINT16_DTYPE)
        self.rolling_count = 0
        self.state.setdefault("writers", {}).setdefault(split, {})
        self._bootstrap_or_recover()
    @property
    def shard_token_capacity(self) -> int: return SHARD_SIZE_BYTES // UINT16_BYTES
    def _bootstrap_or_recover(self) -> None:
        ws = self.state["writers"][self.split]
        if not ws: ws.update({"current_index": 0, "tokens_in_current_shard": 0, "records": []})
        self.current_index, self.tokens_in_current_shard = int(ws["current_index"]), int(ws["tokens_in_current_shard"])
        self.records = ws["records"]
        self.current_path = self.output_dir / f"{self.split}_{self.current_index:03d}.bin"
        if self.current_path.exists():
            size = self.current_path.stat().st_size
            aligned = (size // UINT16_BYTES) * UINT16_BYTES
            if aligned != size:
                with self.current_path.open('r+b') as f: f.truncate(aligned)
            self.tokens_in_current_shard = aligned // UINT16_BYTES
        else:
            self.tokens_in_current_shard = 0
        self._sync_state()
    def _sync_state(self) -> None:
        ws = self.state["writers"][self.split]
        ws["current_index"], ws["tokens_in_current_shard"], ws["records"] = self.current_index, self.tokens_in_current_shard, self.records
    def _ensure_record(self, idx: int) -> dict:
        while len(self.records) <= idx:
            i = len(self.records)
            self.records.append(ShardRecord(index=i, filename=f"{self.split}_{i:03d}.bin", split=self.split).__dict__)
        return self.records[idx]
    def _flush_rolling(self) -> int:
        if self.rolling_count == 0: return 0
        with self.current_path.open('ab') as f:
            self.rolling_buffer[:self.rolling_count].tofile(f)
            f.flush(); os.fsync(f.fileno())
        w = self.rolling_count
        self.tokens_in_current_shard += w
        rec = self._ensure_record(self.current_index)
        rec["tokens"], rec["bytes_written"], rec["updated_at"] = self.tokens_in_current_shard, self.tokens_in_current_shard * UINT16_BYTES, datetime.now(timezone.utc).isoformat()
        self.rolling_count = 0
        self._sync_state()
        return w
    def checkpoint_flush(self) -> None:
        self._flush_rolling()
    def append_tokens(self, token_ids: List[int]) -> Tuple[int, Set[int]]:
        if not token_ids: return 0, set()
        arr = np.asarray(token_ids, dtype=np.int64)
        if arr.min(initial=0) < 0 or arr.max(initial=0) > UINT16_MAX: raise ValueError("Token IDs exceed uint16 range")
        arr16 = arr.astype(UINT16_DTYPE, copy=False)
        cursor, total, touched = 0, arr16.size, set()
        while cursor < total:
            if self.tokens_in_current_shard >= self.shard_token_capacity:
                self._flush_rolling(); self.current_index += 1
                self.current_path = self.output_dir / f"{self.split}_{self.current_index:03d}.bin"
                self.tokens_in_current_shard = 0; self._sync_state()
            touched.add(self.current_index)
            take = min(total - cursor, self.shard_token_capacity - self.tokens_in_current_shard, self.rolling_buffer_capacity_tokens - self.rolling_count)
            self.rolling_buffer[self.rolling_count:self.rolling_count+take] = arr16[cursor:cursor+take]
            self.rolling_count += take; cursor += take
            if self.rolling_count == self.rolling_buffer_capacity_tokens or self.tokens_in_current_shard + self.rolling_count >= self.shard_token_capacity:
                self._flush_rolling()
        return total, touched
    def mark_source_file_for_shards(self, shard_indices: Set[int]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for idx in shard_indices:
            rec = self._ensure_record(idx); rec["contributing_source_files"] += 1; rec["updated_at"] = now
        self._sync_state()
    def finalize(self) -> None: self._flush_rolling()

def discover_txt_files_stream(root: Path) -> Iterator[Path]:
    for p in root.rglob('*.txt'):
        if p.is_file(): yield p

def relative_key(path: Path, root: Path) -> str:
    try: rel = path.resolve().relative_to(root.resolve())
    except Exception: rel = path.name
    return rel.as_posix()

def split_for_key(rel_key: str, val_ratio: float) -> str:
    tr, vr = 1.0 - 3.0 * val_ratio, val_ratio
    if tr <= 0: raise ValueError("Invalid val_ratio")
    h = hashlib.sha256(rel_key.encode('utf-8')).digest()
    n = int.from_bytes(h[:8], 'big') / float(2**64 - 1)
    if n < tr: return 'train'
    if n < tr + vr: return 'val'
    return 'test'

def stream_text_chunks_bounded(path: Path, chars_per_chunk: int, start_offset: int = 0) -> Iterator[Tuple[int, str]]:
    # bounded memory: fixed-size reads, then yield on newline boundary when possible
    with path.open('r', encoding='utf-8', errors='replace') as f:
        if start_offset: f.seek(start_offset)
        carry = ''
        while True:
            part = f.read(chars_per_chunk)
            if not part:
                if carry: yield f.tell(), carry
                break
            data = carry + part
            # Prefer newline boundary when available,
            # otherwise fall back safely to fixed-size boundary.
            split_at = data.rfind('\n')

            if split_at == -1 or split_at < chars_per_chunk // 2:
                emit = data[:chars_per_chunk]
                carry = data[chars_per_chunk:]
                checkpoint = f.tell() - len(carry)
                yield checkpoint, emit
            else:
                emit = data[:split_at + 1]
                carry = data[split_at + 1:]
                checkpoint = f.tell() - len(carry)
                yield checkpoint, emit

def tokenizer_adds_special_tokens(tokenizer: Tokenizer) -> bool:
    t = tokenizer.encode("অসমীয়া").tokens
    return bool(t) and (t[0] == '[BOS]' or t[-1] == '[EOS]')

def write_metadata(output_root: Path, state: dict, tokenizer_json: Path, val_ratio: float) -> None:
    writers, split_counts = state.get('writers', {}), {}
    for split, w in writers.items():
        recs = w.get('records', [])
        split_counts[split] = {"num_shards": len(recs), "total_tokens": sum(r.get('tokens', 0) for r in recs), "total_bytes": sum(r.get('bytes_written', 0) for r in recs), "total_contributing_source_files": sum(r.get('contributing_source_files', 0) for r in recs)}
    meta = {"created_at": datetime.now(timezone.utc).isoformat(), "state_version": STATE_VERSION, "shard_size_bytes": SHARD_SIZE_BYTES, "dtype": "uint16", "dtype_bytes": UINT16_BYTES, "tokenizer_json": str(tokenizer_json), "split_rule": {"train": 1 - 3 * val_ratio, "val": val_ratio, "test": 2 * val_ratio}, "stats": state.get('stats', {}), "splits": split_counts, "shards": writers}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'dataset_index.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, default=Path('data'))
    p.add_argument('--output-root', type=Path, default=Path('./token_shards'))
    p.add_argument('--tokenizer-json', type=Path, default=Path('The_Assamese_Tokenizer/assamese_tokenizer/tokenizer.json'))
    p.add_argument('--state-file', type=Path, default=Path('./preprocess_state.json'))
    p.add_argument('--val-ratio', type=float, default=0.05)
    p.add_argument('--chars-per-chunk', type=int, default=DEFAULT_CHARS_PER_CHUNK)
    p.add_argument('--rolling-buffer-mb', type=int, default=DEFAULT_BUFFER_MB)
    p.add_argument('--sleep-seconds', type=float, default=3.0)
    p.add_argument('--large-file-threshold-bytes', type=int, default=DEFAULT_LARGE_FILE_THRESHOLD)
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s | %(levelname)s | %(message)s')
    input_root = args.input_root.resolve()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    auto_special, eos_id, bos_id = tokenizer_adds_special_tokens(tokenizer), tokenizer.token_to_id('[EOS]'), tokenizer.token_to_id('[BOS]')
    if eos_id is None: raise ValueError('[EOS] token missing')
    state_mgr, state = StateManager(args.state_file), StateManager(args.state_file).load()
    completed: Dict[str, dict] = state.setdefault('completed_files', {})
    writers = {s: ShardWriter(s, args.output_root / s, state, int(args.rolling_buffer_mb * 1024 * 1024)) for s in ('train', 'val', 'test')}

    for idx, path in enumerate(discover_txt_files_stream(input_root), start=1):
        rel_key = relative_key(path, input_root)
        if rel_key in completed: continue
        in_prog = state.get('in_progress')
        if in_prog and in_prog.get('rel_key') == rel_key:
            split, start_offset = in_prog['split'], int(in_prog.get('offset', 0))
            touched_shards, file_tokens = set(in_prog.get('touched_shards', [])), int(in_prog.get('tokens', 0))
        else:
            split, start_offset, touched_shards, file_tokens = split_for_key(rel_key, args.val_ratio), 0, set(), 0
            state['in_progress'] = {"rel_key": rel_key, "split": split, "offset": 0, "tokens": 0, "touched_shards": []}
            state_mgr.save(state)
        writer, size = writers[split], path.stat().st_size
        logging.info('[%d] %s | split=%s | resume_offset=%d', idx, rel_key, split, start_offset)
        try:
            for checkpoint_offset, chunk in stream_text_chunks_bounded(path, args.chars_per_chunk, start_offset):
                ids = tokenizer.encode(chunk).ids
                if auto_special:
                    if bos_id is not None and ids and ids[0] == bos_id: ids = ids[1:]
                    if ids and ids[-1] == eos_id: ids = ids[:-1]
                n, touched = writer.append_tokens(ids)
                writer.checkpoint_flush()  # crash-safe: persist tokens before persisting file offset
                file_tokens += n; touched_shards.update(touched)
                state['in_progress'] = {"rel_key": rel_key, "split": split, "offset": checkpoint_offset, "tokens": file_tokens, "touched_shards": sorted(touched_shards)}
                state_mgr.save(state)
            n_eos, touched_eos = writer.append_tokens([eos_id]); writer.checkpoint_flush()
            file_tokens += n_eos; touched_shards.update(touched_eos)
            writer.mark_source_file_for_shards(touched_shards)
            completed[rel_key] = {"split": split, "tokens": file_tokens, "size_bytes": size, "completed_at": datetime.now(timezone.utc).isoformat()}
            state['in_progress'] = None
            state['stats']['files_processed'] = state['stats'].get('files_processed', 0) + 1
            state['stats']['tokens_written'] = state['stats'].get('tokens_written', 0) + file_tokens
            state_mgr.save(state)
        except Exception:
            logging.exception('Failed: %s', rel_key)
            state_mgr.save(state)
        if size >= args.large_file_threshold_bytes: time.sleep(args.sleep_seconds)

    for w in writers.values(): w.finalize()
    state_mgr.save(state)
    write_metadata(args.output_root, state, args.tokenizer_json, args.val_ratio)

if __name__ == '__main__':
    main()
