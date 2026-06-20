"""
sft_train.py
============
Supervised Fine-Tuning for নিশা (Nisha) — Assamese Conversational AI.

Base model  : checkpoint-198680   (pretrained Assamese LLM)
Platform    : Kaggle  ·  2 × NVIDIA Tesla T4 16 GB
Precision   : FP16 mixed-precision (PyTorch AMP)
Framework   : Hugging Face Transformers 4.52.4  ·  PyTorch
Architecture: LLaMA-style decoder  (RMSNorm · RoPE · SDPA · MHA)

নিশাৰ বিষয়ে (About Nisha)
────────────────────────────
Nisha is a warm, friendly, and occasionally playful Assamese AI assistant.
She supports normal conversation, creative writing, mathematics, and
logical reasoning.  When uncertain, she says so honestly.
"""

import os
import sys
import json
import math
import time
import bisect
import logging
import warnings
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import transformers
from transformers import (
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# ── Suppress noisy but harmless HF warnings ───────────────────────────────
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────────────────────────────────────
# PATHS  —  edit these to match your Kaggle working-directory layout
# ─────────────────────────────────────────────────────────────────────────────
BASE_CHECKPOINT = "/kaggle/working/outputs"         # pretrained weights; never modified
TOKENIZER_PATH  = "tokenizer.json"             # Assamese HF tokenizer JSON
SHARD_DIR       = "FT_token_shards"            # directory of .bin shard pairs
MANIFEST_PATH   = f"{SHARD_DIR}/manifest.json" # shard inventory
HPARAMS_PATH    = "hparams.txt"                # single source of truth for hp

LOG_DIR  = Path("sft_logs")
CKPT_DIR = Path("sft_checkpoints")
OUT_DIR = Path("sft_outputs")

for _d in (LOG_DIR, CKPT_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
_log_ts   = time.strftime("%Y%m%d_%H%M%S")
_log_file = LOG_DIR / f"sft_{_log_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Log file : {_log_file.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# HPARAMS PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_hparams(path: str = HPARAMS_PATH) -> Dict[str, str]:
    """
    Parse a flat 'key = value' config file.

    Rules:
      · Lines that start with '#' (after stripping) are comments / headers.
      · Blank lines are ignored.
      · Duplicate keys: last occurrence wins  (handles hparams.txt's repeated
        'dtype = FP16' entry safely).
      · Bare words with no '=' are silently skipped  (e.g. 'CrossEntropyLoss').
    """
    hp: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"hparams.txt not found at '{path}'.  "
            "Ensure the file is in the working directory."
        )
    with open(p, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                k, v = key.strip(), val.strip()
                if k and v:
                    hp[k] = v          # last-write-wins for duplicates
    return hp


def _cast(val: Optional[str], default, cast_fn):
    """Safe type-cast with fallback.  Returns *default* if val is None/empty."""
    if val is None or str(val).strip() == "":
        return default
    try:
        return cast_fn(str(val).strip())
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# SFT DATASET
# ─────────────────────────────────────────────────────────────────────────────
class SFTShardDataset(Dataset):
    """
    Memory-efficient SFT dataset over pre-tokenised binary shard pairs.

    Shard layout
    ────────────
    Each shard consists of two files:
      *_input_ids.bin   shape=(N, context_length)  dtype=uint16
      *_labels.bin      shape=(N, context_length)  dtype=int32

    Row i is one complete, ready-to-train sample.  No packing, padding, or
    re-tokenisation is performed — shards are consumed as-is.

    Label semantics  (CRITICAL — do not alter)
    ───────────────────────────────────────────
    labels[i] stores -100 at question / padding positions and the real token
    id at answer positions.  LlamaForCausalLM performs the causal shift
    internally before computing cross-entropy:

        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        loss = CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)

    We must NOT pre-shift or recompute labels.

    Scalability design
    ──────────────────
    • O(S) RAM overhead (S = number of shards), not O(total_samples).
      A cumulative-count array lets us map any global_idx → (shard, row)
      via bisect in O(log S) without materialising a flat tuple list.

    • Per-worker mmap cache.
      DataLoader worker processes inherit a fork of the Dataset object.
      Each worker populates its own _iid_cache / _lbl_cache dict on first
      access and reuses the open handles for subsequent samples from the
      same shard — eliminating per-sample open() overhead entirely.
    """

    # Dtypes that match the on-disk binary layout
    INPUT_DTYPE = np.uint16
    LABEL_DTYPE = np.int32

    def __init__(
        self,
        shard_dir: str      = SHARD_DIR,
        manifest_path: str  = MANIFEST_PATH,
        context_length: int = 1024,
    ):
        super().__init__()
        self.shard_dir      = Path(shard_dir)
        self.context_length = context_length

        # Parallel lists: index i → (input_path, label_path, n_samples)
        self._input_paths: List[Path] = []
        self._label_paths: List[Path] = []
        self._n_per_shard: List[int]  = []

        self._discover_shards(manifest_path)

        if not self._input_paths:
            raise FileNotFoundError(
                f"No shard pairs found in '{shard_dir}'.  "
                "Check manifest.json or shard file naming conventions."
            )

        # ── Cumulative sample counts — O(S+1) integer memory ─────────────
        # cumulative[i] = total samples in shards [0 … i-1].
        # Used by bisect to resolve any global_idx in O(log S).
        self._cumulative: List[int] = [0]
        for n in self._n_per_shard:
            self._cumulative.append(self._cumulative[-1] + n)

        # ── Per-worker mmap caches (lazily populated, fork-safe) ──────────
        self._iid_cache: Dict[int, np.memmap] = {}
        self._lbl_cache: Dict[int, np.memmap] = {}

        n_total = self._cumulative[-1]
        idx_ram = (len(self._n_per_shard) + 1) * 8   # bytes for cumulative list
        logger.info(
            f"  SFT Dataset : {len(self._input_paths)} shard pair(s)  "
            f"→ {n_total:,} samples  "
            f"(index overhead: {idx_ram} bytes)"
        )

    # ── Shard discovery ───────────────────────────────────────────────────

    def _discover_shards(self, manifest_path: str):
        """
        Populate parallel shard lists from manifest.json or glob fallback.

        Supported manifest formats
        ──────────────────────────
        A) Dict-list:
           {"shards": [{"input_ids": "fname.bin", "labels": "fname.bin",
                        "num_samples": N}, ...]}

        B) String-list (prefix-based):
           {"shards": ["train_000000", "train_000001", ...]}
           → auto-appends _input_ids.bin / _labels.bin

        C) Absent / empty manifest:
           → glob *_input_ids.bin in shard_dir (sorted for determinism)
        """
        mp = Path(manifest_path)
        entries: list = []

        if mp.exists():
            with open(mp, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            entries = manifest.get("shards", [])
        else:
            logger.warning(
                f"  Manifest not found at '{manifest_path}' — "
                "auto-discovering shards via glob."
            )

        if entries:
            for entry in entries:
                if isinstance(entry, dict):
                    iid = self.shard_dir / (
                        entry.get("input_ids")
                        or entry.get("input_file")
                    )
                    lbl = self.shard_dir / (
                        entry.get("labels")
                        or entry.get("label_file")
                    )
        
                    n = (
                        int(
                            entry.get("num_samples")
                            or entry.get("num_examples")
                        )
                        if (
                            "num_samples" in entry
                            or "num_examples" in entry
                        )
                        else self._rows_from_file(iid, self.INPUT_DTYPE)
                    )
        
                elif isinstance(entry, str):
                    # Prefix form: "train_000000" → "train_000000_input_ids.bin"
                    iid = self.shard_dir / f"{entry}_input_ids.bin"
                    lbl = self.shard_dir / f"{entry}_labels.bin"
                    n = self._rows_from_file(iid, self.INPUT_DTYPE)
        
                else:
                    logger.warning(
                        f"  Unrecognised manifest entry type: {type(entry)} — skipping"
                    )
                    continue
                
                if iid.exists() and lbl.exists():
                    self._input_paths.append(iid)
                    self._label_paths.append(lbl)
                    self._n_per_shard.append(n)
                else:
                    logger.warning(
                        f"  Shard file(s) missing: {iid} / {lbl} — skipping"
                    )
        
        else:
            # Glob fallback — sorted so shard order is deterministic across runs
            for iid in sorted(self.shard_dir.glob("*_input_ids.bin")):
                stem = iid.name.replace("_input_ids.bin", "")
                lbl = iid.parent / f"{stem}_labels.bin"
        
                if lbl.exists():
                    n = self._rows_from_file(iid, self.INPUT_DTYPE)
                    self._input_paths.append(iid)
                    self._label_paths.append(lbl)
                    self._n_per_shard.append(n)
                else:
                    logger.warning(
                        f"  No matching labels file for {iid.name} — skipping"
                    )

    def _rows_from_file(self, path: Path, dtype) -> int:
        """
        Compute row count N of a (N, context_length) binary file from file size.
        No data is read; only stat() is called.
        """
        item_bytes = np.dtype(dtype).itemsize
        total_bytes = path.stat().st_size
        return total_bytes // (self.context_length * item_bytes)

    # ── Mmap cache ────────────────────────────────────────────────────────

    def _get_mmaps(self, shard_idx: int) -> Tuple[np.memmap, np.memmap]:
        """
        Return the (input_ids_mmap, labels_mmap) pair for *shard_idx*.

        Opens and caches on first access within this worker process.
        Subsequent calls for the same shard_idx return the cached handle —
        no syscall overhead.  Safe across workers because each worker
        process has its own heap (fork semantics).
        """
        if shard_idx not in self._iid_cache:
            n = self._n_per_shard[shard_idx]
            self._iid_cache[shard_idx] = np.memmap(
                self._input_paths[shard_idx],
                dtype=self.INPUT_DTYPE,
                mode="r",
                shape=(n, self.context_length),
            )
            self._lbl_cache[shard_idx] = np.memmap(
                self._label_paths[shard_idx],
                dtype=self.LABEL_DTYPE,
                mode="r",
                shape=(n, self.context_length),
            )
        return self._iid_cache[shard_idx], self._lbl_cache[shard_idx]

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ── O(log S) global → local mapping ──────────────────────────────
        # bisect_right(cumulative, idx) returns the first position p where
        # cumulative[p] > idx; subtracting 1 gives the owning shard.
        shard_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_row = idx - self._cumulative[shard_idx]

        iid_mm, lbl_mm = self._get_mmaps(shard_idx)

        # .astype() copies the mmap view into process memory and widens to
        # int64, which PyTorch's embedding table and CrossEntropyLoss require.
        input_ids = torch.from_numpy(iid_mm[local_row].astype(np.int64))
        labels    = torch.from_numpy(lbl_mm[local_row].astype(np.int64))

        # Return labels EXACTLY as stored.
        # -100 positions are ignored by CrossEntropyLoss.
        # LlamaForCausalLM shifts (input_ids, labels) internally.
        # DO NOT pre-shift here.
        return {"input_ids": input_ids, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
class SFTMetricsCallback(TrainerCallback):
    """
    Log evaluation loss and perplexity to console and a JSONL file.

    Output: outputs/sft_metrics.jsonl  (one JSON object per eval step;
    easy to post-process with pandas or jq).
    """

    _metrics_file = OUT_DIR / "sft_metrics.jsonl"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            # Cap before exp() to avoid OverflowError on large losses
            ppl = math.exp(min(eval_loss, 20.0))
            metrics["perplexity"] = round(ppl, 4)
            logger.info(
                f"  [Step {state.global_step}]  "
                f"eval_loss={eval_loss:.4f}  perplexity={ppl:.2f}"
            )
        record = {
            "step":  state.global_step,
            "epoch": round(float(state.epoch or 0), 4),
            **metrics,
        }
        with open(self._metrics_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def print_environment():
    logger.info("═" * 64)
    logger.info("ENVIRONMENT — নিশা SFT")
    logger.info(f"  Python         : {platform.python_version()}")
    logger.info(f"  PyTorch        : {torch.__version__}")
    logger.info(f"  Transformers   : {transformers.__version__}")
    logger.info(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU [{i}]  {p.name}  "
                f"{p.total_memory / 1024**3:.1f} GB  CC {p.major}.{p.minor}"
            )
    sdpa_ok = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    logger.info(f"  SDPA           : {'available' if sdpa_ok else 'NOT available (upgrade PyTorch)'}")
    logger.info(f"  Precision      : FP16 mixed-precision (AMP)")
    logger.info(f"  Architecture   : LLaMA-style (RMSNorm · RoPE · SDPA)")
    logger.info(f"  Assistant      : নিশা  (Nisha — Assamese conversational AI)")
    logger.info("═" * 64)


def print_model_stats(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp16_mb   = total * 2 / 1024 ** 2
    fp32_opt  = total * 4 * 2 / 1024 ** 2   # Adam: 2× FP32 copies (m, v)
    logger.info("MODEL STATISTICS")
    logger.info(f"  Total params       : {total:,}")
    logger.info(f"  Trainable params   : {trainable:,}")
    logger.info(f"  Model size (FP16)  : {fp16_mb:.1f} MB")
    logger.info(f"  Optimizer states   : ~{fp32_opt:.1f} MB (FP32 AdamW)")
    logger.info("─" * 64)
    
class CheckpointLogCallback(TrainerCallback):
    """Log the path of each saved checkpoint."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        logger.info("Checkpoint saved → %s  (safetensors=True)", ckpt)

class MetricsLogCallback(TrainerCallback):
    """Forward Trainer metrics to our named logger (file + stdout)."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        parts = [f"step={state.global_step}"]
        for key in ("epoch", "loss", "learning_rate", "grad_norm", "train_loss"):
            val = logs.get(key)
            if val is not None:
                parts.append(
                    f"{key}={val:.6f}" if isinstance(val, float) else f"{key}={val}"
                )
        logger.info(" │ ".join(parts))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Reproducibility ───────────────────────────────────────────────────
    SEED = 42
    set_seed(SEED)

    print_environment()

    # ── Parse hparams.txt ─────────────────────────────────────────────────
    # We extract only the parameters that the SFT script controls directly.
    # LR-scheduler state, global step, best metric, etc. are persisted in
    # checkpoints/checkpoint-N/trainer_state.json and are restored
    # automatically by the HF Trainer on resume — no manual parsing needed.
    logger.info(f"Loading hyperparameters from '{HPARAMS_PATH}' …")
    hp = parse_hparams(HPARAMS_PATH)
    logger.info(f"  Parsed {len(hp)} key-value pair(s)")

    context_length   = _cast(hp.get("context_length"),        1024,  int)
    micro_batch_size = _cast(hp.get("micro_batch_size"),      4,     int)
    eff_batch_size   = _cast(hp.get("effective_batch_size"),  32,    int)
    gradient_clip    = _cast(hp.get("gradient_clip"),         1.0,   float)
    max_lr           = _cast(hp.get("max_lr"),                3e-4,  float)
    min_lr           = _cast(hp.get("min_lr"),                3e-5,  float)
    warmup_ratio     = _cast(hp.get("warmup_ratio"),          0.01,  float)
    beta1            = _cast(hp.get("beta1"),                 0.9,   float)
    beta2            = _cast(hp.get("beta2"),                 0.95,  float)
    weight_decay     = _cast(hp.get("weight_decay"),          0.1,   float)
    adam_eps         = _cast(hp.get("eps"),                   1e-8,  float)
    ignore_index     = _cast(hp.get("ignore_index"),          -100,  int)

    logger.info(
        f"  context_length={context_length}  micro_bs={micro_batch_size}  "
        f"eff_bs={eff_batch_size}  max_lr={max_lr}  "
        f"warmup_ratio={warmup_ratio}  ignore_index={ignore_index}"
    )

    # Sanity-check: LlamaForCausalLM hardcodes ignore_index=-100 in its loss.
    # Warn loudly if hparams.txt asks for something different.
    if ignore_index != -100:
        logger.warning(
            f"  hparams.txt specifies ignore_index={ignore_index}, but "
            "LlamaForCausalLM's built-in loss uses -100.  "
            "Labels in your .bin shards must use -100 for masked positions."
        )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from '{TOKENIZER_PATH}' …")
    if not Path(TOKENIZER_PATH).exists():
        raise FileNotFoundError(
            f"Tokenizer file not found at '{TOKENIZER_PATH}'.  "
            "Place tokenizer.json in the working directory."
        )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

    # Log and verify special token ids — DO NOT add or resize embeddings.
    logger.info(f"  vocab_size : {len(tokenizer):,}")
    logger.info(f"  pad_token  : {tokenizer.pad_token_id}  (expected 1)")
    logger.info(f"  bos_token  : {tokenizer.bos_token_id}  (expected 2)")
    logger.info(f"  eos_token  : {tokenizer.eos_token_id}  (expected 3)")

    for name, tid, expected in [
        ("PAD", tokenizer.pad_token_id, 1),
        ("BOS", tokenizer.bos_token_id, 2),
        ("EOS", tokenizer.eos_token_id, 3),
    ]:
        if tid != expected:
            logger.warning(
                f"  {name} token id is {tid}, expected {expected}.  "
                "Verify tokenizer is correct."
            )

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info("Initialising SFT shard dataset …")
    train_dataset = SFTShardDataset(
        shard_dir      = SHARD_DIR,
        manifest_path  = MANIFEST_PATH,
        context_length = context_length,
    )

    n_train = len(train_dataset)
    if n_train == 0:
        raise RuntimeError("Training dataset is empty. Check shard files and manifest.")

    # ── Gradient accumulation — DDP-aware ─────────────────────────────────
    # With Trainer/DDP, the effective global batch size is:
    #   per_device_batch × n_gpus × grad_accum_steps
    # We solve for grad_accum_steps so the global batch matches hparams.txt.
    num_gpus   = max(1, torch.cuda.device_count())
    grad_accum = max(1, eff_batch_size // (micro_batch_size * num_gpus))
    actual_eff = micro_batch_size * num_gpus * grad_accum

    logger.info(
        f"  GPUs={num_gpus}  per_device={micro_batch_size}  "
        f"grad_accum={grad_accum}  → effective_batch={actual_eff}"
    )
    if actual_eff != eff_batch_size:
        logger.warning(
            f"  Effective batch {actual_eff} ≠ target {eff_batch_size} "
            f"(num_gpus={num_gpus} doesn't divide evenly)."
        )

    # ── Training duration ─────────────────────────────────────────────────
    # Compute steps_per_epoch from dataset size so the script adapts to
    # any corpus size automatically.  Trainer also computes this internally;
    # we use it here only for logging and save_steps calculation.
    NUM_EPOCHS      = 3
    steps_per_epoch = math.ceil(n_train / eff_batch_size)
    total_steps     = steps_per_epoch * NUM_EPOCHS

    logger.info(
        f"  Samples={n_train:,}  steps/epoch≈{steps_per_epoch:,}  "
        f"total_steps≈{total_steps:,}  epochs={NUM_EPOCHS}"
    )

    # ── Checkpoint / save cadence ─────────────────────────────────────────
    # Save every ~10 % of an epoch (min 100 steps) for crash safety.
    save_steps = max(100, steps_per_epoch // 10)
    logger.info(f"  Saving checkpoint every {save_steps} steps")

    # ── Resume detection ──────────────────────────────────────────────────
    # Priority:
    #   1. SFT Trainer checkpoint in CKPT_DIR  (crash recovery mid-SFT)
    #   2. None → start SFT from BASE_CHECKPOINT weights
    #
    # Note: BASE_CHECKPOINT is always the source of model *weights*.
    # trainer_state.json inside an SFT checkpoint carries optimizer /
    # scheduler state and is restored automatically by the HF Trainer.
    sft_ckpt = get_last_checkpoint(str(CKPT_DIR))
    if sft_ckpt:
        logger.info(f"  Resuming SFT from Trainer checkpoint : '{sft_ckpt}'")
    else:
        logger.info(
            f"  No SFT checkpoint found in '{CKPT_DIR}'.  "
            f"Starting SFT from base model '{BASE_CHECKPOINT}'."
        )

    # ── Load model ────────────────────────────────────────────────────────
    if not Path(BASE_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"Base checkpoint directory not found: '{BASE_CHECKPOINT}'.  "
            "Download or symlink it to the working directory."
        )

    sdpa_ok   = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    attn_impl = "sdpa" if sdpa_ok else "eager"

    logger.info(
        f"Loading model weights from '{BASE_CHECKPOINT}'  "
        f"(attn_implementation='{attn_impl}') …"
    )

    try:
        # attn_implementation= was added in transformers 4.35; safe on 4.52.4
        model = LlamaForCausalLM.from_pretrained(
            BASE_CHECKPOINT,
            attn_implementation=attn_impl,
        )
    except TypeError:
        # Graceful fallback for unexpected version mismatches
        logger.warning(
            "  attn_implementation= kwarg rejected — "
            "falling back to manual config patch."
        )
        model = LlamaForCausalLM.from_pretrained(BASE_CHECKPOINT)
        if sdpa_ok:
            model.config._attn_implementation = "sdpa"

    # ── Training-time model config ────────────────────────────────────────
    # use_cache=True conflicts with gradient checkpointing; disable it.
    model.config.use_cache = False

    # Gradient checkpointing: recomputes activations on backward pass,
    # trading ~33 % extra FLOPs for large VRAM savings on T4.
    model.gradient_checkpointing_enable()

    logger.info("─" * 64)
    print_model_stats(model)
    logger.info(f"  Attention backend  : {attn_impl}")
    logger.info(f"  use_cache          : {model.config.use_cache}")
    logger.info(f"  grad_checkpointing : enabled")
    logger.info("─" * 64)

    # ── LR scheduler — version-aware ──────────────────────────────────────
    # 'cosine_with_min_lr' decays to min_lr instead of 0; requires >= 4.33.
    _tf_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if _tf_ver >= (4, 33):
        scheduler_type   = "cosine_with_min_lr"
        scheduler_kwargs = {"min_lr": min_lr}
        logger.info(f"  LR scheduler : cosine_with_min_lr  (min_lr={min_lr})")
    else:
        scheduler_type   = "cosine"
        scheduler_kwargs = {}
        logger.warning(
            f"  transformers {transformers.__version__} < 4.33 — "
            "using 'cosine' (decays to 0 instead of min_lr)"
        )

    # ── DataLoader workers ────────────────────────────────────────────────
    # Kaggle T4 nodes typically have 4 vCPUs.  2–4 workers balance
    # prefetch throughput against process spawn overhead.
    num_workers = max(2, min(4, os.cpu_count() or 2))
    logger.info(f"  DataLoader workers : {num_workers}")

    # ── TrainingArguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        # ── Output directories ──────────────────────────────────────────
        output_dir  = str(CKPT_DIR),
        logging_dir = str(LOG_DIR),

        # ── Duration ────────────────────────────────────────────────────
        # Epoch-driven training; Trainer computes total steps from dataset
        # size automatically.  max_steps=-1 defers to num_train_epochs.
        num_train_epochs = NUM_EPOCHS,
        max_steps        = -1,

        # ── Batch sizes and gradient accumulation ────────────────────────
        per_device_train_batch_size  = micro_batch_size,
        per_device_eval_batch_size   = micro_batch_size,
        gradient_accumulation_steps  = grad_accum,

        # ── Precision ───────────────────────────────────────────────────
        # T4 supports FP16 but NOT BF16.
        fp16 = True,
        bf16 = False,

        # ── Optimizer (AdamW from hparams.txt) ──────────────────────────
        optim         = "adamw_torch",
        learning_rate = max_lr,
        weight_decay  = weight_decay,
        adam_beta1    = beta1,
        adam_beta2    = beta2,
        adam_epsilon  = adam_eps,
        max_grad_norm = gradient_clip,

        # ── LR scheduler ────────────────────────────────────────────────
        lr_scheduler_type   = scheduler_type,
        lr_scheduler_kwargs = scheduler_kwargs,
        # warmup_ratio lets Trainer compute warmup_steps = ratio × total_steps,
        # which remains correct even when resuming mid-training.
        warmup_ratio        = warmup_ratio,

        # ── Checkpointing ───────────────────────────────────────────────
        save_strategy    = "steps",
        save_steps       = save_steps,
        save_total_limit = 3,          # keep the 3 most recent checkpoints
        save_safetensors = True,       # safetensors format; faster and safer

        # ── Evaluation ──────────────────────────────────────────────────
        # No explicit validation shards are specified in the SFT pipeline.
        # Set eval_strategy="no" to skip; add "steps" + eval_dataset later.
        eval_strategy = "no",

        # ── Logging ─────────────────────────────────────────────────────
        logging_strategy   = "steps",
        logging_steps      = 10,       # console log every 10 optimiser steps
        logging_first_step = True,     # always log step 0 for diagnostics
        report_to          = "none",   # no WandB / MLflow

        # ── DataLoader ──────────────────────────────────────────────────
        dataloader_num_workers        = num_workers,
        dataloader_pin_memory         = torch.cuda.is_available(),
        dataloader_persistent_workers = True,   # keep workers alive across epochs
        dataloader_prefetch_factor    = 2,      # prefetch 2 batches per worker

        # ── Correctness and multi-GPU safety ────────────────────────────
        remove_unused_columns      = False,   # never silently drop 'labels'
        ddp_find_unused_parameters = False,   # all params used; speeds up DDP

        # ── Reproducibility ─────────────────────────────────────────────
        seed      = SEED,
        data_seed = SEED,

        # ── Misc ────────────────────────────────────────────────────────
        disable_tqdm           = False,
        load_best_model_at_end = False,   # we want the final checkpoint, not best
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    # processing_class= replaces the deprecated tokenizer= parameter
    # (introduced in transformers 4.46; available on 4.52.4).
    # The tokenizer is saved alongside every checkpoint automatically.
    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = None,
        data_collator    = default_data_collator,
        callbacks=[
            MetricsLogCallback(),
            SFTMetricsCallback(),
            CheckpointLogCallback(),
        ],
        processing_class = tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("═" * 64)
    logger.info("STARTING SUPERVISED FINE-TUNING")
    logger.info("  নিশা (Nisha) — Assamese Conversational AI")
    logger.info("═" * 64)

    # If sft_ckpt is None, training starts fresh from BASE_CHECKPOINT weights.
    # If sft_ckpt is a path, Trainer restores model weights + optimizer +
    # scheduler + RNG state from that checkpoint for seamless crash recovery.
    train_result = trainer.train(resume_from_checkpoint=sft_ckpt)

    # ── Persist final artefacts ───────────────────────────────────────────
    logger.info(f"Saving final model to '{OUT_DIR}' …")
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    trainer.save_state()   # writes trainer_state.json to output_dir

    # ── Final summary ─────────────────────────────────────────────────────
    metrics   = train_result.metrics
    final_loss = metrics.get("train_loss", float("nan"))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("═" * 64)
    logger.info("TRAINING COMPLETE — নিশা")
    logger.info(f"  train_loss  : {final_loss:.4f}")
    logger.info(f"  model saved : {OUT_DIR.resolve()}")
    logger.info(f"  metrics log : {SFTMetricsCallback._metrics_file.resolve()}")
    logger.info("═" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
