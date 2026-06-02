"""
train.py — Production-quality training script for an Assamese decoder-only language model.

Architecture  : GPT / LLaMA Decoder-Only  (LlamaForCausalLM via Hugging Face)
Attention      : SDPA  (torch.nn.functional.scaled_dot_product_attention)
Precision      : FP16 mixed-precision
Target HW      : NVIDIA T4 16 GB  (Kaggle free tier)
Resume-safe    : auto-detects latest HF checkpoint and resumes transparently
Multi-GPU ready: uses Trainer abstractions; no single-GPU hard-coding

Author         : generated from hparams.txt + user spec
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import sys
import glob
import math
import json
import time
import bisect
import random
import logging
import warnings
import argparse
import platform
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np

# ── Torch ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Hugging Face ────────────────────────────────────────────────────────────
import transformers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

# ── Suppress noisy HF warnings that are not actionable ──────────────────────
warnings.filterwarnings("ignore", message=".*resume_download.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOGGING SETUP
#     We create structured log directories and wire up both console + file
#     handlers so every run is fully reproducible from logs alone.
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR   = Path("logs")
CKPT_DIR  = Path("checkpoints")
OUT_DIR   = Path("outputs")

for _d in [LOG_DIR, CKPT_DIR, OUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

_log_file = LOG_DIR / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
logger.info(f"Log file  : {_log_file.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  HPARAMS PARSER
#     Reads the plain-text hparams.txt file whose format is:
#       key = value   (lines starting with # are comments)
#     Returns a flat dict; caller is responsible for type casting.
# ──────────────────────────────────────────────────────────────────────────────
def parse_hparams(path: str = "hparams.txt") -> Dict[str, str]:
    """
    Parse a simple key = value config file.

    Rules:
      * Lines that start with # (after stripping) are section headers / comments.
      * Blank lines are ignored.
      * Values may span the same line only; multi-line values are not supported.
      * Inline comments after the value are NOT stripped (keep values clean).
    """
    hparams: Dict[str, str] = {}
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"hparams.txt not found at '{path}'.  "
            f"Please place it in the working directory or pass --hparams <path>."
        )

    with open(path_obj, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            # skip blank lines and comment / section-header lines
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                hparams[key.strip()] = val.strip()

    return hparams


def cast(val: str, default=None, cast_fn=None):
    """
    Safely cast a string value.  Returns *default* if val is empty/None.
    cast_fn is called on the stripped string, e.g. int, float.
    """
    if val is None or val.strip() == "":
        return default
    try:
        return cast_fn(val.strip()) if cast_fn else val.strip()
    except (ValueError, TypeError):
        logger.warning(f"Could not cast '{val}' — using default {default}")
        return default


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BUILD LlamaConfig FROM hparams.txt
#
#     This model is a modern LLaMA-style decoder-only transformer, intentionally
#     designed with the following architectural choices:
#
#       RMSNorm  — Root Mean Square normalisation instead of LayerNorm.
#                  RMSNorm omits the mean-centering term, reducing compute by
#                  ~15 % while matching or exceeding LayerNorm quality in
#                  practice.  All leading open-weight LLMs (LLaMA, Mistral,
#                  Gemma) use RMSNorm for this reason.
#
#       RoPE     — Rotary Position Embeddings.  Applied directly to query/key
#                  projections inside each attention head.  RoPE generalises
#                  better to longer sequences than learned absolute embeddings
#                  and requires no extra parameters.
#
#       SDPA     — PyTorch Scaled Dot-Product Attention fused kernel.
#                  Uses Flash-Attention-style tiling when available, delivering
#                  2–4× memory and speed improvements over naive attention.
#
#       MHA      — Full multi-head attention (num_key_value_heads ==
#                  num_attention_heads).  No grouped-query compression; chosen
#                  for simplicity at this model scale (~85 M parameters).
#
#     hparams.txt key → LlamaConfig field mapping:
#       n_layer        → num_hidden_layers
#       n_embd         → hidden_size
#       n_head         → num_attention_heads
#       context_length → max_position_embeddings
#       ffn_hidden     → intermediate_size
#       layer_norm_eps → rms_norm_eps   (epsilon for RMSNorm, not LayerNorm)
# ──────────────────────────────────────────────────────────────────────────────
def build_llama_config(hp: Dict[str, str], tokenizer) -> LlamaConfig:
    """
    Construct a LlamaConfig for the Assamese LLaMA-style decoder.

    Architecture is intentionally LLaMA-style:
      - RMSNorm at every residual boundary (pre-norm)
      - Rotary Position Embeddings (RoPE) on Q and K projections
      - SDPA fused attention kernel
      - GELU activation in the feed-forward network
      - Weight tying between token embedding and LM head (saves ~38 M params)
    """
    n_embd    = cast(hp.get("n_embd"),    default=768,  cast_fn=int)
    head_dim  = cast(hp.get("head_dim"),  default=64,   cast_fn=int)
    n_head    = cast(hp.get("n_head"),    default=12,   cast_fn=int)

    # FIX #7: Validate head_dim consistency — LlamaConfig derives head dim
    # internally as hidden_size // num_attention_heads.  If the hparams value
    # disagrees with that derivation, log a warning so the user is not silently
    # surprised.  n_embd must also be divisible by n_head.
    if n_embd % n_head != 0:
        raise ValueError(
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        )
    
    expected_head_dim = n_embd // n_head
    if head_dim != expected_head_dim:
        logger.warning(
            f"  head_dim={head_dim} in hparams does not match "
            f"n_embd // n_head = {n_embd} // {n_head} = {expected_head_dim}. "
            f"LlamaConfig will use {expected_head_dim}; hparams value ignored."
        )

    config = LlamaConfig(
        vocab_size            = len(tokenizer),          # always match tokenizer
        hidden_size           = n_embd,
        intermediate_size     = cast(hp.get("ffn_hidden"),      default=3072, cast_fn=int),
        num_hidden_layers     = cast(hp.get("n_layer"),         default=12,   cast_fn=int),
        num_attention_heads   = n_head,
        # Full MHA: num_key_value_heads == num_attention_heads (no GQA compression)
        num_key_value_heads   = n_head,
        hidden_act            = "gelu",
        max_position_embeddings = cast(hp.get("context_length"), default=1024, cast_fn=int),
        # rms_norm_eps: epsilon for RMSNorm stability — intentionally RMSNorm,
        # not LayerNorm.  The hparams key "layer_norm_eps" is reused as the
        # generic normalisation epsilon since the value (1e-5) is appropriate
        # for both norm variants.
        rms_norm_eps          = cast(hp.get("layer_norm_eps"),   default=1e-5, cast_fn=float),
        # RoPE: built into LlamaConfig; no additional positional embedding params
        rope_theta            = 10000.0,
        # Weight tying: the output projection matrix is shared with the input
        # embedding table, reducing parameter count and improving generalisation
        tie_word_embeddings   = (hp.get("weight_tying", "True").lower() == "true"),
        bos_token_id          = tokenizer.bos_token_id,
        eos_token_id          = tokenizer.eos_token_id,
        pad_token_id          = tokenizer.pad_token_id,
        # Disable KV-cache at config level; gradient checkpointing requires it off
        use_cache             = False,
    )
    return config


# ──────────────────────────────────────────────────────────────────────────────
# 4.  DATASET
#
#     Scalable binary-shard dataset for very large token corpora.
#
#     Design goals
#     ────────────
#     1. O(n_shards) memory overhead — NOT O(total_chunks).
#        Instead of materialising a Python list of millions of
#        (shard_path, token_offset) tuples, we store only one integer per
#        shard (its chunk count) and use a cumulative-sum array + binary
#        search to map any global index → (shard, local_chunk) in O(log S)
#        time, where S = number of shards.
#
#     2. Per-worker mmap caching.
#        np.memmap() is cheap (it just maps virtual address space), but
#        calling it on every __getitem__ still incurs OS overhead that
#        accumulates across millions of samples.  We cache open memmaps
#        in a worker-local dict keyed by shard path.  Because each DataLoader
#        worker is a separate process, the dict is never shared across
#        processes — no locking needed.  The cache is populated lazily on
#        first access and lives for the worker's lifetime.
#
#     3. Epoch-level shard reshuffling without index rebuild.
#        Reshuffling only permutes the shard order array (length = S), not
#        a list of millions of tuples.  The cumulative-sum array is recomputed
#        from the same per-shard chunk counts (just in a new order), which is
#        O(S) and typically takes microseconds.
#
#     4. Correct causal LM label semantics.
#        input_ids and labels are identical token sequences.
#        LlamaForCausalLM performs the causal shift internally.
#
#     Memory comparison (example: 1 B tokens, context_length=1024)
#       Old design : ~1 M tuples × ~120 bytes/tuple  ≈ 120 MB RAM
#       New design : S shard counts (S ≤ ~1000)       ≈   8 KB RAM
# ──────────────────────────────────────────────────────────────────────────────
class BinaryShardDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset over a directory of np.uint16 .bin shards.

    Global index → data mapping
    ---------------------------
    Each shard i contains  chunks_per_shard[i]  non-overlapping chunks of
    exactly `context_length` tokens.  We maintain a cumulative-sum array:

        cumulative[i] = sum(chunks_per_shard[0:i])   (length = S+1, cumulative[0]=0)

    For a global index g:
        shard_pos  = bisect_right(cumulative, g) - 1
        local_idx  = g - cumulative[shard_pos]
        token_offset = local_idx * context_length

    This is O(log S) per sample and requires only S+1 integers of overhead.

    Label alignment
    ---------------
    LlamaForCausalLM shifts labels internally.  input_ids and labels are the
    same token sequence; the model predicts token[i+1] from token[i].
    """

    def __init__(
        self,
        shard_dir: str,
        context_length: int = 1024,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.context_length = context_length
        self.shuffle_shards = shuffle_shards
        self.seed           = seed

        # Collect all .bin shards in canonical sorted order.
        # This sorted list is the stable reference; we permute an index array
        # rather than the list itself so the path→chunk-count mapping is
        # always consistent regardless of epoch.
        self._all_shards: List[Path] = sorted(Path(shard_dir).glob("*.bin"))
        if not self._all_shards:
            raise FileNotFoundError(
                f"No .bin shards found in '{shard_dir}'.  "
                f"Expected files named like train_000.bin, val_000.bin, …"
            )

        logger.info(f"  Found {len(self._all_shards)} shard(s) in '{shard_dir}'")
        
        # Initialize chunk-count storage
        self._chunks_per_shard: List[int] = []

        # ── Compute per-shard chunk counts once at startup ────────────────
        # We mmap each shard briefly just to read its length (no data loaded).            
            
        for p in self._all_shards:
            mm = np.memmap(p, dtype=np.uint16, mode="r")
        
            total_tokens = len(mm)
            full_chunks  = total_tokens // context_length
            remainder    = total_tokens % context_length
        
            if remainder != 0:
                logger.warning(
                    f"Shard '{p.name}' has {remainder} leftover tokens "
                    f"(not divisible by context_length={context_length}). "
                    f"These tokens will be ignored."
                )
        
            if full_chunks == 0:
                raise ValueError(
                    f"Shard '{p.name}' is too small for one full chunk. "
                    f"Tokens: {total_tokens}, context_length: {context_length}"
                )
        
            self._chunks_per_shard.append(full_chunks)
            del mm


        # ── Build the initial epoch-0 ordering ───────────────────────────
        # _shard_order is an array of indices into self._all_shards.
        # Permuting it is all that reshuffling ever does.
        self._shard_order: List[int] = list(range(len(self._all_shards)))
        self._cumulative: List[int]  = []   # populated by _rebuild_cumulative
        self._total_chunks: int      = 0

        self._rebuild_cumulative()          # sets _cumulative and _total_chunks

        logger.info(
            f"  Total chunks (samples): {self._total_chunks:,}  "
            f"| shard index RAM: {(len(self._all_shards) + 1) * 8 // 1024} KB"
        )

        # ── Per-worker mmap cache ─────────────────────────────────────────
        # Populated lazily inside worker processes; each worker maintains its
        # own dict — safe without any locking.
        self._mmap_cache: Dict[str, np.memmap] = {}

    # ── Internal helpers ──────────────────────────────────────────────────

    def _rebuild_cumulative(self):
        """
        Recompute the cumulative chunk-count array from the current shard
        order.  Called at construction and whenever set_epoch() reshuffles.

        Result: self._cumulative[i] = total chunks before shard_order[i].
        Length = len(shard_order) + 1, with _cumulative[0] = 0.
        """
        counts = [self._chunks_per_shard[i] for i in self._shard_order]
        cumulative = [0]
        for c in counts:
            cumulative.append(cumulative[-1] + c)
        self._cumulative  = cumulative
        self._total_chunks = cumulative[-1]

    def _global_to_local(self, global_idx: int) -> Tuple[Path, int]:
        """
        Map a global sample index to (shard_path, token_offset).

        Uses bisect on the cumulative array — O(log S).
        Never allocates or iterates over full chunk lists.
        """
        # bisect_right gives the first cumulative value strictly greater than
        # global_idx, so subtracting 1 gives the shard that owns this index.
        shard_pos   = bisect.bisect_right(self._cumulative, global_idx) - 1
        local_idx   = global_idx - self._cumulative[shard_pos]
        token_offset = local_idx * self.context_length
        shard_path  = self._all_shards[self._shard_order[shard_pos]]
        return shard_path, token_offset

    def _get_mmap(self, shard_path: Path) -> np.memmap:
        """
        Return a cached np.memmap for *shard_path*, opening it if necessary.

        The cache is keyed by the string form of the path and lives in the
        current process (each DataLoader worker process has its own copy).
        This eliminates the per-sample mmap open/close overhead that
        accumulates to ~5–15 % of dataloader time on large corpora.

        Safety: np.memmap objects are read-only and do not require cleanup
        between samples; the OS will reclaim mapped pages under memory
        pressure automatically.
        """
        key = str(shard_path)
        if key not in self._mmap_cache:
            self._mmap_cache[key] = np.memmap(shard_path, dtype=np.uint16, mode="r")
        return self._mmap_cache[key]

    # ── Epoch reshuffling ─────────────────────────────────────────────────

    def set_epoch(self, epoch: int):
        """
        Reshuffle shard order for the given epoch.

        Cost: O(S) — permutes a list of S integers and recomputes S+1
        cumulative sums.  No shard data is read; no chunk tuples are built.
        Called by EpochReshuffleCallback via Trainer's on_epoch_begin hook.
        """
        if not self.shuffle_shards:
            return
        logger.info(f"  Reshuffling shard order for epoch {epoch} …")
        rng = random.Random(self.seed + epoch)
        self._shard_order = list(range(len(self._all_shards)))
        rng.shuffle(self._shard_order)
        self._rebuild_cumulative()
        # Clear the mmap cache so workers do not hold stale file-descriptor
        # handles after the shard order changes.  Workers re-open lazily.
        self._mmap_cache.clear()
        logger.info(f"  [Epoch {epoch}] Total chunks after reshuffle: {self._total_chunks:,}")

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._total_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Auto-correct invalid indices instead of crashing
        if idx < 0:
            idx = 0
        elif idx >= self._total_chunks:
            idx = self._total_chunks - 1
        shard_path, token_offset = self._global_to_local(idx)
        
        # Retrieve (or open) the cached mmap for this shard.
        # Reading a slice from a mmap returns a numpy view backed by the
        # OS page cache — no Python-level copy until .astype() is called.
        mm    = self._get_mmap(shard_path)
        chunk = mm[token_offset : token_offset + self.context_length].astype(np.int64)

        # input_ids and labels carry the same token sequence.
        # LlamaForCausalLM shifts internally: position i predicts position i+1,
        # and cross-entropy is computed against labels[i+1].  Providing a
        # pre-shifted labels tensor would cause a double-shift error.
        tokens = torch.from_numpy(chunk)   # [context_length]
        # input_ids and labels must be independent tensors.  .astype() above
        # already created a fresh numpy copy, but both dict values pointing to
        # the same Tensor object is unsafe if any in-place op touches one key.
        return {"input_ids": tokens, "labels": tokens.clone()}



# ──────────────────────────────────────────────────────────────────────────────
# 5.  MODEL BUILDER
#     Wraps LlamaForCausalLM construction + optional weight init + gradient
#     checkpointing.  Does NOT load from checkpoint here — the HF Trainer
#     handles resume automatically.
# ──────────────────────────────────────────────────────────────────────────────
def build_model(config: LlamaConfig, init_std: float = 0.02) -> LlamaForCausalLM:
    """
    Instantiate and initialise a LlamaForCausalLM for Assamese pretraining.

    Architecture summary (all choices are intentional):
      - RMSNorm        : pre-norm at every transformer block boundary.
                         Numerically stable, ~15 % cheaper than LayerNorm.
      - RoPE           : rotary embeddings on Q/K; no learned position params.
      - SDPA           : fused scaled dot-product attention (PyTorch >= 2.0).
      - GELU           : activation in the feed-forward sublayer.
      - Weight tying   : embedding ↔ LM-head weight sharing.
      - Grad checkpoint: recompute activations on backward to save VRAM on T4.

    When resuming from a checkpoint the Trainer overwrites these random
    weights with the saved ones; this path only matters for fresh runs.
    """
    logger.info("Building LLaMA-style model  (RMSNorm · RoPE · SDPA) …")

    # ── Explicitly request SDPA attention backend ─────────────────────────
    # Passing attn_implementation="sdpa" to from_config / __init__ is the
    # officially supported way (transformers >= 4.36).  This sets
    # config._attn_implementation internally and is preserved through
    # save/load cycles.  We also guard for older transformers that don't
    # accept the kwarg.
    sdpa_available = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    attn_impl = "sdpa" if sdpa_available else "eager"
    if not sdpa_available:
        logger.warning(
            "  PyTorch SDPA not found (requires PyTorch >= 2.0).  "
            "Falling back to eager attention — higher VRAM usage."
        )

    try:
        model = LlamaForCausalLM(config, attn_implementation=attn_impl)
    except TypeError:
        # Older transformers versions don't accept attn_implementation in __init__
        logger.warning(
            "  attn_implementation kwarg not supported by this transformers version.  "
            "Setting config._attn_implementation manually after construction."
        )
        model = LlamaForCausalLM(config)
        if sdpa_available:
            model.config._attn_implementation = "sdpa"

    logger.info(f"  Attention implementation   : {attn_impl}")

    # ── Ensure use_cache is OFF during training ───────────────────────────
    # gradient_checkpointing and KV-cache are mutually exclusive.
    # Explicitly set this on the live model.config (not just LlamaConfig)
    # so it persists even if the config was modified after construction.
    model.config.use_cache = False
    logger.info("  use_cache                 : False  (required for grad checkpointing)")

    # ── Custom weight initialisation ─────────────────────────────────────
    # Residual projection layers (o_proj, down_proj) use scaled init
    # std / sqrt(2 * n_layers) per GPT-2/LLaMA to prevent gradient
    # explosion in deep networks.  All other Linear/Embedding layers
    # use the base init_std.
    n_layers = config.num_hidden_layers
    scaled_std = init_std / math.sqrt(2 * n_layers)
    _residual_proj_names = {"o_proj", "down_proj"}


    # Two-pass init: first all layers with base std, then residual projections
    # with scaled std.
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        elif isinstance(module, nn.Linear):
            layer_name = name.split(".")[-1]
            std = scaled_std if layer_name in _residual_proj_names else init_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    logger.info(f"  Weight initialisation std : {init_std}")

    # ── Gradient checkpointing ────────────────────────────────────────────
    # Enabled via TrainingArguments(gradient_checkpointing=True) so that
    # Trainer manages it correctly under DDP/FSDP.  Do NOT call
    # model.gradient_checkpointing_enable() here — Trainer does it after
    # wrapping the model, and calling it twice causes a no-op at best and
    # a DDP hook conflict at worst.
    logger.info("  Gradient checkpointing    : ENABLED (managed by Trainer)")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 6.  UTILITY: MODEL STATISTICS
#     Prints total/trainable parameter counts and estimated FP16 size.
# ──────────────────────────────────────────────────────────────────────────────
def print_model_stats(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp16_mb   = total * 2 / 1024 ** 2          # 2 bytes per FP16 param
    fp32_mb   = total * 4 / 1024 ** 2          # 4 bytes per FP32 param (optimizer states)

    logger.info("─" * 60)
    logger.info("MODEL STATISTICS")
    logger.info(f"  Total parameters          : {total:,}")
    logger.info(f"  Trainable parameters      : {trainable:,}")
    logger.info(f"  Estimated size (FP16)     : {fp16_mb:.1f} MB")
    logger.info(f"  Optimizer states (FP32 ×2): {fp32_mb * 2:.1f} MB")
    logger.info("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  UTILITY: ENVIRONMENT SUMMARY
#     Prints CUDA info, precision mode, package versions so that every
#     run starts with a self-contained diagnostic header.
# ──────────────────────────────────────────────────────────────────────────────
def print_environment():
    logger.info("═" * 60)
    logger.info("ENVIRONMENT")
    logger.info(f"  Python                    : {platform.python_version()}")
    logger.info(f"  PyTorch                   : {torch.__version__}")
    logger.info(f"  Transformers              : {transformers.__version__}")
    logger.info(f"  CUDA available            : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU [{i}]  {props.name}  "
                f"{props.total_memory / 1024**3:.1f} GB VRAM  "
                f"CC {props.major}.{props.minor}"
            )
    else:
        logger.warning("  No CUDA GPU found — training will be VERY slow on CPU.")

    # Check SDPA availability (requires PyTorch >= 2.0)
    sdpa_ok = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    logger.info(f"  SDPA attention available  : {sdpa_ok}")
    if not sdpa_ok:
        logger.warning(
            "  SDPA not found.  Upgrade to PyTorch >= 2.0.  "
            "Falling back to eager attention (slower, more VRAM)."
        )

    logger.info(f"  Architecture              : LLaMA-style  (RMSNorm · RoPE · SDPA · MHA)")
    logger.info(f"  Precision mode            : FP16 mixed-precision")
    logger.info("═" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CUSTOM TRAINER CALLBACK
#     Logs validation perplexity and saves a human-readable metrics JSON.
# ──────────────────────────────────────────────────────────────────────────────
class MetricsCallback(TrainerCallback):
    """
    After each evaluation step:
      * compute perplexity from eval_loss
      * log to console
      * append to outputs/metrics.jsonl

    The JSONL format (one JSON object per line) is easy to post-process
    with pandas or jq for analysis.
    """

    METRICS_FILE = OUT_DIR / "metrics.jsonl"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        # FIX #2: Clamp eval_loss before exp() to avoid OverflowError during early
        # training when loss can exceed 20+.  exp(85) ≈ 8.5e36 (within float range);
        # anything beyond that is reported as inf rather than crashing the run.
        perplexity = math.exp(min(eval_loss, 85)) if eval_loss is not None else None

        if perplexity is not None:
            logger.info(
                f"[Step {state.global_step}]  "
                f"eval_loss={eval_loss:.4f}  "
                f"perplexity={perplexity:.2f}"
            )
            metrics["perplexity"] = perplexity

        # Persist to JSONL — only on the main process to avoid interleaved
        # writes from all DDP ranks hitting the same file simultaneously.
        if state.is_world_process_zero:
            record = {"step": state.global_step, **metrics}
            with open(self.METRICS_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 8b. EPOCH RESHUFFLE CALLBACK  (FIX #8)
#     Calls dataset.set_epoch(epoch) at the start of every epoch so shard
#     order is re-randomised without loading any data into RAM.
#     Compatible with both single- and multi-GPU Trainer setups.
# ──────────────────────────────────────────────────────────────────────────────
class EpochReshuffleCallback(TrainerCallback):
    """
    Triggers BinaryShardDataset.set_epoch(epoch) before each training epoch
    begins.  This re-builds the flat chunk index with a new per-epoch shard
    shuffle, giving true epoch-level data diversity without ever pulling
    full shard contents into RAM.
    """

    def __init__(self, train_dataset: "BinaryShardDataset"):
        self.train_dataset = train_dataset
        self._last_epoch: int = -1   # FIX #4: track last epoch to avoid duplicate reshuffles

    def on_epoch_begin(self, args, state, control, **kwargs):
        # state.epoch is a float; round to nearest int to avoid truncation
        # causing epoch 1.0 to be misidentified as epoch 0 (e.g., 0.999 → 0).
        epoch = round(state.epoch)
        if epoch != self._last_epoch:
            self._last_epoch = epoch
            self.train_dataset.set_epoch(epoch)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  FIND LATEST CHECKPOINT
#     Wraps HF's get_last_checkpoint with logging so the user always knows
#     whether we are starting fresh or resuming.
# ──────────────────────────────────────────────────────────────────────────────
def find_checkpoint(ckpt_dir: Path) -> Optional[str]:
    """
    Detect the latest HF checkpoint inside *ckpt_dir*.

    HF Trainer saves checkpoints as:
        checkpoints/checkpoint-10/
        checkpoints/checkpoint-20/
        ...

    get_last_checkpoint() returns the path to the highest-numbered one.
    Returns None if no checkpoint exists (fresh training run).
    """
    last = get_last_checkpoint(str(ckpt_dir))
    if last:
        # Verify the checkpoint is complete (not mid-write from a crashed run).
        state_file = Path(last) / "trainer_state.json"
        if not state_file.exists():
            logger.warning(
                f"  Checkpoint '{last}' is missing trainer_state.json — "
                f"likely a partial write from a crashed run.  Ignoring it."
            )
            last = None
        else:
            logger.info(f"  Checkpoint resume         : RESUMING from '{last}'")
    if not last:
        logger.info("  Checkpoint resume         : STARTING FRESH (no valid checkpoint found)")
    return last


# ──────────────────────────────────────────────────────────────────────────────
# 10. COMPUTE NUM TRAINING STEPS
#     We need the total number of steps for the cosine LR scheduler, which
#     requires knowing the dataset size upfront.
# ──────────────────────────────────────────────────────────────────────────────
def compute_total_steps(
    train_dataset: Dataset,
    effective_batch_size: int,
    micro_batch_size: int,
    num_epochs: int = 1,
) -> Tuple[int, int]:
    """
    Returns (total_steps, grad_accum_steps).

    effective_batch_size = GLOBAL batch size across ALL processes.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Global batch processed per optimizer micro-step
    base_global_batch = micro_batch_size * world_size

    if effective_batch_size % base_global_batch != 0:
        raise ValueError(
            f"effective_batch_size ({effective_batch_size}) must be divisible by "
            f"micro_batch_size * world_size "
            f"({micro_batch_size} * {world_size} = {base_global_batch})"
        )

    grad_accum = effective_batch_size // base_global_batch

    steps_per_epoch = math.ceil(
        len(train_dataset) / effective_batch_size
    )

    total_steps = steps_per_epoch * num_epochs

    return total_steps, grad_accum


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── 11.1  CLI (minimal — mostly driven by hparams.txt) ───────────────
    parser = argparse.ArgumentParser(
        description="Train an Assamese GPT-decoder LM from scratch."
    )
    parser.add_argument(
        "--hparams",
        default="/kaggle/working/hparams.txt",
        help="Path to the hparams.txt config file  (default: hparams.txt)",
    )
    parser.add_argument(
        "--train_dir",
        default="/kaggle/working/token_shards/train",
        help="Directory of training .bin shards  (default: train/)",
    )
    parser.add_argument(
        "--val_dir",
        default="/kaggle/working/token_shards/val",
        help="Directory of validation .bin shards  (default: val/)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="/kaggle/working/tokenizer.json",
        help="Path to the HF tokenizer JSON  (default: The_Assamese_Tokenizer/…)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs  (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed  (default: 42)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile() for the model  (disabled by default on T4)",
    )
    args = parser.parse_args()

    # ── 11.2  Environment diagnostics ────────────────────────────────────
    print_environment()

    # ── 11.3  Reproducibility ─────────────────────────────────────────────
    set_seed(args.seed)
    logger.info(f"Global random seed set to  : {args.seed}")

    # ── 11.4  Parse hparams.txt ───────────────────────────────────────────
    logger.info(f"Loading hyperparameters from: {args.hparams}")
    hp = parse_hparams(args.hparams)
    logger.info(f"  Parsed {len(hp)} key-value pairs")
   
    # ── 11.5  Load tokenizer ──────────────────────────────────────────────
    logger.info(f"Loading tokenizer from      : {args.tokenizer_path}")

    if not Path(args.tokenizer_path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found at '{args.tokenizer_path}'.  "
            "Please check the path or the repository structure."
        )

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── REQUIRED TOKENS ───────────────────────────────────────────────────
    # EOS is mandatory for causal LMs.
    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer is missing eos_token_id. "
            "Your tokenizer must define an EOS token."
        )

    # BOS fallback
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
        logger.warning(
            f"  BOS token missing — using EOS as BOS (id={tokenizer.eos_token_id})"
        )

    # PAD fallback
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(
            f"  PAD token missing — using EOS as PAD (id={tokenizer.eos_token_id})"
        )

    # Optional UNK warning
    if tokenizer.unk_token_id is None:
        logger.warning("  UNK token missing.")

    # Final verification
    logger.info(f"  BOS token id : {tokenizer.bos_token_id}")
    logger.info(f"  EOS token id : {tokenizer.eos_token_id}")
    logger.info(f"  PAD token id : {tokenizer.pad_token_id}")
    logger.info(f"  UNK token id : {tokenizer.unk_token_id}")
    logger.info(f"  Vocabulary size : {len(tokenizer):,}")


    # FIX #3: PreTrainedTokenizerFast loaded from a bare tokenizer.json (without
    # an accompanying tokenizer_config.json) often leaves pad_token_id as None.
    # Passing None into LlamaConfig causes silent bugs in HF utilities that rely
    # on pad_token_id.  Fall back to eos_token_id, which is the standard practice
    # for decoder-only causal LMs that do not use padding during training.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.warning(
            "  pad_token not set in tokenizer — using EOS token as PAD "
            f"(id={tokenizer.eos_token_id}).  This is standard for causal LMs."
        )

    # ── 11.6  Build model config ──────────────────────────────────────────
    context_length = cast(hp.get("context_length"), default=1024, cast_fn=int)
    llama_config   = build_llama_config(hp, tokenizer)
    logger.info("LlamaConfig built successfully.")

    # ── 11.7  Build datasets ──────────────────────────────────────────────
    logger.info("Building training dataset …")
    train_dataset = BinaryShardDataset(
        shard_dir=args.train_dir,
        context_length=context_length,
        shuffle_shards=True,
        seed=args.seed,
    )

    logger.info("Building validation dataset …")
    val_dataset = BinaryShardDataset(
        shard_dir=args.val_dir,
        context_length=context_length,
        shuffle_shards=False,  # deterministic validation
        seed=args.seed,
    )

    # ── 11.8  Training hyper-parameters from hparams.txt ─────────────────
    max_lr           = cast(hp.get("max_lr"),             default=3e-4,  cast_fn=float)
    min_lr           = cast(hp.get("min_lr"),             default=3e-5,  cast_fn=float)
    warmup_ratio     = cast(hp.get("warmup_ratio"),       default=0.01,  cast_fn=float)
    gradient_clip    = cast(hp.get("gradient_clip"),      default=1.0,   cast_fn=float)
    eff_batch_size   = cast(hp.get("effective_batch_size"), default=32,  cast_fn=int)
    micro_batch_size = cast(hp.get("micro_batch_size"),   default=2,     cast_fn=int)

    beta1       = cast(hp.get("beta1"),        default=0.9,   cast_fn=float)
    beta2       = cast(hp.get("beta2"),        default=0.95,  cast_fn=float)
    weight_decay= cast(hp.get("weight_decay"), default=0.1,   cast_fn=float)
    adam_eps    = cast(hp.get("eps"),          default=1e-6,  cast_fn=float)
    init_std    = cast(hp.get("init_std"),     default=0.02,  cast_fn=float)

    total_steps, grad_accum_steps = compute_total_steps(
        train_dataset, eff_batch_size, micro_batch_size, args.num_epochs
    )
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    # Proportional checkpoint/eval cadence: ~5 % of total steps, clamped
    # between 50 and 500 so small datasets still get at least one checkpoint.
    eval_steps = max(1, min(50, total_steps // 20))
    save_steps = eval_steps

    logger.info("TRAINING HYPERPARAMETERS")
    logger.info(f"  max_lr                    : {max_lr}")
    logger.info(f"  min_lr                    : {min_lr}")
    logger.info(f"  warmup_ratio              : {warmup_ratio}  ({warmup_steps} steps)")
    logger.info(f"  effective_batch_size      : {eff_batch_size}")
    logger.info(f"  micro_batch_size          : {micro_batch_size}")
    logger.info(f"  gradient_accumulation     : {grad_accum_steps} steps")
    logger.info(f"  gradient_clip             : {gradient_clip}")
    logger.info(f"  total_train_steps         : {total_steps}")
    logger.info(f"  eval every                : {eval_steps} steps")
    logger.info(f"  save every                : {save_steps} steps")

    # ── FIX #9: Scheduler type with version fallback ──────────────────────
    # "cosine_with_min_lr" (transformers >= 4.33) decays to min_lr instead
    # of 0.  If the installed version is older, fall back to "cosine" which
    # decays to 0 — a minor difference in practice at this model scale.
    # FIX #9: Use re.findall to extract only digit groups from the version string.
    # Splitting on "." and calling int() fails on pre-release tags like "4.46.0rc1"
    # or "4.46.dev0" where a segment contains non-numeric characters.
    _tf_version = tuple(int(x) for x in re.findall(r'\d+', transformers.__version__)[:2])
    if _tf_version >= (4, 33):
        scheduler_type   = "cosine_with_min_lr"
        scheduler_kwargs = {"min_lr": min_lr}
        logger.info(f"  LR scheduler              : cosine_with_min_lr  (min_lr={min_lr})")
    else:
        scheduler_type   = "cosine"
        scheduler_kwargs = {}
        logger.warning(
            f"  transformers {transformers.__version__} < 4.33: "
            f"'cosine_with_min_lr' unavailable — using 'cosine' (decays to 0)."
        )

    # ── 11.9  Detect latest checkpoint for auto-resume ───────────────────
    resume_from = find_checkpoint(CKPT_DIR)

    # ── 11.10  Build model (random init or will be overwritten by Trainer) ─
    model = build_model(llama_config, init_std=init_std)
    print_model_stats(model)

    # ── 11.11  Optional torch.compile ────────────────────────────────────
    # torch.compile is powerful but adds warm-up overhead and may conflict
    # with SDPA on older CUDA drivers.  Disabled by default on T4 Kaggle.
    if args.compile:
        if hasattr(torch, "compile"):
            logger.info("torch.compile()            : ENABLED  (may take ~2 min to warm up)")
            model = torch.compile(model)
        else:
            logger.warning("torch.compile requested but not available (PyTorch < 2.0). Skipping.")

    # ── 11.12  Num CPU workers for DataLoader ─────────────────────────────
    # On Kaggle T4, typically 4 vCPUs are available.
    # We cap at 4 to avoid memory pressure from worker spawning.
    num_workers = min(4, os.cpu_count() or 1)
    logger.info(f"  DataLoader num_workers    : {num_workers}")

    # ── 11.13  TrainingArguments ──────────────────────────────────────────
    # FIX #4: Use max_steps as the primary training control.
    # Kaggle sessions are time-limited; epoch-based training can silently
    # run past the session wall and lose progress.  max_steps gives an
    # explicit hard stop that pairs naturally with cosine LR scheduling.
    # num_train_epochs is kept as a soft upper bound but max_steps wins.
    training_args = TrainingArguments(
        # ── Paths ──────────────────────────────────────────────────────
        output_dir            = str(CKPT_DIR),
        logging_dir           = str(LOG_DIR),

        # ── Batch sizes and accumulation ───────────────────────────────
        per_device_train_batch_size = micro_batch_size,
        per_device_eval_batch_size  = micro_batch_size,
        gradient_accumulation_steps = grad_accum_steps,

        # ── Epochs / steps  (FIX #4) ───────────────────────────────────
        # max_steps takes priority over num_train_epochs in HF Trainer.
        # We set both: max_steps as the hard cap, num_train_epochs as the
        # logical epoch count for the LR scheduler's period.
        max_steps             = total_steps,      # explicit step-driven training
        num_train_epochs      = args.num_epochs,  # kept for scheduler period calc

        # ── Precision ──────────────────────────────────────────────────
        fp16                  = True,             # FP16 mixed-precision
        bf16                  = False,            # T4 does not support BF16
        gradient_checkpointing= True,             # managed here, not in build_model

        # ── Optimizer ──────────────────────────────────────────────────
        optim                 = "adamw_torch",    # native PyTorch AdamW
        learning_rate         = max_lr,
        weight_decay          = weight_decay,
        adam_beta1            = beta1,
        adam_beta2            = beta2,
        adam_epsilon          = adam_eps,
        max_grad_norm         = gradient_clip,

        # ── LR Scheduler  (FIX #9 wired in) ───────────────────────────
        lr_scheduler_type     = scheduler_type,
        lr_scheduler_kwargs   = scheduler_kwargs,
        warmup_steps          = warmup_steps,

        # ── Checkpointing  (FIX #5) ────────────────────────────────────
        save_strategy         = "steps",
        save_steps            = save_steps,       # every 100 steps
        # save_steps            = 5,       # every 5 steps for testing only
        save_total_limit      = 2,                # keep only last 2 checkpoints
        save_safetensors      = True,             # use safetensors format

        # ── Evaluation  (FIX #5) ───────────────────────────────────────
        eval_strategy         = "steps",
        eval_steps            = eval_steps,       # every 100 steps
        # eval_steps            = 5,       # every 5 steps for testing only

        # ── Logging ────────────────────────────────────────────────────
        logging_strategy      = "steps",
        logging_steps         = 10,               # console log every 10 steps
        logging_first_step    = True,
        report_to             = "none",           # no W&B / MLflow on Kaggle

        # ── DataLoader ─────────────────────────────────────────────────
        dataloader_num_workers        = num_workers,
        dataloader_pin_memory         = torch.cuda.is_available(),
        dataloader_persistent_workers = False,

        # ── FIX #7: Prevent Trainer dropping dataset columns ───────────
        # Without this, Trainer inspects model.forward() signature and
        # silently removes fields it doesn't recognise (e.g. "labels"
        # under certain transformers versions).
        remove_unused_columns = False,

        # ── Reproducibility ────────────────────────────────────────────
        seed                  = args.seed,
        data_seed             = args.seed,

        # ── Resume / crash recovery ────────────────────────────────────
        # When resume_from is set, Trainer restores model weights, optimizer
        # state, LR scheduler, and RNG state automatically.

        # ── Multi-GPU readiness ────────────────────────────────────────
        # Trainer automatically uses all visible GPUs via DataParallel /
        # DistributedDataParallel when more than one GPU is detected.
        ddp_find_unused_parameters = False,       # speeds up DDP when all params used

        # ── Misc ───────────────────────────────────────────────────────
        disable_tqdm          = False,
        load_best_model_at_end= False,            # saves VRAM; we want the last ckpt
        # FIX #6: metric_for_best_model and greater_is_better are meaningless
        # (and trigger HF deprecation warnings) when load_best_model_at_end=False.
        # Removed to keep TrainingArguments clean and avoid future surprises if
        # a transformers update starts honouring this metric for checkpoint pruning.
    )

    # ── 11.14  Trainer  (FIX #3) ─────────────────────────────────────────
    # Passing tokenizer= is deprecated in transformers >= 4.46.
    # The replacement is processing_class= which accepts tokenizers,
    # feature extractors, and processors uniformly, and still saves the
    # tokenizer alongside every checkpoint.
    try:
        trainer = Trainer(
            model               = model,
            args                = training_args,
            train_dataset       = train_dataset,
            eval_dataset        = val_dataset,
            data_collator       = default_data_collator,
            callbacks           = [MetricsCallback(), EpochReshuffleCallback(train_dataset)],
            processing_class    = tokenizer,
        )
    except TypeError:
        logger.warning(
            "processing_class not supported by this transformers version. "
            "Falling back to tokenizer=."
        )

        trainer = Trainer(
            model               = model,
            args                = training_args,
            train_dataset       = train_dataset,
            eval_dataset        = val_dataset,
            data_collator       = default_data_collator,
            callbacks           = [MetricsCallback(), EpochReshuffleCallback(train_dataset)],
            tokenizer           = tokenizer,
        )


    # ── 11.15  Train ──────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STARTING TRAINING")
    logger.info("═" * 60)

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # ── 11.16  Save final model ───────────────────────────────────────────
    logger.info("Saving final model to outputs/ …")
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    # ── 11.17  Log final metrics ──────────────────────────────────────────
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    final_loss = metrics.get("train_loss", float("nan"))
    # FIX #2: Clamp before exp() — same overflow guard as in MetricsCallback.
    final_ppl  = math.exp(min(final_loss, 85)) if not math.isnan(final_loss) else float("nan")

    logger.info("═" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Final train loss          : {final_loss:.4f}")
    logger.info(f"  Final train perplexity    : {final_ppl:.2f}")
    logger.info(f"  Model saved to            : {OUT_DIR.resolve()}")
    logger.info(f"  Metrics log               : {MetricsCallback.METRICS_FILE.resolve()}")
    logger.info("═" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()