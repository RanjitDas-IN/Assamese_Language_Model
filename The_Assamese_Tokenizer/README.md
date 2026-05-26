# Assamese Tokenizer

A high-performance custom BPE tokenizer built specifically for Assamese Large Language Models (LLMs).

This project was created as part of a larger effort to build a fully native Assamese AI ecosystem — including datasets, tokenization pipelines, and future GPT-style language models trained primarily on Assamese text.

---

# Why I Built This

Most existing multilingual tokenizers do not properly handle Assamese.

Assamese is usually grouped together with Bengali or other Indic languages inside multilingual vocabularies. While this works at a basic level, it creates several problems:

- Poor subword segmentation
- Fragmented Assamese words
- Unnatural token boundaries
- Inefficient token compression
- Reduced language modeling quality
- Weak handling of Assamese morphology and suffix structures

Generic multilingual tokenizers are optimized for many languages simultaneously.
This tokenizer was built specifically for Assamese.

The goal is to:

- Preserve Assamese linguistic structure
- Improve token efficiency
- Reduce fragmentation
- Support large-scale Assamese language model training
- Create a tokenizer optimized for GPT-style autoregressive transformers
- Build foundational infrastructure for future Assamese AI systems

---

# Key Features

## 1. Custom Assamese BPE Vocabulary

This tokenizer uses Byte Pair Encoding (BPE) trained directly on Assamese text.

Features:

- Learns Assamese subwords automatically
- Captures common suffixes and morphemes
- Handles compound Assamese words efficiently
- Reduces vocabulary redundancy
- Improves token compression ratio

Vocabulary size:

```python
VOCAB_SIZE = 50_000
```

---

## 2. SQLite Streaming Training Pipeline

One of the most important features of this project is the streaming training architecture.

Instead of:

- loading massive text files into RAM
- generating temporary files
- requiring huge memory usage

this tokenizer streams data directly from SQLite.

Benefits:

- Extremely memory efficient
- Scales to huge datasets
- Faster dataset management
- Easier preprocessing workflows
- Better handling of terabyte-scale corpora in the future

Streaming occurs in configurable batches:

```python
BATCH_SIZE = 50_000
```

This makes the tokenizer suitable for large Assamese corpus training.

---
## 2.1 Dataset used to build this Tokenizer:
| Topic / Dataset                              | Tokens            | Approx. Scale | Source |
|----------------------------------------------|------------------:|---------------|--------|
| Poems Dataset                                | 92.6K             | 0.0000926B    | Kaggle & Sosanko Sarmah (Contributor) |
| Song Lyrics Dataset                          | 4.5M              | 0.0045B       | Kaggle (Spotify API) |
| Story Dataset                                | 52.6B             | 52.6 Billion  | HuggingFace Dataset |
| Crawled Data                                 | 7T                | 7 Trillion    | Various Web Sources |
| CC-100 Dataset                               | 5.9M              | 0.0059B       | Common Crawl |
| Qwen3 Tokens                                 | 2B                | 2 Billion     | Kaggle |
| Kaggle News Articles Dataset                 | 49.6B             | 49.6 Billion  | Kaggle |
| IndicCorp v2 (AI4Bharat)                     | 37.8T             | 37.8 Trillion | AI4Bharat Dataset |
| Assamese Monolingual Corpus (MWire-Labs)     | 38.7B             | 38.7 Billion  | MWire-Labs |
| DailyHunt Dataset                            | 184.2B            | 184.2 Billion | Rahular Varta Dataset |
| Wikipedia Dump (2019–2025)                   | 0.2T              | 200 Billion   | Wikipedia |
|                                    |         || |
| **Total**                                    | **45.3T**         | **45.333 Trillion** | |

---
## 3. Unicode Normalization for Assamese

Indic scripts often contain visually identical Unicode sequences represented differently internally.

This tokenizer applies NFC normalization:

```python
normalizers.NFC()
```

Benefits:

- Prevents token fragmentation
- Standardizes Unicode representations
- Improves vocabulary consistency
- Handles Assamese/Bengali script more reliably

---

## 4. GPT-Style Special Tokens

The tokenizer includes:

- `[UNK]`
- `[PAD]`
- `[BOS]`
- `[EOS]`

These are integrated using Hugging Face post-processing templates.

This design makes the tokenizer compatible with:

- Decoder-only transformers
- GPT-style training
- Autoregressive generation
- Custom Assamese language models

Example formatting:

```text
[BOS] Assamese sentence here [EOS]
```

---

## 5. Hugging Face Compatibility

The tokenizer is exported using:

```python
PreTrainedTokenizerFast
```

This allows direct compatibility with:

- Hugging Face Transformers
- PyTorch training pipelines
- Custom dataloaders
- AutoTokenizer
- Future Assamese LLM checkpoints

Load example:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./assamese_tokenizer")
```

---

## 6. Built-in Sanity Testing

The project includes automatic validation tests.

The tokenizer:

- Encodes Assamese sentences
- Decodes them back
- Verifies reconstruction integrity
- Displays token breakdowns

This ensures:

- Stable encoding
- Proper decoding
- Reliable tokenizer behavior

---

# Technical Architecture

## Tokenizer Type

- Algorithm: BPE (Byte Pair Encoding)
- Model Style: GPT-style autoregressive LM
- Script: Assamese (Bengali-Assamese script)

---

## Pre-tokenization Strategy

This tokenizer intentionally uses simple whitespace pre-tokenization:

```python
pre_tokenizers.Whitespace()
```

The BPE model then learns subword merges automatically.

This avoids unnecessary early fragmentation while allowing the tokenizer to learn Assamese structure naturally from data.

---

# Training Pipeline Overview

```text
SQLite Database
      ↓
Streaming Generator
      ↓
Unicode Normalization
      ↓
Whitespace Pre-tokenization
      ↓
BPE Training
      ↓
Special Token Processing
      ↓
Hugging Face Export
```

---

# Project Structure

```text
assamese_tokenizer/
│
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```

---

# Generated Files

## tokenizer.json

Contains:

- Vocabulary
- Merge rules
- Decoder rules
- Normalization configuration
- Pre-tokenizer configuration
- Post-processing templates
- Special token definitions

This is the core tokenizer model.

---

## tokenizer_config.json

Contains:

- Hugging Face metadata
- Special token configuration
- Model max length
- Tokenizer settings

This enables easy loading with:

```python
AutoTokenizer.from_pretrained()
```

---

# Example Usage

## Loading the Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./assamese_tokenizer")
```

---

## Encoding Assamese Text

```python
text = "অসমীয়া ভাষা এটি সুন্দৰ ভাষা।"

encoded = tokenizer(text)
print(encoded["input_ids"])
```

---

## Decoding

```python
decoded = tokenizer.decode(encoded["input_ids"])
print(decoded)
```

---

# Design Philosophy

This tokenizer was not built as a generic multilingual tokenizer.

It was designed specifically for Assamese language modeling.

The focus is:

- linguistic preservation
- scalable infrastructure
- efficient training
- Assamese-first optimization
- future LLM compatibility

The long-term vision is to help build fully native Assamese AI systems.

---

# Future Goals

Planned improvements include:

- Larger Assamese corpora training
- Improved punctuation handling
- Advanced normalization research
- Token compression benchmarking
- Comparison against multilingual tokenizers
- Native Assamese conversational models
- Public Hugging Face release
- Integration with custom Assamese GPT models

---

# Author

Ranjit Das

Developer focused on Assamese AI infrastructure, tokenization systems, large-scale datasets, and native language model development.

---

# License

This project is intended for research and educational purposes. And it is free to all user including commercial uses