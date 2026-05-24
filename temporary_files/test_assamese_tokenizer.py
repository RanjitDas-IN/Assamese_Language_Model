"""
test_assamese_tokenizer.py  v0.1.2
-----------------------------------
Usage:
    pip install rich transformers
    python test_assamese_tokenizer.py \
        --tokenizer_dir ./assamese_tokenizer \
        --sample_file   ./data/sample.txt
"""

import argparse, os, sys, time
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box
from transformers import PreTrainedTokenizerFast

console = Console()

# ── tiny colour helpers ───────────────────────────────────────────────────────
def ok(s):   return f"[bold green]✓  {s}[/]"
def err(s):  return f"[bold red]✗  {s}[/]"
def warn(s): return f"[bold yellow]⚠  {s}[/]"
def val(s):  return f"[bold white]{s}[/]"

# ── results tracker ───────────────────────────────────────────────────────────
results = {"pass": 0, "fail": 0, "warn": 0}

def record(status):
    results[status] += 1

# ─────────────────────────────────────────────────────────────────────────────
# TEST 01 — tokenizer summary
# ─────────────────────────────────────────────────────────────────────────────
def test_summary(tok):
    console.print(Rule("[bold cyan]01  Tokenizer Summary[/]", style="cyan"))
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim", width=22)
    t.add_column()
    t.add_row("vocab size",       val(f"{tok.vocab_size:,}"))
    t.add_row("model max length", val(str(tok.model_max_length)))
    t.add_row("bos token",        f"[yellow]{tok.bos_token}[/]  id=[cyan]{tok.bos_token_id}[/]")
    t.add_row("eos token",        f"[yellow]{tok.eos_token}[/]  id=[cyan]{tok.eos_token_id}[/]")
    t.add_row("pad token",        f"[yellow]{tok.pad_token}[/]  id=[cyan]{tok.pad_token_id}[/]")
    t.add_row("unk token",        f"[yellow]{tok.unk_token}[/]  id=[cyan]{tok.unk_token_id}[/]")
    console.print(t)
    record("pass")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 02 — vocab size
# ─────────────────────────────────────────────────────────────────────────────
def test_vocab_size(tok, expected=50_000):
    console.print(Rule("[bold cyan]02  Vocab Size[/]", style="cyan"))
    actual = tok.vocab_size
    if actual == expected:
        console.print(ok(f"vocab_size = {actual:,}  (expected {expected:,})"))
        record("pass")
    else:
        console.print(err(f"vocab_size = {actual:,}  expected {expected:,}"))
        record("fail")
    console.print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 03 — special tokens
# ─────────────────────────────────────────────────────────────────────────────
def test_special_tokens(tok):
    console.print(Rule("[bold cyan]03  Special Tokens[/]", style="cyan"))
    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    t.add_column("role",   style="dim",    width=10)
    t.add_column("token",  style="yellow", width=10)
    t.add_column("id",     style="cyan",   width=8)
    t.add_column("status",                 width=12)

    all_ok = True
    for role, token, tid in [
        ("bos", tok.bos_token, tok.bos_token_id),
        ("eos", tok.eos_token, tok.eos_token_id),
        ("pad", tok.pad_token, tok.pad_token_id),
        ("unk", tok.unk_token, tok.unk_token_id),
    ]:
        present = token is not None and tid is not None
        if not present: all_ok = False
        t.add_row(role, str(token), str(tid),
                  "[green]present[/]" if present else "[red]MISSING[/]")
    console.print(t)

    ids = tok("অসমীয়া ভাষা")["input_ids"]
    console.print(ok("BOS prepended") if ids[0]  == tok.bos_token_id else err("BOS NOT prepended"))
    console.print(ok("EOS appended")  if ids[-1] == tok.eos_token_id else err("EOS NOT appended"))
    console.print()
    record("pass" if all_ok else "fail")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 04 — max length
# ─────────────────────────────────────────────────────────────────────────────
def test_max_length(tok, expected=2048):
    console.print(Rule("[bold cyan]04  Max Length[/]", style="cyan"))
    ml = tok.model_max_length
    if ml == expected:
        console.print(ok(f"model_max_length = {ml}"))
        record("pass")
    else:
        console.print(err(f"model_max_length = {ml}  expected {expected}"))
        record("fail")
    console.print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 05 — encoding & roundtrip (from file lines)
# ─────────────────────────────────────────────────────────────────────────────
def test_encoding_roundtrip(tok, lines):
    console.print(Rule("[bold cyan]05  Encoding & Roundtrip[/]", style="cyan"))
    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1))
    t.add_column("#",         style="dim",     width=4,  no_wrap=True)
    t.add_column("input",                      width=30, no_wrap=True, overflow="ellipsis")
    t.add_column("tokens",    style="magenta", width=32, no_wrap=True, overflow="ellipsis")
    t.add_column("len",       style="cyan",    width=5,  justify="right")
    t.add_column("roundtrip",                  width=10, justify="center")

    failed = 0
    for i, sent in enumerate(lines, 1):
        ids     = tok(sent, truncation=True)["input_ids"]
        tokens  = tok.convert_ids_to_tokens(ids)
        decoded = tok.decode(ids, skip_special_tokens=True).strip()
        match   = decoded == sent.strip()
        if not match: failed += 1
        tok_str = " ".join(tokens[:7]) + (" …" if len(tokens) > 7 else "")
        t.add_row(str(i), sent.strip(), tok_str, str(len(ids)),
                  "[green]✓[/]" if match else "[red]✗[/]")

    console.print(t)
    total = len(lines)
    passed = total - failed
    if failed == 0:
        console.print(ok(f"{passed}/{total} roundtrips passed"))
        record("pass")
    else:
        console.print(warn(f"{passed}/{total} roundtrips passed  ({failed} mismatches)"))
        record("warn")
    console.print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 06 — suffix / morphology splitting
# ─────────────────────────────────────────────────────────────────────────────
SUFFIX_WORDS = [
    ("অসমীয়াত",        "অসমীয়া + ত"),
    ("মানুহখিনিৰ",      "মানুহ + খিনি + ৰ"),
    ("শিক্ষাৰ্থীসকলে",  "শিক্ষাৰ্থী + সকল + ে"),
    ("চৰকাৰখনে",       "চৰকাৰ + খন + ে"),
    ("ৰাজ্যখনৰ",       "ৰাজ্য + খন + ৰ"),
]

def test_suffix_splitting(tok):
    console.print(Rule("[bold cyan]06  Suffix / Morphology Splitting[/]", style="cyan"))
    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1))
    t.add_column("word",     style="white",   width=24)
    t.add_column("expected", style="dim",     width=22)
    t.add_column("pieces",   style="magenta", width=30, overflow="ellipsis")
    t.add_column("n",        style="cyan",    width=4,  justify="right")
    t.add_column("split?",                    width=8,  justify="center")

    for word, expected in SUFFIX_WORDS:
        ids    = tok(word, add_special_tokens=False)["input_ids"]
        tokens = tok.convert_ids_to_tokens(ids)
        t.add_row(word, expected, " | ".join(tokens), str(len(tokens)),
                  "[green]✓[/]" if len(tokens) > 1 else "[dim]—[/]")

    console.print(t)
    console.print()
    record("pass")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 07 — UNK rate
# ─────────────────────────────────────────────────────────────────────────────
def test_unk_rate(tok, text):
    console.print(Rule("[bold cyan]07  UNK Rate[/]", style="cyan"))
    ids      = tok(text, truncation=False, add_special_tokens=False)["input_ids"]
    unk_cnt  = ids.count(tok.unk_token_id)
    unk_rate = unk_cnt / len(ids) * 100 if ids else 0

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim", width=18)
    t.add_column()
    t.add_row("sample chars",  val(f"{len(text):,}"))
    t.add_row("total tokens",  val(f"{len(ids):,}"))
    t.add_row("unk tokens",    val(str(unk_cnt)))
    t.add_row("unk rate",      val(f"{unk_rate:.4f}%"))
    console.print(t)

    if unk_rate < 0.1:
        console.print(ok(f"unk rate {unk_rate:.4f}%  < 0.1%  — excellent"))
        record("pass")
    elif unk_rate < 1.0:
        console.print(warn(f"unk rate {unk_rate:.4f}%  — consider larger vocab"))
        record("warn")
    else:
        console.print(err(f"unk rate {unk_rate:.4f}%  — too many unknowns"))
        record("fail")
    console.print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 08 — fertility (tokens per word)
# ─────────────────────────────────────────────────────────────────────────────
def test_fertility(tok, text):
    console.print(Rule("[bold cyan]08  Fertility  (tokens / word)[/]", style="cyan"))
    words     = text.split()
    ids       = tok(text, truncation=False, add_special_tokens=False)["input_ids"]
    fertility = len(ids) / max(len(words), 1)

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim", width=18)
    t.add_column()
    t.add_row("words",        val(f"{len(words):,}"))
    t.add_row("tokens",       val(f"{len(ids):,}"))
    t.add_row("tokens/word",  val(f"{fertility:.3f}"))
    console.print(t)

    if fertility < 1.2:
        console.print(warn("ratio < 1.2 — tokenizer may be memorising full words"))
        record("warn")
    elif fertility > 3.0:
        console.print(warn("ratio > 3.0 — vocab may be too small"))
        record("warn")
    else:
        console.print(ok(f"fertility {fertility:.3f} in healthy range  [1.2 – 3.0]"))
        record("pass")
    console.print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 09 — top 20 frequent tokens
# ─────────────────────────────────────────────────────────────────────────────
def test_top_tokens(tok, text):
    console.print(Rule("[bold cyan]09  Top 20 Frequent Tokens[/]", style="cyan"))
    ids    = tok(text, truncation=False, add_special_tokens=False)["input_ids"]
    counts = Counter(ids).most_common(20)
    special_ids = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}

    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    t.add_column("rank",  style="dim",     width=6,  justify="right")
    t.add_column("id",    style="cyan",    width=8,  justify="right")
    t.add_column("token", style="magenta", width=22)
    t.add_column("count", style="white",   width=10, justify="right")
    t.add_column("type",  style="dim",     width=12)

    for rank, (tid, cnt) in enumerate(counts, 1):
        token = tok.convert_ids_to_tokens([tid])[0]
        kind  = "[yellow]special[/]" if tid in special_ids else \
                ("[dim]subword[/]"   if "##" in token or token.endswith("</w>") else "word")
        t.add_row(str(rank), str(tid), token, f"{cnt:,}", kind)

    console.print(t)
    console.print()
    record("pass")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — batch encoding
# ─────────────────────────────────────────────────────────────────────────────
def test_batch(tok, lines):
    console.print(Rule("[bold cyan]10  Batch Encoding[/]", style="cyan"))
    try:
        enc = tok(lines[:8], padding=True, truncation=True,
                  max_length=128, return_tensors="pt")
        console.print(ok(f"batch size           {enc['input_ids'].shape[0]}"))
        console.print(ok(f"padded seq length    {enc['input_ids'].shape[1]}"))
        console.print(ok(f"input_ids shape      {list(enc['input_ids'].shape)}"))
        console.print(ok(f"attention_mask shape {list(enc['attention_mask'].shape)}"))
        record("pass")
    except Exception as e:
        console.print(err(f"batch encoding failed: {e}"))
        record("fail")
    console.print()

# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD
# ─────────────────────────────────────────────────────────────────────────────
def print_scorecard(elapsed):
    console.print(Rule(style="cyan"))
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 3))
    t.add_column(width=14)
    t.add_column(justify="right", width=6)
    t.add_row("[bold green]passed[/]",    f"[bold green]{results['pass']}[/]")
    t.add_row("[bold red]failed[/]",      f"[bold red]{results['fail']}[/]")
    t.add_row("[bold yellow]warnings[/]", f"[bold yellow]{results['warn']}[/]")
    console.print(t)

    total = sum(results.values())
    if results["fail"] == 0 and results["warn"] == 0:
        console.print(f"[bold green]All {total} tests passed.[/]  [dim]({elapsed:.2f}s)[/]\n")
    elif results["fail"] == 0:
        console.print(f"[bold yellow]{total} tests done — {results['warn']} warning(s), no failures.[/]  [dim]({elapsed:.2f}s)[/]\n")
    else:
        console.print(f"[bold red]{results['fail']} failure(s) — review output above.[/]  [dim]({elapsed:.2f}s)[/]\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Assamese tokenizer test suite v0.1.2")
    p.add_argument("--tokenizer_dir", default="./assamese_tokenizer",
                   help="Directory with saved tokenizer artifacts")
    p.add_argument("--sample_file",   required=True,
                   help="Assamese .txt file to test on")
    p.add_argument("--max_lines",     type=int, default=20,
                   help="Lines for encoding/roundtrip table (default 20)")
    p.add_argument("--sample_chars",  type=int, default=100_000,
                   help="Chars for UNK/fertility/top-token tests (default 100k)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.tokenizer_dir):
        console.print(f"[red]tokenizer dir not found: {args.tokenizer_dir}[/]"); sys.exit(1)
    if not os.path.isfile(args.sample_file):
        console.print(f"[red]sample file not found: {args.sample_file}[/]"); sys.exit(1)

    console.print()
    console.print(Rule("[bold cyan]Assamese Tokenizer Test Suite  v0.1.2[/]", style="cyan"))
    console.print(f"  tokenizer : [dim]{args.tokenizer_dir}[/]")
    console.print(f"  file      : [dim]{args.sample_file}[/]\n")

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    with open(args.sample_file, encoding="utf-8") as f:
        raw = f.read()

    lines      = [l.strip() for l in raw.splitlines() if l.strip()]
    text       = raw[:args.sample_chars]
    test_lines = lines[:args.max_lines]

    console.print(f"  total lines : [white]{len(lines):,}[/]  "
                  f"using first [white]{len(test_lines)}[/] for encoding table")
    console.print(f"  total chars : [white]{len(raw):,}[/]  "
                  f"using first [white]{len(text):,}[/] for stats\n")

    t0 = time.time()

    test_summary(tok)
    test_vocab_size(tok)
    test_special_tokens(tok)
    test_max_length(tok)
    test_encoding_roundtrip(tok, test_lines)
    test_suffix_splitting(tok)
    test_unk_rate(tok, text)
    test_fertility(tok, text)
    test_top_tokens(tok, text)
    test_batch(tok, test_lines)

    print_scorecard(time.time() - t0)


if __name__ == "__main__":
    main()