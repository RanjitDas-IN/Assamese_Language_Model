# Assamese LM Training Status and Epoch Inconsistency Report

Generated for Kaggle training review

Prepared from the training script and Kaggle log artifacts.

---

# 1. Executive Summary

The training run completed normally at the configured step budget, but the saved trainer state shows an epoch fraction that does not match the expected full-epoch progress.

The key evidence is that the dataset size and batch configuration mathematically produce 40,443 optimizer steps per epoch, and the training run reached that exact step count.

The misleading part is the `epoch` field in `trainer_state.json`, which reports approximately `0.25` instead of approximately `1.0`.

---

# 2. Evidence Collected

- `train.py` computes:

```python
total_steps = steps_per_epoch * num_epochs
max_steps = total_steps
```

- Training log shows:

```text
Total chunks (samples) = 1,294,152
```

for the training set.

- Training log shows:

```text
effective_batch_size = 32
gradient_accumulation = 1
```

- Training log shows:

```text
total_train_steps = 40443
```

and resume from:

```text
checkpoint-17500
```

- `trainer_state.json` for `checkpoint-17500`:

```json
{
  "global_step": 17500,
  "max_steps": 40443,
  "epoch": 0.1081789
}
```

- `trainer_state.json` for `checkpoint-40444`:

```json
{
  "global_step": 40444,
  "max_steps": 40443,
  "epoch": 0.2500046
}
```

---

# 3. Diagnosis

The step budget is internally consistent:

```text
1,294,152 chunks ÷ 32 effective batch size
= 40,443 optimizer steps per epoch
```

That matches the configured `max_steps` exactly.

Therefore, the trainer did not stop early.

The trainer stopped because it reached the planned step limit.

The inconsistent value is:

```json
"epoch": 0.2500046
```

Because training was resumed from a checkpoint, the epoch counter appears to have been carried forward in a way that does not match the current dataset accounting.

This appears to be a bookkeeping/reporting issue rather than a training failure.

---

# 4. Impact on the Model

- The model trained to the configured end of the run.
- The epoch display should not be used as the primary signal of training completion for this run.
- The observed generation issues:
  - Topic drifting
  - Weak semantic coherence
  - Text blending
  - Poor instruction following

are model-quality issues and not evidence that training terminated early.

---

# 5. Recommended Action

## Immediate

1. Trust `global_step` and `max_steps` for completion status.
2. Use `checkpoint-40444` as the completed base model from this run.

## Additional Training

If another full pass over the same data is desired:

```bash
torchrun --nproc_per_node=2 train.py --num_epochs 2
```

Resume from:

```text
checkpoint-40444
```

## Quality Improvements

To improve generation quality:

1. Further clean the corpus.
2. Remove duplicate articles.
3. Improve document boundary handling.
4. Add EOS separators between documents.
5. Continue base pretraining.
6. Perform instruction tuning after base pretraining stabilizes.

---

# 6. Current Model Problems

The current model exhibits:

## 1. Topic Drifting

Starts with a prompt such as:

```text
অসম
```

or

```text
নৰেন্দ্ৰ মোদী
```

but rapidly moves to unrelated topics.

## 2. Weak Semantic Coherence

Sentences are locally grammatical but do not maintain a consistent topic.

## 3. Text Blending

Multiple articles appear merged into a single continuation.

## 4. No Instruction Following

The model behaves as a next-token predictor rather than a question-answering assistant.

## 5. Unstable Knowledge Retrieval

The model recognizes entities but does not consistently generate correct facts about them.

---

# 7. Proposed Roadmap

## Phase 1 — Corpus Quality

- Deduplicate documents.
- Preserve document boundaries.
- Insert EOS between documents.
- Remove noisy boilerplate.

## Phase 2 — Additional Base Pretraining

Train for additional passes over the corpus:

```text
Epoch 1 → completed
Epoch 2 → recommended
```

Evaluate quality after Epoch 2.

## Phase 3 — Instruction Tuning

Create datasets such as:

```text
User: অসমৰ ৰাজধানী কি?
Assistant: দিছপুৰ।
```

and:

```text
প্ৰশ্ন: ...
উত্তৰ: ...
```

Train separately from base pretraining.

## Phase 4 — Retrieval-Augmented Generation (Optional)

For factual accuracy:

- Knowledge base
- Search
- RAG pipeline

instead of relying solely on parameter memory.

---

# 8. Final Conclusion

The training job completed successfully.

Evidence:

```text
global_step = 40444
max_steps   = 40443
```

The reported:

```text
epoch = 0.25
```

is inconsistent with the step accounting and should be treated as a reporting anomaly rather than proof that only 25% of the dataset was processed.

The run should be considered a completed one-pass training run according to the configured step budget.

Recommended next step:

```bash
torchrun --nproc_per_node=2 train.py --num_epochs 2
```

and continue evaluation from `checkpoint-40444`.