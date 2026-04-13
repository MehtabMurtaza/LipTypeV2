# Post Error Correction Architecture (LipTypeV2 rebuild)

This document describes the post-processing error correction pipeline implemented in the rebuild codebase, aligned to the LipType paper section on error reduction.

---

## Overview

The pipeline has three stages:

1. **Recognizer output**  
   Raw decoded character sequence from LipType recognizer.

2. **Error reduction (DDA)**  
   A deep denoising autoencoder corrects noisy character sequences.

3. **Sequence decoder (spell + LM + threshold repair)**  
   Corrected characters are converted to words and refined with:
   - Norvig spell corrector
   - Bidirectional count-based trigram LM
   - Threshold-driven edit-distance candidate replacement (`tau1`, `tau2`)

---

## Character Space and Representation

- Vocabulary size: **28**
  - `a-z` -> `0..25`
  - space -> `26`
  - newline/end token -> `27`

- Sequence representation:
  - Sentences are converted to character IDs and chunked to fixed length `seq_len=28`.
  - Each chunk is converted to a one-hot matrix of shape `[28, 28]`.
  - DDA input/output uses flattened vectors of length `784` (`28*28`).

---

## DDA Module

### Architecture

- MLP: `784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784`
- Hidden activations: `tanh`
- Output activation: `sigmoid`
- Loss: `binary_crossentropy`

### Training Data Construction

From text corpus lines:

1. Normalize text (lowercase, keep `a-z` and spaces).
2. Build clean/noisy pairs.
3. Inject one character-level error per word using cyclic operations:
   - delete
   - transpose
   - replace
   - insert
4. Split train/test (default 80/20).
5. Convert to fixed one-hot chunk matrices.

### Implemented in

- `src/liptype_rebuild/postprocess/dda.py`
- `src/liptype_rebuild/postprocess/dda_data.py`
- CLI training command in `src/liptype_rebuild/cli/train.py` (`train dda`)

---

## Language Model Module

### Model

- Bidirectional count-based trigram LM:
  - forward trigram probability
  - backward trigram probability
  - combined score as product

### Implemented in

- `src/liptype_rebuild/postprocess/ngram_lm.py`
- CLI training command: `train lm`

---

## Repair/Decoder Module

Given text (optionally DDA-corrected):

1. Optional spell correction (`NorvigSpell`)
2. Tokenize words
3. For each word, compute combined LM word score:
   - if score >= `tau1`, keep word
   - else:
     - query dictionary candidates with edit distance <= `tau2` (BK-tree)
     - substitute candidate maximizing sentence-level LM score
4. Return repaired sentence

### Thresholds

- `tau1`: confidence threshold for flagging words as erroneous
- `tau2`: max edit distance for replacement candidate set

### Implemented in

- `src/liptype_rebuild/postprocess/repair.py`
- `src/liptype_rebuild/postprocess/bktree.py`
- `src/liptype_rebuild/postprocess/spell.py`

---

## Inference-Time Pipeline

Current `predict liptype` flow (when options are provided):

1. LipType decode -> raw text
2. Optional DDA correction (`--repair-dda-weights`)
3. Optional LM + dictionary + spell repair (`--repair-lm`, `--repair-dict`, `--repair-spell-corpus`)
4. Final repaired text output

Implemented in:

- `src/liptype_rebuild/cli/predict.py`

---

## Evaluation Flow for Repair Model

A repair evaluation command is provided:

- `eval repair`
  - input: TSV with `<reference>\t<hypothesis>`
  - optional DDA before repair
  - sweeps `tau1` and `tau2`
  - reports baseline WER and repaired WER

Implemented in:

- `src/liptype_rebuild/cli/eval.py`

---

## CLI Commands (high-level)

- Train DDA:
  - `python -m liptype_rebuild.cli.entrypoint train dda ...`
- Train LM:
  - `python -m liptype_rebuild.cli.entrypoint train lm ...`
- Predict with repair:
  - `python -m liptype_rebuild.cli.entrypoint predict liptype ... --repair-dda-weights ... --repair-lm ...`
- Sweep repair thresholds:
  - `python -m liptype_rebuild.cli.entrypoint eval repair ... --tau1-values ... --tau2-values ...`

See `CLI_RUNBOOK.md` for full runnable command examples.

