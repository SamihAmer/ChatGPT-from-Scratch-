# Medical BPE Tokenizer — Project Plan & Context

> Final project for the GPT-from-Scratch course. Author: Samih Amer.
> Proposal: `Proposal_Medical_Tokenizer.pdf` (submitted).
> Instructions: `project_instructions.md` (this folder).

---

## 1. Project Goal

Implement a **domain-specific Byte Pair Encoding (BPE) tokenizer** trained on medical text, swap it into the GPT we built across Modules 1–7, and evaluate how it changes model behavior compared to the general-purpose tokenizer the course used. The **only variable under study is the tokenizer** — architecture, training loop, optimizer, and hyperparameters stay identical between runs.

**Expected effect.** Medical text is dense with specialized terminology (`cholecystectomy`, `laparoscopic`), abbreviations (`MRI`, `CT`, `DICOM`, `pt`, `dx`, `hx`), and compound words. A general tokenizer fragments these into many subwords; a medical tokenizer should capture them as single tokens, producing shorter sequences and potentially better convergence / generation quality on medical text.

---

## 2. Baseline Architecture (what we're comparing against)

Pulled from Modules 5–7 and kept unchanged for the project:

| Component | Value / Source |
|---|---|
| Model class | `GPTModel` (from `Module 5 - Transformers/my_gpt.py`, finalized in `Module 6 - Training LLMS/gpt.py`) |
| `d_model` | 512 |
| `n_heads` | 8 |
| `layers` | 6 |
| `vocab_size` | 10,000 |
| `max_seq_len` | 256 |
| Parameter count | ~29.3M |
| Block | Pre-LN: `LN → MHA → +`, `LN → FC(4d) → ReLU → FC(d) → dropout(0.1) → +` |
| Attention | Custom causal MHA with one fused `qkv` projection |
| Optimizer | AdamW, lr=3e-4 |
| LR schedule | cosine with 500-step warmup (`train_model.py`) |
| Batch size | 32 |
| Grad clip | 1.0 |
| Loss | `CrossEntropyLoss` over `logits.transpose(1,2)` vs shifted targets |
| Sampler | `Module 7 - Sampling and Inference/sampler.py` (top-k / top-p + freq/presence penalties) |

The baseline tokenizer (`hftokenizer.py`) wraps HuggingFace's GPT-2 tokenizer and retrains it on our corpus at `vocab_size=10000, limit_alphabet=500`. Byte-level BPE.

---

## 3. Variation Design

Two tokenizers, both built with the **same HF byte-level BPE implementation** at `vocab_size=10,000`. Only the training corpus differs:

| Tokenizer | Training corpus | Artifact |
|---|---|---|
| `general` (baseline) | wikitext-103 | `artifacts/tokenizers/general/` |
| `medical` (variation) | PubMed abstracts | `artifacts/tokenizers/medical/` |

**Training data for both GPT runs: the same PubMed medical corpus.** Both GPTs see identical raw text; only the tokenization lens differs.

A second natural control (mentioned in the proposal) is the HF BPE retrained on the medical corpus — which *is* our `medical` tokenizer above. So the general-tokenizer GPT trained on medical text serves as the in-domain-data-but-out-of-domain-vocab baseline.

---

## 4. Datasets

| Use | Dataset | Size target | Source |
|---|---|---|---|
| GPT training corpus | PubMed abstracts (`scientific_papers/pubmed`) | ~50–100 MB text (~10–20M tokens) | HuggingFace `datasets` |
| In-domain held-out eval | PubMed abstracts, different split | ~5 MB | HuggingFace `datasets` |
| Cross-domain medical eval | MTSamples (clinical reports) | ~5k docs | Best-effort HF mirror; fallback to PubMed-held-out only |
| General-text eval | Wikitext-103 validation | ~1 MB | HuggingFace `datasets` |

All datasets are stored in `artifacts/data/` as plain text, one document per line. We do not commit them to git (they're large and reproducible).

---

## 5. Evaluation Plan

### 5.1 Token efficiency (first comparison)
For each tokenizer, encode:
- held-out PubMed abstracts
- held-out wikitext
- a small list of "showcase" medical words (`cholecystectomy`, `laparoscopic`, `MRI`, …)

Report **average tokens per document** and **compression ratio** (chars ÷ tokens) for each (tokenizer, corpus) pair. Also show side-by-side token splits for a handful of showcase words.

### 5.2 Training convergence
Train both GPTs with identical hyperparameters, on the same PubMed corpus. Log loss at every `log_interval=100` steps, and plot loss vs both:
1. **Training steps** (fair under "same compute")
2. **Tokens seen** (fair under "same data")
3. **Characters seen** (fair under "same text exposure" — important since medical tokenizer packs more chars per token)

### 5.3 Held-out perplexity — **must use BPC**
Per-token perplexity is **not comparable across tokenizers** because the units differ. We report:

- **Perplexity per token** (standard metric, for each model vs its own tokenization of held-out text)
- **Bits-per-character (BPC)**: `(log2(e) · CE_loss · num_tokens) / num_chars` — normalized to the raw text, comparable across tokenizers. This is the honest comparison.

Compute BPC on:
- Held-out PubMed (in-domain for both models' training data)
- Held-out wikitext (does specialization hurt general text?)
- MTSamples (if obtainable — tests generalization to a different medical subgenre)

### 5.4 Qualitative generation
Use the Module 7 `Sampler` (`top_p=0.8, freq_penalty=1.1, presence_penalty=1.1`) to generate continuations from a fixed set of prompts:
- `"Patient presents with"`
- `"CT of the abdomen shows"`
- `"The patient was administered"`
- `"Postoperative diagnosis:"`
- one neutral prompt like `"The quick brown fox"` as a sanity check

Save side-by-side outputs for the report.

---

## 6. File Layout

```
Project/
├── PLAN.md                     # This document
├── README.md                   # How to run (required deliverable)
├── requirements.txt            # Python deps
├── main.py                     # Orchestrator (runs full pipeline end-to-end)
├── gpt.py                      # GPTModel — copied verbatim from Module 6
├── sampler.py                  # Sampler — copied verbatim from Module 7
├── tokenizer.py                # HF BPE wrapper: train / load / encode / decode
├── prepare_data.py             # Download PubMed + wikitext, split, write plain-text files
├── construct_dataset.py        # Tokenize + pack into (N, seq_len+1) .npy
├── train.py                    # Device-agnostic training (CUDA → MPS → CPU)
├── evaluate.py                 # Perplexity + BPC on held-out
├── token_efficiency.py         # Compression comparison across tokenizers + corpora
├── generate.py                 # Qualitative sample generation
└── artifacts/                  # All outputs. Gitignored.
    ├── data/                   # pubmed_train.txt, pubmed_eval.txt, wikitext_eval.txt
    ├── tokenizers/{general,medical}/   # HF tokenizer folders
    ├── datasets/               # tokenized .npy packs, one per tokenizer
    ├── weights/                # model_general.pt, model_medical.pt
    └── results/                # loss curves, eval tables, generation samples
```

No subdirectories per script — everything at the top level of `Project/` except `artifacts/`. Mirrors the course module style.

---

## 7. Compute Considerations

| Machine | Expected time per model per epoch | Use for |
|---|---|---|
| Current M1 Pro (MPS) | ~4–8 hours (seq_len=256, batch=32, 29M params) | Data prep, tokenizer training, analyses, **smoke test** |
| CUDA GPU PC | ~30–60 min | Full GPT training runs |

**Plan:** do everything except the full training runs on the M1. Run a 100-step smoke test on MPS to validate the training loop end-to-end. Move to CUDA for the two real training runs. `train.py` auto-selects the device.

---

## 8. Status & Next Steps

### Phase A — Scaffolding (this session, on M1)
- [x] Lay out `Project/` structure and plan document
- [ ] Copy `gpt.py`, `sampler.py` from course modules into `Project/`
- [ ] Write `tokenizer.py`, `prepare_data.py`, `construct_dataset.py`, `train.py`, `evaluate.py`, `token_efficiency.py`, `generate.py`
- [ ] Write `requirements.txt`, `README.md`, `main.py`

### Phase B — Data + Tokenizers (on M1)
- [ ] Download PubMed abstracts + wikitext eval split
- [ ] Split PubMed into train / held-out
- [ ] Train `tokenizers/general` on wikitext-103 (reuse Module 6 data if present, else re-download)
- [ ] Train `tokenizers/medical` on PubMed train split
- [ ] Run `token_efficiency.py` — this is already a result for the report

### Phase C — Dataset Packing (on M1)
- [ ] Pack PubMed train using each tokenizer → `dataset_general.npy`, `dataset_medical.npy`
- [ ] Pack held-out PubMed and wikitext-val for evaluation

### Phase D — Smoke Test (on M1)
- [ ] Run `train.py` for 100 steps on one tokenizer, confirm loss decreases, weights save, plot renders

### Phase E — Real Training (on CUDA PC)
- [ ] Train GPT with general tokenizer, save `model_general.pt` + `loss_general.csv`
- [ ] Train GPT with medical tokenizer, save `model_medical.pt` + `loss_medical.csv`

### Phase F — Evaluation & Report (either machine)
- [ ] Run `evaluate.py` → PPL + BPC table across eval sets
- [ ] Run `generate.py` → side-by-side samples
- [ ] Build plots: loss-vs-step, loss-vs-tokens, loss-vs-chars
- [ ] Write the 3–5 page final report using sections from `project_instructions.md`
- [ ] Prepare ~10 min presentation

---

## 9. Key Risks / Open Questions

- **MTSamples availability.** Not a first-class HF dataset. We'll try a mirror; if it doesn't pan out, drop it and rely on held-out PubMed + wikitext for evaluation. Report should still be strong.
- **PubMed abstract length.** Abstracts are short (~200 words). Fine for `seq_len=256` — one or two abstracts fit per sequence with EOS separators. Matches how wikitext was packed.
- **BPC normalization math.** Be careful: `BPC = loss_in_nats · log2(e) · (num_tokens / num_chars)`. Derive once and unit-test on a known small example.
- **Vocab size fairness.** Keeping both at 10,000 matches the proposal's "control for vocabulary size" commitment. If we later want to explore a vocab-size ablation, that's a stretch goal.
