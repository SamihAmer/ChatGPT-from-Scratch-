# Medical BPE Tokenizer — Final Project

**Author:** Samih Amer (individual project)
**Course:** GPT-from-Scratch
**Report:** `report.pdf` (in this folder)

This project trains a domain-specific Byte-Pair Encoding tokenizer on
medical text, plugs it into the GPT model from the course assignments,
and measures how it changes held-out performance compared to a general
tokenizer trained on Wikipedia. Three tokenizers are tested: a general
one (wikitext-103), a medical one (PubMed abstracts), and an MTSamples
one (clinical transcripts). The model architecture, optimizer, training
data, and hyperparameters are held fixed; only the tokenizer changes.

The headline finding is a **register-distinctness scaling**: the size
of a domain tokenizer's win is approximately a function of how distinct
its training register is from general English. See `report.pdf` for the
full writeup with figures, tables, and statistical analysis.

## Quick start

```bash
# one-time setup
pip install -r requirements.txt

# run the pipeline
python main.py --phase prepare_data        # download corpora (~5-15 min)
python main.py --phase train_tokenizers    # train 3 BPE tokenizers (~5-10 min, CPU)
python main.py --phase efficiency          # compression analysis, no GPT needed
python main.py --phase construct_datasets  # tokenize PubMed under each tokenizer
python main.py --phase smoke               # 100-step sanity check
python main.py --phase sanity_check        # tokenizer diagnostics

# multi-seed sweep (the slow part: ~3.7 hr on RTX 4070 Ti for 13 runs)
python main.py --phase train_sweep

# analysis
python aggregate.py                        # multi-seed summary, t-tests, figures
```

Or use `python main.py --phase all` to run every phase end to end.

## Files

### Code

| File | What it does |
|---|---|
| `main.py` | Pipeline driver. Each `--phase X` runs one stage of the experiment. |
| `gpt.py` | GPT model from the course assignments (unchanged). |
| `sampler.py` | Top-k / top-p sampler from Module 7 (unchanged). |
| `tokenizer.py` | Wrapper around HuggingFace's byte-level BPE. Same wrapper for all three tokenizers; only the training corpus differs. |
| `prepare_data.py` | Downloads PubMed, wikitext-103, and MTSamples and writes them as one-doc-per-line text files. |
| `construct_dataset.py` | Tokenizes a corpus and packs it into a numpy array of fixed-length training windows. |
| `train.py` | GPT training loop. Logs (step, tokens, chars, loss) so we can plot loss vs any of the three. |
| `evaluate.py` | Computes cross-entropy, perplexity, and bits-per-character on a held-out file. BPC is the only cross-tokenizer-comparable metric. |
| `token_efficiency.py` | Compression ratios and showcase tokenizations (no GPT required). |
| `generate.py` | Samples text continuations from a trained GPT. |
| `aggregate.py` | Reads the multi-seed eval CSV, computes means/stds and paired t-tests, writes summary markdown + figures. |
| `make_plots.py` | Produces the single-run figures referenced in the report. |

### Documentation

- `README.md` — this file
- `installation.txt` — environment setup notes
- `requirements.txt` — pip dependencies
- `report.pdf` — the final report

### Outputs (under `artifacts/`)

| Path | Contents |
|---|---|
| `data/` | Plain-text training and eval corpora (created by `prepare_data.py`) |
| `tokenizers/{general,medical,mtsamples}/` | Trained HF tokenizer files |
| `datasets/pubmed_train_*.npy` | Tokenized + packed training data, one per tokenizer |
| `weights/{general,medical}/` | Trained GPT weights (single-run) |
| `weights/{general,medical,mtsamples}_seed{1..N}/` | Weights from the multi-seed sweep |
| `results/eval_table.csv` | One row per (seed, tokenizer, checkpoint, eval corpus) |
| `results/aggregate_summary.md` | Means, stds, paired t-tests across the sweep |
| `results/aggregate_table.csv` | Same as above, in tidy CSV form |
| `results/figures/*.png` | All figures from the report |
| `results/samples_{general,medical}.txt` | Qualitative generation samples |

## Reproducing the report numbers

The 13-run sweep produces the numbers in Tables 4 and 5 of the report
and Figures 4 and 5:

```bash
python main.py --phase prepare_data
python main.py --phase train_tokenizers
python main.py --phase construct_datasets
python main.py --phase train_sweep        # ~3.7 hr on RTX 4070 Ti
python aggregate.py
```

`artifacts/results/eval_table.csv` is the source of truth; the markdown
summary, the tidy CSV, and the figures all derive from it.

## Notes on compute

- Training is GPU-bound. With CUDA on a consumer card (e.g. RTX 4070 Ti)
  one 2-epoch run takes ~13–17 min. The full 13-run sweep is ~3.7 hr.
- Without a GPU you can still run everything except the training (data
  prep, tokenizer training, compression analysis, evaluation of
  pre-saved weights). `train.py` falls back to MPS on Apple Silicon
  and CPU otherwise, but training on CPU will be very slow.
- The dataset downloads (PubMed especially) are around 1 GB.

## Why bits-per-character?

Per-token perplexity isn't comparable across tokenizers — a tokenizer
that compresses more makes each token harder to predict, so the loss
per token doesn't tell you whether the model got better at the
underlying text. BPC normalizes the model's surprisal to characters
of raw text, so two models with different tokenizers can be compared
on equal footing. See `evaluate.py` and the report's Background
section for the derivation.
