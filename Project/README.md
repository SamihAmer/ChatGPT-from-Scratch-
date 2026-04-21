# Medical BPE Tokenizer — Final Project

Domain-specific Byte Pair Encoding tokenizer trained on medical text, plugged
into the GPT model we built across Modules 1–7. This project isolates the
tokenizer as the single variable and measures its effect on token efficiency,
training convergence, perplexity, and qualitative generation.

See `PLAN.md` for the full context document, design decisions, and evaluation
methodology. See `Proposal_Medical_Tokenizer.pdf` for the graded proposal.

## Quick start

```bash
# one-time setup
pip install -r requirements.txt

# orchestrated pipeline (phases are independent, run one at a time)
python main.py --phase prepare_data          # ~5-15 min, downloads corpora
python main.py --phase train_tokenizers      # ~5-10 min, CPU-only
python main.py --phase efficiency            # seconds; first result
python main.py --phase construct_datasets    # ~5 min
python main.py --phase smoke                 # 100-step training smoke test
python main.py --phase train_all             # SLOW — run on CUDA machine
python main.py --phase evaluate              # after training
python main.py --phase generate              # after training

# or run every phase end-to-end
python main.py --phase all
```

## What's here

| File | Purpose |
|---|---|
| `PLAN.md` | Full project context, design, and roadmap |
| `gpt.py` | GPT model — copied verbatim from Module 6 |
| `sampler.py` | Inference sampler — copied verbatim from Module 7 |
| `tokenizer.py` | HF byte-level BPE wrapper (`train`, `load`, `encode`, `decode`) |
| `prepare_data.py` | Downloads PubMed + wikitext; tries MTSamples |
| `construct_dataset.py` | Tokenizes + packs a text file into `(N, 257)` `.npy` |
| `train.py` | Device-agnostic (CUDA → MPS → CPU) training loop |
| `evaluate.py` | CE loss, perplexity, and BPC on held-out text |
| `token_efficiency.py` | Compares tokenizers on compression ratio and showcase splits |
| `generate.py` | Generates samples from fixed prompts for qualitative comparison |
| `main.py` | Orchestrator; runs phases end-to-end |
| `artifacts/` | All outputs (data, tokenizers, datasets, weights, results). Gitignored. |

## Outputs produced

- `artifacts/data/*.txt` — cleaned plain-text corpora, one doc per line
- `artifacts/tokenizers/{general,medical}/` — trained HF tokenizers
- `artifacts/datasets/pubmed_train_{general,medical}.npy` — packed token sequences
- `artifacts/weights/{general,medical}/model_weights.pt` — trained GPT weights
- `artifacts/weights/{general,medical}/loss_log.csv` + `loss_curve.png`
- `artifacts/results/token_efficiency.txt` — compression table
- `artifacts/results/samples_{general,medical}.txt` — side-by-side generations

## Compute notes

Training the 29.3M-param GPT for one epoch over PubMed:
- **CUDA GPU** (e.g. RTX 4070+): ~30–60 min per model
- **Apple Silicon (MPS backend)**: ~4–8 hours per model

Everything except `train_all` (and, to a lesser extent, `smoke`) is fast on
CPU or MPS. The recommended flow is to run phases 1–4 on any machine, do a
short `smoke` test to validate the pipeline end-to-end, then move to a CUDA
machine for `train_all`.

`train.py` automatically selects the best available device.

## Key methodological note: BPC vs perplexity

Perplexity per token is **not comparable across tokenizers** because the unit
(one token) is different for each tokenizer. `evaluate.py` therefore reports
both per-token perplexity (for reference) and **bits-per-character (BPC)**
normalized to the raw text. BPC is the honest cross-tokenizer metric.

## Reproducing the experiments

Defaults follow the assignment baseline:
- `d_model=512, n_heads=8, layers=6, vocab_size=10000, max_seq_len=256`
- `batch_size=32, lr=3e-4, warmup=500, grad_clip=1.0, AdamW`

Any of these can be overridden via `train.py` flags.
