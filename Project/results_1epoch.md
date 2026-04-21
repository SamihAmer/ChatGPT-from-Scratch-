# 1-Epoch Results Log — Medical BPE Tokenizer Project

**Run date:** 2026-04-20
**Machine:** Windows 11, RTX 4070 Ti (12 GB), conda env `vessel_seg` (torch 2.7.1 + CUDA 11.8)
**Purpose:** Snapshot of every result produced by the 1-epoch pipeline run. Keep for comparison against tomorrow's 2-epoch rerun and for use in the final report.

---

## 1. Experimental setup

### Shared model/training config (identical across both tokenizer variants)
| Setting | Value |
|---|---|
| Architecture | GPTModel (from Module 6, pre-LN) |
| `d_model` | 512 |
| `n_heads` | 8 |
| `layers` | 6 |
| `vocab_size` | 10,000 |
| `max_seq_len` | 256 |
| Parameter count | 29,283,088 |
| Optimizer | AdamW, lr=3e-4 |
| LR schedule | Cosine with 500-step warmup (decays to 10% of peak) |
| Batch size | 32 |
| Grad clip | 1.0 |
| Epochs | 1 |

### Data corpora (all prepared from HuggingFace `datasets`)
| File | Size | Purpose |
|---|---|---|
| `pubmed_train.txt` | 100,000 abstracts | GPT training (shared by both runs) |
| `pubmed_eval.txt` | 2,000 abstracts | In-domain held-out |
| `wikitext_train.txt` | 129,240 lines | General tokenizer training |
| `wikitext_eval.txt` | 1,298 lines | Out-of-domain (general English) held-out |
| `mtsamples_eval.txt` | 4,966 clinical transcripts | Cross-domain medical held-out (bonus — was flagged as risk in PLAN) |

### Tokenizers
Both trained with HuggingFace byte-level BPE, `vocab_size=10000`. Only the training corpus differed.

| Tokenizer | Training corpus | Words tokenized during training |
|---|---|---|
| `general` | wikitext_train | 151,792 unique words |
| `medical` | pubmed_train | 190,421 unique words |

Both reached ~9,743 merges before the vocab cap.

---

## 2. Token efficiency (Phase 3)

### Compression ratio (chars/token, higher = better compression for that tokenizer)
| Corpus | General | Medical | Medical's advantage |
|---|---|---|---|
| pubmed_eval (in-domain) | 3.82 | **4.85** | **+27%** |
| wikitext_eval (out-of-domain) | **4.17** | 2.85 | **−31%** |
| mtsamples_eval (cross-domain med) | 3.20 | 3.21 | ≈ tied |

### Token counts per corpus
| Corpus | Chars | General tokens | Medical tokens |
|---|---|---|---|
| pubmed_eval | 2,460,144 | 643,455 (321.7/doc) | 506,807 (253.4/doc) |
| wikitext_eval | 576,145 | 138,314 | 201,898 |
| mtsamples_eval | 15,006,542 | 4,685,678 | 4,679,044 |

### Showcase words — how each tokenizer splits medical terminology
| Word | General | Medical |
|---|---|---|
| cholecystectomy | ch\|ole\|cy\|st\|ect\|omy (6) | **cholecystectomy** (1) |
| laparoscopic | lap\|ar\|os\|c\|op\|ic (6) | **laparoscopic** (1) |
| intraoperative | int\|ra\|oper\|ative (4) | **intraoperative** (1) |
| myocardial | my\|oc\|ard\|ial (4) | **myocardial** (1) |
| electroencephalogram | elect\|ro\|ence\|ph\|alog\|ram (6) | electro\|encephal\|ogram (3) |
| pneumonia | p\|ne\|um\|onia (4) | **pneumonia** (1) |
| hypertension | hy\|per\|t\|ension (4) | **hypertension** (1) |
| metastasis | met\|ast\|as\|is (4) | **metastasis** (1) |
| aspirin | asp\|ir\|in (3) | **aspirin** (1) |
| ibuprofen | \|ib\|up\|ro\|f\|en (5) | ib\|up\|rof\|en (4) |
| MRI / CT / DICOM | char-by-char | char-by-char |
| pt / dx / hx | char-by-char | **pt** (1) / d\|x (2) / h\|x (2) |

**Observation:** `ibuprofen` fragments in both — PubMed abstracts rarely use consumer drug brand names.

---

## 3. Dataset packing (Phase 4)

Packed into `(N, 257)` `.npy` arrays (seq_len+1 for shifted targets).

| Tokenizer | Sequences | Total tokens | Steps per epoch (bs=32) |
|---|---|---|---|
| general | 124,542 | 32,007,294 | 3,892 |
| medical | 98,248 | 25,249,736 | 3,070 (actually 3,071; last batch partial) |

Both cover the **same raw 122M-character corpus**; medical just packs it into 21% fewer tokens.

---

## 4. Smoke test (Phase 5)

100-step training on medical dataset. Validated:
- Device = `cuda` ✓
- Model loads with 29,283,088 params ✓
- Loss dropped 9.22 → 7.26 (random baseline for vocab=10k is ln(10000) ≈ 9.21 → learning confirmed)
- `chars/token` estimate = 4.895 matches Phase 3 efficiency table (4.85) ✓

---

## 5. Full training (Phase 6)

Single epoch each. Log excerpts (every 100 steps; full CSV in `artifacts/weights/{name}/loss_log.csv`).

### General (3,892 steps, ~30–45 min on RTX 4070 Ti)
| Step | Loss | Tokens seen | Chars seen |
|---|---|---|---|
| 0 | 9.214 | 8,192 | 31,305 |
| 1,000 | 4.173 | 8.2M | 31.3M |
| 2,000 | 3.484 | 16.4M | 62.6M |
| **3,000** | **3.156** | **24.6M** | **93.9M** |
| 3,070 | ≈3.16 | — | — (**checkpoint saved**) |
| 3,500 | 3.117 | 28.7M | 109.6M |
| 3,800 | 3.339 | 31.1M | 119.0M |

Curve characteristic: **plateaued** from ~step 2,500 onward, oscillating in 3.1–3.4 band. LR approaching schedule floor (3e-5) by end.

### Medical (3,070 steps, ~25–35 min)
| Step | Loss | Tokens seen | Chars seen |
|---|---|---|---|
| 0 | 9.210 | 8,192 | 40,102 |
| 1,000 | 4.911 | 8.2M | 40.1M |
| 2,000 | 4.478 | 16.4M | 80.2M |
| 3,000 | 4.026 | 24.6M | 120.3M |

Curve characteristic: **still descending** at endpoint (4.25 → 4.03 in last 800 steps). Likely under-converged.

**Per-token loss NOT comparable across tokenizers** (each medical token carries ~27% more chars than each general token → harder per-token prediction task). BPC is the honest metric — see below.

---

## 6. Held-out evaluation (Phase 7)

Three checkpoints × three eval corpora = 9 evaluations.

### BPC (bits per character, lower = better; **only cross-tokenizer-comparable metric**)
| Eval corpus | `general@3070` (Opt. 1) | `general@3891` (Opt. 2) | `medical@3070` |
|---|---|---|---|
| pubmed_eval | 1.2207 | 1.1926 | **1.1942** |
| wikitext_eval | **2.7536** | 2.7664 | 3.8424 |
| mtsamples_eval | **3.0737** | 3.0756 | 3.6571 |

### Perplexity (tokenizer-dependent, for reference only)
| Eval corpus | `general@3070` | `general@3891` | `medical@3070` |
|---|---|---|---|
| pubmed_eval | 25.49 | 23.66 | 55.68 |
| wikitext_eval | 2726.67 | 2828.42 | 1973.18 |
| mtsamples_eval | 937.15 | 941.09 | 3475.24 |

### Raw NLL (nats) and tokens
| Eval | Checkpoint | avg_nll_nats | num_tokens |
|---|---|---|---|
| pubmed_eval | general@3070 | 3.2382 | 642,816 |
| pubmed_eval | general@3891 | 3.1637 | 642,816 |
| pubmed_eval | medical@3070 | 4.0196 | 506,624 |
| wikitext_eval | general@3070 | 7.9108 | 139,008 |
| wikitext_eval | general@3891 | 7.9475 | 139,008 |
| wikitext_eval | medical@3070 | 7.5874 | 202,240 |
| mtsamples_eval | general@3070 | 6.8428 | 4,672,256 |
| mtsamples_eval | general@3891 | 6.8470 | 4,672,256 |
| mtsamples_eval | medical@3070 | 8.1534 | 4,665,600 |

---

## 7. Interpretation of evaluation results

### On the fairness frames
- **Option 1 — same compute (step 3070 for both):** On in-domain PubMed, **medical wins by 2.3% BPC** (1.1942 vs 1.2207). This is the compute-efficiency claim.
- **Option 2 — same text exposure (full epoch):** On in-domain PubMed, result is essentially **tied** (1.1942 vs 1.1926). General used its extra 800 steps to recover from 1.2207 down to 1.1926 — closing the gap.

### Three-way narrative across eval sets
1. **PubMed (in-domain):** medical wins at matched compute; tied at matched text. Compute-efficiency story.
2. **Wikitext (out-of-domain general English):** general wins decisively (~1.1 BPC gap). Specialization cost — medical tokenizer fragments general English into 3.51 tokens/char vs 2.41 for general.
3. **MTSamples (cross-domain medical):** general wins by ~0.58 BPC — **surprising**. Two plausible causes:
   - MTSamples (clinical transcripts, SOAP notes) is a very different register from PubMed (academic abstracts), so the medical *model* is effectively out-of-register.
   - Medical *tokenizer* had no compression edge here (3.20 vs 3.21 chars/tok — tied in Phase 3).

### Defensible thesis for the report
*"A domain-specific BPE tokenizer yields a compute-efficiency advantage on in-domain text (2.3% lower BPC at matched training budget), but the advantage is narrow: it vanishes once the general tokenizer is trained to the same text-exposure budget, it is costly on out-of-domain general text, and it fails to transfer even within the same domain when the register shifts (academic → clinical). Domain-specific tokenization is a compute-budget optimization, not a generalization win."*

### Asterisks
- **Medical is likely under-converged.** Its loss curve was still descending at epoch end. The Option-2 tie could tip in medical's favor with more training (hence tomorrow's Option B 2-epoch run).
- **We are ~18× below Chinchilla-optimal** (29M params × 20 tok/param = ~580M tokens optimal; we trained on 25–32M). Both models are severely under-trained; overfitting risk is negligible even at 2 epochs.
- **General's plateau may be LR-limited, not data-limited.** Cosine schedule decays to 10% of peak (3e-5) by epoch end — very low learning rate. A fresh schedule could unlock further learning.

---

## 8. Qualitative generation (Phase 8)

Both models generate medical-flavored text with sentence-level incoherence (expected at 29M params + undertraining). The qualitative **tokenizer signature** is visible, though:

### Medical model — produces real, correctly-spelled clinical terms as intact units
- "endotracheal intubation", "elective appendectomy"
- "laparoscopic clipping system", "iliac vein", "tracheal collapse"
- "intramedullary", "lumbar kyphotic surgeries", "cervical dislocation"

### General model — invents medical-sounding portmanteaus
- **"cholesterolecystitis"** (cholesterol + cholecystitis — doesn't exist)
- **"patellarfin faigure"** (gibberish)
- "duodenal endothelioma" (real words, wrong combination)

**Mechanism:** medical treats `cholecystectomy` as one token (emits correctly or not at all); general stitches it from 6 subwords (`ch|ole|cy|st|ect|omy`), so a wrong subword mid-stream yields a novel fake word like `cholesterolecystitis`. **The tokenizer constrains the shape of the errors.**

### Sanity check
- Both models immediately convert "The quick brown fox" into medical content — strong domain adaptation, no linguistic capacity for out-of-domain prompts. Expected.

Full samples saved to `artifacts/results/samples_general.txt` and `artifacts/results/samples_medical.txt`.

---

## 9. Artifact inventory (end of 1-epoch run)

```
Project/
├── results_1epoch.md  (this file)
└── artifacts/
    ├── data/
    │   ├── pubmed_train.txt    (100k abstracts)
    │   ├── pubmed_eval.txt     (2k abstracts)
    │   ├── wikitext_train.txt  (129k lines)
    │   ├── wikitext_eval.txt   (1.3k lines)
    │   └── mtsamples_eval.txt  (4,966 transcripts)
    ├── tokenizers/
    │   ├── general/   (vocab=10,000, trained on wikitext)
    │   └── medical/   (vocab=10,000, trained on PubMed)
    ├── datasets/
    │   ├── pubmed_train_general.npy   (124,542 × 257)
    │   └── pubmed_train_medical.npy   (98,248 × 257)
    ├── weights/
    │   ├── smoke/     (100-step smoke test)
    │   ├── general/
    │   │   ├── model_weights.pt            (step 3,891)
    │   │   ├── model_weights_step3070.pt   (matched-compute checkpoint)
    │   │   ├── loss_log.csv
    │   │   ├── loss_curve.png
    │   │   └── train_meta.txt
    │   └── medical/
    │       ├── model_weights.pt  (step 3,070)
    │       ├── loss_log.csv
    │       ├── loss_curve.png
    │       └── train_meta.txt
    └── results/
        ├── token_efficiency.txt
        ├── samples_general.txt
        └── samples_medical.txt
```

---

## 10. Open items going into the 2-epoch rerun (Option B)

1. **Resume vs from-scratch.** Decide on LR schedule strategy (continued-pretraining w/ fresh warmup, vs full rerun with cosine spanning 2 epochs).
2. **Preserve current 1-epoch artifacts** — rename or move before the rerun overwrites them.
3. **Matched-compute checkpoint for 2-epoch run.** New boundary is step ≈ 6,141 (medical's new endpoint). Save a general checkpoint there for the Option-1 comparison.
4. **Expected outcomes:**
   - Medical BPC should drop noticeably on PubMed (its curve was still descending).
   - General BPC should drop modestly or flat (curve had plateaued, but a fresh schedule might unlock more).
   - If medical's 2-epoch PubMed BPC drops below general's 2-epoch PubMed BPC → Option-2 narrative flips to "medical wins even at matched text exposure."
   - Wikitext and MTSamples gaps likely stay large in general's favor — those are structural, not compute-limited.
5. **Report plots to add.** Loss-vs-chars (PLAN §5.2 calls for three axes; current plot saves step and tokens only).
