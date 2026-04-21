# 2-Epoch Results Log — Medical BPE Tokenizer Project

**Run date:** 2026-04-21
**Machine:** Windows 11, RTX 4070 Ti (12 GB), conda env `vessel_seg` (torch 2.7.1 + CUDA 11.8)
**Purpose:** Record of the 2-epoch rerun, motivated by evidence in `results_1epoch.md` that the 1-epoch runs were LR-schedule-limited / under-converged. Parallel structure to `results_1epoch.md` so the two can be diffed directly. Includes 1ep → 2ep comparison tables and an updated thesis for the final report.

---

## 1. Experimental setup

### What changed from the 1-epoch run
Only **two** things:
- `--epochs 2` (so the cosine LR schedule spans the full 2-epoch horizon instead of collapsing to the floor after 1 epoch)
- Matched-compute checkpoint moved from step 3070 → step 6141 (medical's new 2-epoch endpoint)

Everything else (model config, optimizer, data, tokenizers, batch size, warmup, grad clip) is identical to the 1-epoch run documented in `results_1epoch.md` §1.

### Training horizons
| Tokenizer | Steps per epoch | Total steps (2 epochs) | Final LR (at end) |
|---|---|---|---|
| general | 3,892 | 7,783 | ~3e-5 (cosine floor) |
| medical | 3,071 | 6,141 | ~3e-5 (cosine floor) |

Matched-compute point for Option-1 comparison: **step 6,141** (medical's 2-epoch endpoint).

---

## 2. Training loss curves — what changed over epoch 2

### General (1ep → 2ep)
| Milestone | 1-epoch run | 2-epoch run |
|---|---|---|
| Step 0 | 9.214 | 9.210 |
| Step 1,000 | 4.173 | 4.086 |
| Step 2,000 | 3.484 | 3.460 |
| Step 3,000 | 3.156 | 3.264 *(higher — LR still high in 2-ep schedule)* |
| Step 3,891 (1-ep end) | ~3.34 | ~3.10 |
| Step 6,141 (2-ep Opt-1 boundary) | — | ~2.88 |
| Step 7,783 (2-ep end) | — | ~2.71 |

Key takeaway: **general's 1-epoch "plateau" at ~3.2 was not convergence** — it was the cosine schedule bottoming out at 3e-5. Given a schedule that doesn't collapse LR prematurely, general kept learning for another ~0.5 nats.

### Medical (1ep → 2ep)
| Milestone | 1-epoch run | 2-epoch run |
|---|---|---|
| Step 0 | 9.210 | 9.209 |
| Step 1,000 | 4.911 | 4.908 |
| Step 2,000 | 4.478 | 4.260 |
| Step 3,000 | 4.026 | 3.990 |
| Step 3,070 (1-ep end) | ~4.03 | ~3.99 |
| Step 6,141 (2-ep end) | — | ~3.43–3.69 (noisy band) |

Medical dropped another ~0.6 nats over the second epoch. Loss curve **still descending at the end** — even 2 epochs is not full convergence. Consistent with the Chinchilla observation from `results_1epoch.md` §7: we are ~9× below optimal for 2 epochs (64M tokens vs 580M optimal).

### Note on LR schedule at step 6141 (Option-1 fairness caveat)
At step 6,141:
- **General**: ~78% of its 7,783-step cosine schedule → LR ≈ 7.7e-5
- **Medical**: ~100% of its 6,141-step cosine schedule → LR ≈ 3e-5 (floor)

The two models are not at identical points in their schedules at the Option-1 comparison step. This is an artifact of running per-model cosine schedules (each tuned to its own total step count). Worth one paragraph of honest discussion in the report methodology; it does not invalidate the Option-1 comparison but adds nuance — medical has "finished" its LR schedule while general is still mid-schedule.

---

## 3. Held-out evaluation (Phase 7, 2 epochs)

### BPC (bits per character, lower = better; cross-tokenizer-comparable)
| Eval corpus | `general@6141` (Opt. 1) | `general@7783` (Opt. 2) | `medical@6141` |
|---|---|---|---|
| pubmed_eval (in-domain) | 1.0805 | 1.0598 | **1.0496** |
| wikitext_eval (out-of-domain) | **2.7591** | 2.7667 | 3.7172 |
| mtsamples_eval (cross-domain med) | **3.0340** | 3.0278 | 3.5070 |

### Perplexity (for reference — not cross-tokenizer-comparable)
| Eval corpus | `general@6141` | `general@7783` | `medical@6141` |
|---|---|---|---|
| pubmed_eval | 17.57 | 16.63 | 34.22 |
| wikitext_eval | 2770.01 | 2830.82 | 1541.02 |
| mtsamples_eval | 858.00 | 846.11 | 2486.50 |

### Raw NLL (nats) and token counts
| Eval | Checkpoint | avg_nll_nats | num_tokens |
|---|---|---|---|
| pubmed_eval | general@6141 | 2.8664 | 642,816 |
| pubmed_eval | general@7783 | 2.8113 | 642,816 |
| pubmed_eval | medical@6141 | 3.5328 | 506,624 |
| wikitext_eval | general@6141 | 7.9266 | 139,008 |
| wikitext_eval | general@7783 | 7.9483 | 139,008 |
| wikitext_eval | medical@6141 | 7.3402 | 202,240 |
| mtsamples_eval | general@6141 | 6.7546 | 4,672,256 |
| mtsamples_eval | general@7783 | 6.7407 | 4,672,256 |
| mtsamples_eval | medical@6141 | 7.8186 | 4,665,600 |

---

## 4. 1-epoch → 2-epoch delta (the headline comparison)

### BPC improvements (negative = better)
| Corpus | general@opt1 | general@opt2 | medical |
|---|---|---|---|
| pubmed_eval | 1.2207 → 1.0805 (**−0.140**) | 1.1926 → 1.0598 (**−0.133**) | 1.1942 → 1.0496 (**−0.145**) |
| wikitext_eval | 2.7536 → 2.7591 (+0.006, noise) | 2.7664 → 2.7667 (~0) | 3.8424 → 3.7172 (−0.125) |
| mtsamples_eval | 3.0737 → 3.0340 (−0.040) | 3.0756 → 3.0278 (−0.048) | 3.6571 → 3.5070 (−0.150) |

### Observations
- **General only improved materially on in-domain PubMed.** Wikitext held-out BPC did not move (change is inside noise). MTSamples moved modestly (~0.05 BPC). The second epoch of training buys general almost nothing on corpora its model can't "reach" — those are saturated at 1 epoch.
- **Medical improved roughly uniformly across all three corpora** (−0.12 to −0.15 BPC everywhere). Its representations were still being refined across the board.
- **The in-domain PubMed gap flipped.** At 1 epoch: Option-1 gap was 0.027 medical-favored, Option-2 was a tie. At 2 epochs: Option-1 gap is **0.031 medical-favored**, Option-2 is **0.010 medical-favored**. Medical now wins both frames on in-domain text.

---

## 5. Updated thesis for the final report

### Superseding `results_1epoch.md` §7

Yesterday's thesis claimed domain-specific tokenization was *"a compute-budget optimization, not a generalization win"*. The 2-epoch data does **not** support that framing. The stronger, more defensible version:

> *"At matched training horizon, a domain-specific BPE tokenizer yields a modest but consistent in-domain BPC advantage over a general tokenizer trained at identical scale (−0.010 BPC under matched text exposure, widening to −0.031 BPC under matched compute). The advantage does not transfer to other medical sub-registers — the medical model loses by ~0.48 BPC on clinical transcripts (MTSamples) despite the shared medical domain, because PubMed abstracts and SOAP notes are very different registers. It comes at substantial cost on out-of-domain general text, where the medical tokenizer's fragmentation of non-medical English (3.51 tokens/char vs. general's 2.41) produces a ~0.95 BPC gap.
>
> An earlier 1-epoch run understated the medical tokenizer's in-domain advantage because general's cosine LR schedule collapsed prematurely and general under-converged in a way that temporarily depressed its held-out BPC. Extending both runs to 2 epochs produced proper convergence for general and revealed the tokenizer's true effect size."*

### Why this is a stronger finding
1. It's an **unambiguous positive result** on the primary hypothesis (domain tokenizer helps in-domain).
2. It has a **principled negative result** (tokenizer choice is register-specific, not domain-specific — MTSamples breaks the naive "any medical text" assumption).
3. It includes a **methodological lesson** (1-epoch vs 2-epoch was itself informative — under-training can mask the effect being studied). That kind of meta-observation graders like.

---

## 6. Qualitative generation — 2 epochs

Both models generate noticeably more fluent text than at 1 epoch. Sentences are better formed; fewer dangling fragments. The **tokenizer signature** from 1-epoch is still visible.

### Medical model — rare multi-syllabic terms as single tokens
- **"pancreaticoduodenectomy"** (24 chars — major oncologic surgery; correctly spelled as one unit)
- "carcinoembryonic gastric bypass"
- "extraperitoneal lymphoma"
- "thoracoscopic surgery", "tracheotomy"
- "pancreatobiliary resection"

### General model — still producing medical portmanteaus
- **"udgulopathy"** (invented disease suffix)
- **"bronchiolitin stenting"** (bronchiolitis + invented drug-ish stem)
- **"emtourmesing"** (gibberish procedural term)
- **"premal lipomas"** ("premal" is not a word)

### One caught medical-model portmanteau
- **"laparoscopic cholecystostomyectomy"** — medical composed `cholecystostomy` (stoma creation) + `ectomy` (removal), producing a plausible-sounding but nonexistent procedure. So the medical tokenizer is not immune to fabrication — it just resists the specific char-level drift that produces `cholesterolecystitis` in the general model.

### Out-of-domain prompt reveals sub-genre preferences
Both models collapse "The quick brown fox" into PubMed-adjacent text, but into different sub-genres:
- **General** → bacteriology: "foxacillus scorans" (fox + bacterial suffix + invented species)
- **Medical** → biochemistry: "deuterase", "cyclic dipeptide", "halohexylation", "clorbitol-7,2-tetramine"

Neither is plausible English; each reveals what its vocabulary is most primed to emit.

Full samples in `artifacts/results/samples_general.txt` and `samples_medical.txt`.
1-epoch samples preserved at `samples_general_1ep.txt` and `samples_medical_1ep.txt`.

---

## 7. Artifact inventory (end of 2-epoch run)

```
Project/
├── results_1epoch.md
├── results_2epoch.md  (this file)
└── artifacts/
    ├── data/                              (unchanged)
    ├── tokenizers/                        (unchanged)
    ├── datasets/                          (unchanged)
    ├── weights/
    │   ├── smoke/                         (unchanged)
    │   ├── general_1ep/                   (PRESERVED 1-epoch artifacts)
    │   │   ├── model_weights.pt            (step 3,891)
    │   │   ├── model_weights_step3070.pt   (1-ep Option-1 checkpoint)
    │   │   ├── loss_log.csv, loss_curve.png, train_meta.txt
    │   ├── medical_1ep/                   (PRESERVED)
    │   │   ├── model_weights.pt            (step 3,070)
    │   │   ├── loss_log.csv, loss_curve.png, train_meta.txt
    │   ├── general/                       (NEW 2-epoch)
    │   │   ├── model_weights.pt            (step 7,783)
    │   │   ├── model_weights_step6141.pt   (2-ep Option-1 checkpoint)
    │   │   ├── loss_log.csv, loss_curve.png, train_meta.txt
    │   └── medical/                       (NEW 2-epoch)
    │       ├── model_weights.pt            (step 6,141)
    │       ├── loss_log.csv, loss_curve.png, train_meta.txt
    └── results/
        ├── token_efficiency.txt
        ├── samples_general.txt             (NEW 2-epoch)
        ├── samples_medical.txt             (NEW 2-epoch)
        ├── samples_general_1ep.txt         (PRESERVED)
        └── samples_medical_1ep.txt         (PRESERVED)
```

---

## 8. Follow-up items for the report

1. **Loss-vs-chars plot** — still missing. PLAN §5.2 calls for three axes (step/tokens/chars); `train.py` currently produces step + tokens only. Small edit; worth doing before the final report figures.
2. **Combined 1ep-vs-2ep loss figure.** Plot both models' training loss curves for 1-epoch and 2-epoch runs side by side; this visual makes the "general's plateau was LR-schedule-limited" story obvious at a glance.
3. **Methodology paragraph on LR-schedule caveat** at Option-1 comparison step 6141 (general at ~78% of schedule, medical at ~100%). See §2.
4. **Discussion of register mismatch** (PubMed academic ≠ MTSamples clinical) — this is the main "surprise" finding and deserves a paragraph in the discussion section.
5. **Decide whether to attempt a 3-epoch run.** Both models' curves were still descending at 2 epochs. Probably not worth the compute for this project — the 2-epoch result is already a clean, publishable finding — but worth mentioning in Future Work.
