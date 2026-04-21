# Presentation Outline — Medical BPE Tokenizer Project
**Target length:** ~10 minutes, ~11 slides (≈ 55 seconds per slide)
**Audience:** Classmates familiar with LLMs but not with tokenizer specifics.

Each slide has a **title**, **core message** (the one idea the audience should leave that slide with), **visual**, and a **speaker notes** cue.

---

## Slide 1 — Title
- **Title:** *Does a Domain-Specific BPE Tokenizer Improve In-Domain Language Modeling?*
- **Subtitle:** An evaluation on medical text
- **Author:** Samih Amer
- **Visual:** Plain title slide. Maybe a small image of split tokens as a teaser (`cholecystectomy` vs `ch|ole|cy|st|ect|omy`).
- **Notes:** 20 sec — introduce self, say "I swapped the tokenizer in our course GPT and measured what changed. That's the whole project."

## Slide 2 — Why tokenizers matter
- **Core message:** The tokenizer *constrains what the model can learn*. A unit the tokenizer doesn't respect is a unit the model has to reconstruct from pieces.
- **Visual:** Big side-by-side — "General: `ch|ole|cy|st|ect|omy`" vs "Medical: `cholecystectomy`".
- **Notes:** 45 sec — "Before any training happens, the tokenizer decides what vocabulary the model will see. Medical text is full of terms general tokenizers fragment into 4–6 subwords. Claim: a domain-specific BPE should help the model learn these terms better. This project tests that claim."

## Slide 3 — Experimental design (one slide!)
- **Core message:** Only the tokenizer changes. Everything else — model, data, optimizer, schedule — is held fixed.
- **Visual:** A diagram or simple table:

  | Held fixed | Varied |
  |---|---|
  | 29M-param GPT (Module 6) | Tokenizer (general vs medical) |
  | Optimizer, LR, batch size | — |
  | Training corpus (100k PubMed) | — |
  | Eval corpora (3) | — |

- **Notes:** 60 sec — "Both tokenizers are 10k-vocab byte-level BPE. Only difference: general was trained on wikitext; medical on PubMed. Same pipeline otherwise. Eval on three corpora — in-domain PubMed, out-of-domain wikitext, and cross-domain clinical transcripts (MTSamples) — which I'll explain why matters in a minute."

## Slide 4 — The fairness problem
- **Core message:** Because each tokenizer compresses differently, you can't hold compute AND tokens AND text all constant at once. Two honest framings:
- **Visual:** Two columns:
  - **Option 1 — same compute** (same gradient updates). Medical sees more raw text per step.
  - **Option 2 — same text exposure** (one epoch of corpus each). General does more gradient updates.
- **Notes:** 45 sec — "Medical packs 27% more chars per token on PubMed, so one epoch is fewer steps. This forces a choice: do you fix compute or fix data? I report both. The primary eval metric is **bits-per-character**, which normalizes across tokenizers — per-token perplexity can't compare tokenizers because the token itself is different."

## Slide 5 — Result 1: Compression
- **Core message:** Medical wins 27% compression on in-domain, loses 31% on wikitext, **ties on MTSamples.** That MTSamples tie is the first clue that something interesting is happening.
- **Visual:** **Figure 1** (bar chart: chars/token per corpus, 2 bars per group).
- **Notes:** 45 sec — "Before training any model at all, you can already see the tokenizer's specialization. Medical wins big on PubMed, loses big on wikitext — expected. But it *ties* on MTSamples, even though MTSamples is medical text. Hold on to that — it'll come back."

## Slide 6 — A methodological surprise: the 1-epoch trap
- **Core message:** I ran 1 epoch first, saw a tie on in-domain BPC, almost concluded the tokenizer didn't help. Actually: general's loss curve had plateaued at ~3.2 because **cosine LR schedule had collapsed to its floor** — not because training was done.
- **Visual:** **Figure 3** (1-ep vs 2-ep loss curves overlaid). Emphasize the solid lines continuing well below the dashed endpoints.
- **Notes:** 60 sec — "This is the meta-lesson of the project. A short pilot showed a tie; I was tempted to call it done. But general's learning-rate had decayed to 10% of peak at the end of one epoch. When I extended the schedule to 2 epochs, general kept learning — and so did medical. Moral: *under-training can mask the effect you're studying.*"

## Slide 7 — Result 2: BPC on held-out (the headline)
- **Core message:** At 2 epochs with proper convergence, **medical wins on in-domain under BOTH fairness frames.** Option 1: −0.031 BPC. Option 2: −0.010 BPC. General wins on wikitext and MTSamples.
- **Visual:** **Figure 4** (BPC bar chart, 1-ep vs 2-ep side by side). Or just the 2-ep half for clarity.
- **Notes:** 60 sec — "Medical's advantage on PubMed held up under both frames once both models properly converged. The gap is modest — 0.01 bits/char under the stricter comparison — but consistent. On out-of-domain wikitext, general wins by ~0.95 BPC. On MTSamples, general wins by ~0.48 BPC. That MTSamples result is the second surprise."

## Slide 8 — The MTSamples surprise
- **Core message:** Medical is a medical tokenizer but loses on medical text — **because register matters more than domain.** PubMed = academic prose. MTSamples = clinical dictation. These are very different languages.
- **Visual:** Simple 2-column comparison of register styles:
  - PubMed: "Patients with acute myocardial infarction were randomly assigned..."
  - MTSamples: "Pt is a 54 y/o M w/ cp x 2 days, admitted for r/o MI..."
- **Notes:** 60 sec — "I naively assumed 'medical text is medical text.' It isn't. The medical tokenizer learned PubMed's formal academic vocabulary. It didn't learn `pt`, `y/o`, `r/o`, `w/`, all the clinical shorthand. Phase-1 compression analysis already hinted this — medical and general tied on MTSamples chars/token. The lesson is that tokenizer specialization is **register-specific, not domain-specific**."

## Slide 9 — Qualitative: the shape of errors
- **Core message:** Even at similar BPC, the two models **fail differently**, and the failure modes trace directly to tokenization.
- **Visual:** Two columns:
  - **General invents portmanteaus:** `cholesterolecystitis`, `udgulopathy`, `bronchiolitin stenting`, `emtourmesing`
  - **Medical emits real rare terms:** `pancreaticoduodenectomy`, `extraperitoneal lymphoma`, `carcinoembryonic gastric bypass`
- **Notes:** 45 sec — "Because general builds `cholecystectomy` from 6 subwords, it can swap in a wrong subword mid-word and produce `cholesterolecystitis`. Medical treats the whole word as one token — it either emits it correctly or not at all. Medical isn't immune to fabrication, but it can't do char-level drift. Different failure *shapes*, not just different error *rates*."

## Slide 10 — Conclusions
- **Core message:** Three bullets.
  1. **Domain-specific tokenizers work, modestly and narrowly** — in-domain win, out-of-domain cost.
  2. **Specialization is register-specific.** "Medical" is not a monolithic target.
  3. **Methodological: check convergence before comparing.** A bad LR schedule produced a misleading 1-epoch result.
- **Visual:** Clean text slide, bold the three claims.
- **Notes:** 45 sec — recap. Don't add new info here.

## Slide 11 — Future work + thank-you
- **Core message:** What I'd do with more compute / time.
- **Bullets:**
  - Mixed-corpus tokenizer (PubMed + MTSamples + general)
  - Register-aware tokenizer switching at inference
  - Chinchilla-scale training to see if the gap widens or closes
  - Tokenizer-conditioned hallucination analysis
- **Visual:** Small footer with "code + report: github.com/..., or see artifacts/"
- **Notes:** 30 sec — end on an open question. "Thanks, happy to take questions."

---

## Design guidance

- **Use the figures I already generated** (`artifacts/results/figures/*.png`): Figure 1, Figure 3, Figure 4 are the three you need for slides 5, 6, 7. The other two (Figure 2, 3-axis training loss; Figure 5, BPC delta) are good for backup slides if someone asks detail questions.
- **Keep text minimal.** Each slide should have at most ~15 words of bullet text + the visual. Let the speaker notes carry the nuance.
- **Colors consistent with figures:** blue = general, red = medical. Your slide deck should follow the same convention.
- **Backup slides (don't present, but have ready):**
  - Full BPC table (Table 4 from report)
  - Training loss 3-axis figure (Figure 2) — in case someone asks about "fair under what"
  - The `(logᵢ(e) · CE · N_tok / N_chars)` formula — in case someone asks about BPC
  - Chinchilla numbers (29M params × 20 tokens/param = 580M vs our 25–32M)

---

## How to build it with Claude

If you're using Claude.ai's **Artifacts**, the most flexible route is to ask Claude to build a **reveal.js HTML deck**. Paste this outline in, attach the 5 figure PNGs, and prompt something like:

> "Build a reveal.js presentation from this outline. Use a dark theme. Embed the five figures I'm attaching. Each slide should have a title and 2–3 bullets max; the visual should dominate. Include speaker notes via `<aside class="notes">` so I can present with them visible."

Claude will render the HTML in an artifact. You can then save the file, open it in a browser, and present from there.

Alternative — if you'd rather hand-edit: **Slidev** (markdown-based) or **Google Slides / PowerPoint** (just paste in the bullets and drag in the PNGs manually). For a 10-minute talk, the manual route is often faster than wiring up a slide framework.

---

## Rehearsal tip

Time yourself reading the speaker notes out loud. A ~10 min target means you should come in around 9:30 on a first read. The MTSamples + methodological surprise slides (6 and 8) are where the project differentiates itself — make sure you hit them, don't rush through.
