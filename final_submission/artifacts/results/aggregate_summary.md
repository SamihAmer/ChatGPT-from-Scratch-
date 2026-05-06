# Multi-seed sweep — aggregated results

## BPC mean ± std across seeds (lower is better)

| Tokenizer | Eval corpus | Checkpoint | n | Mean BPC | Std | Min | Max |
|---|---|---|---|---|---|---|---|
| general | mtsamples_eval | 6141 | 5 | 3.0232 | 0.0162 | 3.0008 | 3.0407 |
| general | mtsamples_eval | 7784 | 5 | 3.0244 | 0.0098 | 3.0135 | 3.0392 |
| general | pubmed_eval | 6141 | 5 | 1.0801 | 0.0022 | 1.0768 | 1.0819 |
| general | pubmed_eval | 7784 | 5 | 1.0591 | 0.0018 | 1.0562 | 1.0605 |
| general | wikitext_eval | 6141 | 5 | 2.7577 | 0.0206 | 2.7213 | 2.7712 |
| general | wikitext_eval | 7784 | 5 | 2.7597 | 0.0150 | 2.7358 | 2.7773 |
| medical | mtsamples_eval | 6141 | 5 | 3.5180 | 0.0389 | 3.4739 | 3.5736 |
| medical | mtsamples_eval | 6142 | 5 | 3.5180 | 0.0389 | 3.4739 | 3.5736 |
| medical | pubmed_eval | 6141 | 5 | 1.0499 | 0.0023 | 1.0466 | 1.0520 |
| medical | pubmed_eval | 6142 | 5 | 1.0499 | 0.0023 | 1.0466 | 1.0520 |
| medical | wikitext_eval | 6141 | 5 | 3.6962 | 0.0502 | 3.6379 | 3.7730 |
| medical | wikitext_eval | 6142 | 5 | 3.6962 | 0.0502 | 3.6379 | 3.7730 |
| mtsamples | mtsamples_eval | 6141 | 3 | 2.2583 | 0.0093 | 2.2479 | 2.2658 |
| mtsamples | mtsamples_eval | 7672 | 3 | 2.2462 | 0.0146 | 2.2294 | 2.2561 |
| mtsamples | pubmed_eval | 6141 | 3 | 1.0783 | 0.0010 | 1.0771 | 1.0790 |
| mtsamples | pubmed_eval | 7672 | 3 | 1.0592 | 0.0012 | 1.0579 | 1.0599 |
| mtsamples | wikitext_eval | 6141 | 3 | 3.2091 | 0.0161 | 3.1941 | 3.2262 |
| mtsamples | wikitext_eval | 7672 | 3 | 3.2018 | 0.0115 | 3.1904 | 3.2134 |

## Paired t-tests across tokenizers (lower BPC = better)

Pairs are matched by seed (seed 1 of A vs seed 1 of B, etc.). Positive `mean_diff` means tok_b wins (A's BPC > B's BPC). Statistical test: `scipy.stats.ttest_rel`; CI: t-interval on paired diffs.

| Comparison | tok_a step | tok_b step | n | Mean (a - b) | 95% CI | t | p | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| PubMed: general vs medical, Option 1 (matched compute, step 6141) | 6141 | 6141 | 5 | +0.0302 BPC | [+0.0264, +0.0340] | 22.029 | 0.0000 | +9.852 |
| PubMed: general vs medical, Option 2 (epoch end) | 7784 | 6142 | 5 | +0.0091 BPC | [+0.0055, +0.0128] | 7.038 | 0.0021 | +3.147 |
| wikitext: general vs mtsamples, epoch end | 7784 | 7672 | 3 | -0.4400 BPC | [-0.4713, -0.4086] | -60.324 | 0.0003 | -34.828 |
| MTSamples: medical vs mtsamples, epoch end | 6142 | 7672 | 3 | +1.2827 BPC | [+1.1426, +1.4229] | 39.378 | 0.0006 | +22.735 |
| MTSamples: general vs mtsamples, epoch end | 7784 | 7672 | 3 | +0.7807 BPC | [+0.7329, +0.8286] | 70.235 | 0.0002 | +40.550 |

Interpretation:
- **PubMed: general vs medical, Option 1 (matched compute, step 6141)**: medical wins by 0.0302 BPC on average across 5 matched seeds; p<0.05 (significant); effect size d=+9.85 (general mean=1.0801, medical mean=1.0499).
- **PubMed: general vs medical, Option 2 (epoch end)**: medical wins by 0.0091 BPC on average across 5 matched seeds; p<0.05 (significant); effect size d=+3.15 (general mean=1.0591, medical mean=1.0499).
- **wikitext: general vs mtsamples, epoch end**: general wins by 0.4400 BPC on average across 3 matched seeds; p<0.05 (significant); effect size d=-34.83 (general mean=2.7619, mtsamples mean=3.2018).
- **MTSamples: medical vs mtsamples, epoch end**: mtsamples wins by 1.2827 BPC on average across 3 matched seeds; p<0.05 (significant); effect size d=+22.73 (medical mean=3.5289, mtsamples mean=2.2462).
- **MTSamples: general vs mtsamples, epoch end**: mtsamples wins by 0.7807 BPC on average across 3 matched seeds; p<0.05 (significant); effect size d=+40.55 (general mean=3.0270, mtsamples mean=2.2462).
