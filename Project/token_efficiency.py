'''
Compare tokenizers on how efficiently they encode text. This is the first
headline result of the project and requires no trained GPT.

For each (tokenizer, corpus) pair reports:
  - total tokens
  - avg tokens per document
  - avg chars per token (compression ratio)
  - tokens per char

Also shows the token split of a handful of showcase medical words so the
difference is legible in a report or slide.

Usage:
  python token_efficiency.py
'''

import os
from tokenizer import HFTokenizer

TOKENIZERS = {
    "general": os.path.join("artifacts", "tokenizers", "general"),
    "medical": os.path.join("artifacts", "tokenizers", "medical"),
}

CORPORA = {
    "pubmed_eval":   os.path.join("artifacts", "data", "pubmed_eval.txt"),
    "wikitext_eval": os.path.join("artifacts", "data", "wikitext_eval.txt"),
    "mtsamples_eval": os.path.join("artifacts", "data", "mtsamples_eval.txt"),
}

SHOWCASE_WORDS = [
    "cholecystectomy",
    "laparoscopic",
    "intraoperative",
    "myocardial",
    "electroencephalogram",
    "pneumonia",
    "hypertension",
    "metastasis",
    "MRI",
    "CT",
    "DICOM",
    "pt",
    "dx",
    "hx",
    "aspirin",
    "ibuprofen",
]


def corpus_stats(tok, text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        docs = [line.rstrip("\n") for line in f if line.strip()]
    total_chars = 0
    total_tokens = 0
    for d in docs:
        total_chars += len(d)
        total_tokens += len(tok.encode(d))
    return {
        "docs": len(docs),
        "chars": total_chars,
        "tokens": total_tokens,
        "tokens_per_doc": total_tokens / max(len(docs), 1),
        "chars_per_token": total_chars / max(total_tokens, 1),
        "tokens_per_char": total_tokens / max(total_chars, 1),
    }


def print_table(results):
    header = f"{'corpus':<18}{'tokenizer':<10}{'docs':>8}{'chars':>12}{'tokens':>12}{'tok/doc':>10}{'char/tok':>10}"
    print(header)
    print("-" * len(header))
    for (corpus, tname), s in results.items():
        print(f"{corpus:<18}{tname:<10}{s['docs']:>8,}{s['chars']:>12,}{s['tokens']:>12,}"
              f"{s['tokens_per_doc']:>10.1f}{s['chars_per_token']:>10.2f}")


def showcase(tokenizers):
    print("\n=== showcase words: how each tokenizer splits medical terms ===")
    print(f"{'word':<24}" + "".join(f"{name:<40}" for name in tokenizers))
    print("-" * (24 + 40 * len(tokenizers)))
    for w in SHOWCASE_WORDS:
        row = f"{w:<24}"
        for name, tok in tokenizers.items():
            ids = tok.encode(" " + w)   # leading space to mimic word-boundary encoding
            pieces = [tok.decode([i]) for i in ids]
            disp = "|".join(pieces)
            if len(disp) > 38:
                disp = disp[:35] + "..."
            row += f"{disp:<40}"
        print(row)


def main():
    tokenizers = {}
    for name, path in TOKENIZERS.items():
        if not os.path.isdir(path):
            print(f"skipping {name}: {path} not found")
            continue
        t = HFTokenizer(path)
        t.load()
        tokenizers[name] = t

    if not tokenizers:
        print("No tokenizers available. Run prepare_data.py and train tokenizers first.")
        return

    results = {}
    for c_name, c_path in CORPORA.items():
        if not os.path.isfile(c_path):
            continue
        for t_name, tok in tokenizers.items():
            results[(c_name, t_name)] = corpus_stats(tok, c_path)

    print("=== token efficiency across (corpus, tokenizer) pairs ===\n")
    print_table(results)

    showcase(tokenizers)

    out_dir = os.path.join("artifacts", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "token_efficiency.txt")
    with open(out_path, "w") as f:
        f.write("corpus,tokenizer,docs,chars,tokens,tokens_per_doc,chars_per_token\n")
        for (c, t), s in results.items():
            f.write(f"{c},{t},{s['docs']},{s['chars']},{s['tokens']},"
                    f"{s['tokens_per_doc']:.2f},{s['chars_per_token']:.4f}\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
