# The Full Pipeline: Building a GPT from Scratch


## Step 1 — download_data.py
Downloads wikitext-103 from HuggingFace: 1.8 million lines of cleaned Wikipedia text, ~103 million words. Each line is a paragraph or article title. Saves it as data.txt — raw text, one sample per line. This is our training corpus.

## Step 2 — hftokenizer.py
A language model can't read strings — it needs numbers. The tokenizer's job is to convert text → integers and back.

It trains a Byte Pair Encoding (BPE) tokenizer on our data. BPE works by:

Starting with every individual character as its own token
Repeatedly finding the most frequent adjacent pair of tokens and merging them into one new token
Repeating until you hit your vocabulary size (10,000 in our case)
So common words become a single token (" the" → token 45), while rare words get split ("Constantinople" → maybe 3-4 subword tokens). The result is a lookup table: any string of text maps to a sequence of integers from 0–9,999.

This tokenizer is trained on our specific dataset, meaning the 10,000 most useful tokens for Wikipedia-style text. A general model like GPT-4 has ~100k tokens.

## Step 3 — construct_dataset.py
Now we convert all 1.8M text samples into packed integer sequences ready for training.

Tokenize every line → list of integers, append EOS token at the end of each sample
Concatenate everything into one giant flat stream of ~130 million token IDs
Slice that stream into chunks of 257 (256 + 1)
Shuffle all chunks, save as dataset.npy
The +1 is the key insight: during training, the input is chunk[:256] and the target is chunk[1:257] — the same sequence shifted by one position. The model learns: given these 256 tokens, what's the next one? This is called next-token prediction (or causal language modeling).

## Step 4 — train_model.py
This is where learning actually happens.

Each training step:

Pull a batch of sequences from dataset.npy
x = batch[:, :-1] — the input (256 tokens)
y = batch[:, 1:] — the target (same 256 tokens, shifted right by 1)
Run x through the GPT → get logits of shape (B, 256, 10000) — a probability distribution over the vocab at every position
CrossEntropyLoss: at each of the 256 positions, how wrong was the model's prediction vs the actual next token?
Backpropagate the error, clip gradients, step the optimizer
The LR schedule warms up for 500 steps then cosine-decays — prevents early instability then lets the model fine-tune
The Whole Repo as Building Blocks

Modules 1-2:  Tokenization  →  text to integers
Modules 3-4:  Attention     →  how tokens relate to each other
Module 5:     Architecture  →  assemble into a full GPT model
Module 6:     Training      →  actually learn from real data
Every module built one piece. Module 6 is where they all connect: real data → tokenizer → packed sequences → GPT forward pass → loss → update weights.

The Numbers: 29,283,088 parameters and 15,863 steps
29.3M parameters comes directly from the architecture (d_model=512, n_heads=8, layers=6, vocab_size=10000, max_seq_len=256):

Component	Calculation	Params
Token embeddings	10,000 × 512	5,120,000
Position embeddings	256 × 512	131,072
Per block: MHA (qkv + wo)	3×512×512 + 512×512	1,048,576
Per block: MLP (fc1 + fc2)	512×2048+2048 + 2048×512+512	2,099,712
Per block: LayerNorms	512×2 × 2	2,048
× 6 blocks	3,150,336 × 6	18,902,016
Output projection	512×10,000 + 10,000	5,130,000
Total		29,283,088
15,863 steps per epoch is simpler:

dataset.npy contains ~507,616 sequences of length 257
With batch_size=32: 507,616 ÷ 32 = 15,863 batches per epoch
Each step processes 32 × 256 = 8,192 tokens
One full epoch processes ~130 million tokens
For reference, GPT-3 (175B params) was trained on ~300 billion tokens. We're training a 29M param model on ~130M tokens — tiny by industry standards, but enough to see the loss curve fall and the model learn statistical patterns of English.