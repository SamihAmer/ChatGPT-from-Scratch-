'''
Your assignment is to implement BPE in the following method. You can add
classes or other routines if you find them helpful. 

This method should save two output files:
./vocab.txt : a list of the final vocabulary in order, one entry per line
./merges.json : a list of tuples of merges, in order

NOTE: Typically these file extensions are reversed (the vocabulary is a
json file and the merge list is a txt file), but for our purposes this way seems
simplier.

Does not need to return anything.

-------

This should implement a GPT-style tokenizer which prefixes words with a space.
You can assume that the base vocabulary contains all single characters that will occur.
Treat punctuation (besides spaces) just like the other characters.

You do NOT need to worry about using a placeholder token in place of a space. 
You do NOT need to worry about special tokens (pad, bos, eos, unk, etc.). We have not covered these yet.

IMPORTANT: If there are ties while computing the merges, you should use lexigraphic order to resolve.
Points will be taken off if a different tie-break is used as it will not match the homework solution.

For example, if the pairs ('ab','out') and ('sp','ite') are tied for most occuring,
then "about" should be recorded before "spite".

'''

def train_tokenizer(txt_file, vocab_size, base_vocabulary):
    '''
    param : txt_file - a string path to a text file of data, i.e. "./data.txt"
    param : vocab_size - integer specifying the final vocab size
    param : base_vocabulary - list of strings to add to the vocabulary by default

    saves:
    ./vocab.txt : a list of the final vocabulary in order, one entry per line, ties broken alphabetically
    ./merges.json : a list of tuples of merges, in order
    '''

    from collections import Counter, defaultdict
    import heapq
    from saving import save_merges, save_vocab
 
    def _pairs_from_tokens(tokens):
        '''
        Helper function 1 - counting adjacent pairs in a token list
        '''
        pair_counts = {} # plain dict for pairs
        
        if len(tokens) < 2:  # if less than 2 tokens, there are no adjacent pairs
            return pair_counts
        
        prev = tokens[0] # previous token pointer at first token
        for tok in tokens[1:]:  # start iterating from second token 
            pair = (prev, tok) # creates current adjacent pair as a tuple
            pair_counts[pair] = pair_counts.get(pair, 0) + 1   # increments count for that pair in this word
            prev = tok  # current token becomes previous token for next step
        
        # returns dict of adjacent pair counts for this sequence of tokens
        return pair_counts   

    def _merge_tokens(tokens, left, right, merged):
        '''
        Helper function 2 - apply a merge to a token list 
        '''
        out = []  # output token list after merging
        i = 0     # index pointer for scanning through tokens
        n = len(tokens)    # stores length 

        # loop over the input tokens by index 
        while i < n:
            # checks if next two tokens match the merge pair 
            if i + 1 < n and tokens[i] == left and tokens[i + 1] == right:
                out.append(merged)
                i += 2 # skips both tokens we just merged to not overlap
            #not a mergeable pair boundary 
            else:
                out.append(tokens[i]) # copies current token unchanged
                i += 1    # move forward one token 
        
        # returns the merged token list
        return out

    vocab = []    # final vocabulary list in order 
    vocab_seen = set()    #set for fast checks 
    
    # iterates through provided base vocabulary list 
    for tok in base_vocabulary:
        if tok not in vocab_seen:  # avoid duplicates
            vocab.append(tok)     # add token to vocabulary list 
            vocab_seen.add(tok)    # add token to set of seen tokens 
    
    # if requested final vocab size is smaller than or equal to the base vocab size, should not merge
    if vocab_size <= len(vocab):   
        save_vocab(vocab, "./vocab.txt")
        save_merges([], "./merges.json")
        return    #exit training early 

    # opens training data file 
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read() #read entire file into memory as one string

    # splits on any whitespace and collapse runs of whitespace
    raw_words = text.split()

    # handles case where the file is empty or only whitespace
    if not raw_words:
        save_vocab(vocab, "./vocab.txt")
        save_merges([], "./merges.json")
        return

    # building gpt-style word units 
    word_freqs = Counter()  # counts how many times each word unit string occurs in corpus
    first = True

    # iterates through every whitespace-split word 
    for w in raw_words:
        if first:
            word_freqs[w] += 1  # counts first word without a leading space
            first = False  # after this point every word gets leading space
        else:
            word_freqs[" " + w] += 1  # counts word with real leading space character

    # flatten word_freqs into parallel arrays
    word_strings = list(word_freqs.keys())   # make a list of all unique word-unit strings
    freqs = [word_freqs[w] for w in word_strings]    # builds a list of frequencies to access word's count by index

    # per -word tokenization and pair count cache
    word_tokens = []  # stores current tokenizations for each unique word
    word_pair_counts = []   # stores a dict of adjacent pair counts in that word's current tokenization 

    # building lookup table 
    pair_total = defaultdict(int)   # mapping tokens (a,b) -> weighted count | lets BPE choose next merge
    pair_words = defaultdict(set)   # (a,b) -> {set of word indices} | updates only affected words when pair merging instead of rescanning 
    
    '''
    Below is fast counting approach from lecture slides 
    * count by scanning words and their pairs, not by enumerating all possible pairs and searching 
    '''
    # iterating through each unique word-unit string with it's index
    for idx, w in enumerate(word_strings):
        tokens = list(w)
        word_tokens.append(tokens)
        pairs = _pairs_from_tokens(tokens)
        word_pair_counts.append(pairs)

        freq = freqs[idx]   # gets how many times this word unit occurs in training data
        for pair, c in pairs.items():
            pair_total[pair] += c * freq
            pair_words[pair].add(idx)

    """
    Below is building heap to select best merge and also apply tie-break
    """
    # main heap of pairs keyed by count, smallest merge token string, and deterministic ordering 
    heap = []  

    # iterates over every pair currently present and its global weighted count 
    for (left, right), count in pair_total.items():
        if count > 0:
            merged = left + right
            # push a tuple to the heap
            # python compares tuples lexicographically so this solves the tie-break 
            heapq.heappush(heap, (-count, merged, left, right))
    # list that will record all merges in order
    merges = []

    """
    Main BPE Loop: repeatedly pick best pair, merge it, update counts 
    Continue until reached desired vocab size or there are no more pairs to merge. 
    """
    while len(vocab) < vocab_size and heap:
        # small heap cleaning loop to delete entries that are "stale"        
        while heap:
            neg_count, merged, left, right = heap[0]
            pair = (left, right)
            cur = pair_total.get(pair, 0)
            if cur == 0 or -neg_count != cur or merged != left + right:
                heapq.heappop(heap)
                continue
            break
        else:
            break
        
        # best pair to be chosen from heap top 
        pair = (left, right)
        count = pair_total.get(pair, 0)
        if count <= 0:
            break
        
        # Record merge and add merged token to vocab
        new_token = merged
        merges.append(pair)
        if new_token not in vocab_seen:
            vocab.append(new_token)
            vocab_seen.add(new_token)

        # find which words are affected by this merge
        affected = list(pair_words.get(pair, ()))
        if not affected:
            pair_total.pop(pair, None)
            continue

        for idx in affected:
            old_tokens = word_tokens[idx]
            new_tokens = _merge_tokens(old_tokens, left, right, new_token)
            if new_tokens == old_tokens:
                s = pair_words.get(pair)
                if s is not None:
                    s.discard(idx)
                    if not s:
                        pair_words.pop(pair, None)
                continue
            
            # recompute per-word pair counts and update global totals incrementally 
            old_pairs = word_pair_counts[idx]
            new_pairs = _pairs_from_tokens(new_tokens)

            # how oftern this word occurs in the dataset
            freq = freqs[idx]
            # Union old keys and new keys to handle edge cases 
            changed_pairs = set(old_pairs.keys())
            changed_pairs.update(new_pairs.keys())

            # iterate through all potentially affected pairs for this word
            for p in changed_pairs:
                old_c = old_pairs.get(p, 0)
                new_c = new_pairs.get(p, 0)
                if old_c != new_c:  # only do the work if count changed 
                    new_total = pair_total.get(p, 0) + (new_c - old_c) * freq
                    if new_total > 0:
                        pair_total[p] = new_total
                        l, r = p
                        heapq.heappush(heap, (-new_total, l + r, l, r))
                    else:
                        pair_total.pop(p, None) # if pair count = 0

            # update the inverted index pair_words  for pairs that appear/diseappear in this word
            for p in old_pairs.keys():
                if p not in new_pairs:
                    s = pair_words.get(p)
                    if s is not None:
                        s.discard(idx)
                        if not s:
                            pair_words.pop(p, None)

            for p in new_pairs.keys():
                if p not in old_pairs:
                    pair_words[p].add(idx)

            word_tokens[idx] = new_tokens # replace stored toneization for the word 
            word_pair_counts[idx] = new_pairs # replace stored per-word pair counts for the word

    # save results 
    save_vocab(vocab, "./vocab.txt")
    save_merges(merges, "./merges.json")





if __name__ == "__main__":

    # example of using this method.

    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'

    train_tokenizer("./data.txt", len(base)+1000, [c for c in base])
