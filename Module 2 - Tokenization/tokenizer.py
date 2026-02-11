import json

'''
This class should be constructed with trained tokenizer data:
vocab_file : a string path to a vocab.txt file
merges_file : a string path to a merges.json file

The class should implement two methods:
encode(string): returns a list of integer ids (tokenized text)
decode(list_of_ids): returns a string re-assembled from token ids

You may assume that only a single sample is passed in at a time (no batching).
You can add additional methods, classes, etc as you find helpful.

Important: Our vocabulary and merges may include 
punctuation. Just treat all non-space characters equally.

---

Notes on validating your solution:

A good sanity check is that decode(encode(x)) should return x.

Additionally, make sure that the tokenizer is using the merges in order.
For example, if your merges contain: ("m","o"), ("s","e"), ("u","s"), then
"mouse" should be represented as mo|u|se.

'''

class Tokenizer:
    
    def __init__(self, vocab_file, merges_file):
        
        # Load vocab: line index = token ID
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = [line.rstrip('\n') for line in f] # I use rstrip because the space caharacter is saved as " \n" which is a valid token 
        
        # Reverse map: token string -> ID, for fast lookup during encode
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

        # Load merges: a list of [left, right] pairs in training order
        with open(merges_file, 'r', encoding='utf-8') as f:
            self.merges = json.load(f)

    def _apply_merge(self, tokens, left, right):
        """
        Scan a token list and merge every adjacent (left,right) pair
        """
        out = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens [i+1] == right:
                out.append(left+right)
                i += 2   # skip both tokens we just merged
            else:
                out.append(tokens[i])
                i += 1
        return out

    def encode(self, string):
        '''
        param string : a string to be encoded
        returns a list of integers (token ids)
        '''
        if not string:
            return []
        
        # gpt-style pre-tokenization
        raw_words = string.split()
        word_units = [raw_words[0]] + [' ' + w for w in raw_words[1:]]

        all_ids = []
        for unit in word_units:
            # Start each word unit as a list of individual characters
            tokens = list(unit)

            # Apply every merge in training order
            for left, right in self.merges:
                tokens = self._apply_merge(tokens, left, right)

            # Map each resulting token to its integer ID
            for tok in tokens:
                all_ids.append(self.token_to_id[tok])
        return all_ids


    def decode(self, list_of_integers):
        '''
        param list_of_integers : a list of token ids
        returns a string formed by decoding these ids.
        '''

        # Map IDs back to token strings and join
        # Because interior words already carry a leading space in their tokens,
        # a plain join reconstructs the original string exactly
        return ''.join(self.vocab[i] for i in list_of_integers)



if __name__ == "__main__":

    # example of using this class

    tok = Tokenizer("./vocab.txt", "./merges.json")
    x = tok.encode("Peter piper picked a peck of pickled peppers.")
    print(x)
    x = tok.decode(x)
    print(x) # should be our original text.
