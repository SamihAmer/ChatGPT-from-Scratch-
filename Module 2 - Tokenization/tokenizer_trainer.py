import json

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

    # TODO




if __name__ == "__main__":

    # example of using this method.

    base = "abcdefghijklmnopqrstuvwxyz"
    base += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base += "0123456789"
    base += "!@#$%^&*()_+-=[]{}|;':,.<>/?`~ "
    base += "\\"
    base += '"'

    train_tokenizer("./data.txt", len(base)+1000, [c for c in base])
