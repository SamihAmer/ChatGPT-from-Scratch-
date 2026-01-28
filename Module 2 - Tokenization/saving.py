import json

# if helpful:

def save_vocab(vocabulary, filename="./vocab.txt"):
    '''
    vocabulary - a list of words in the vocabulary
    '''
    f = open(filename, "w")
    f.writelines([v+"\n" for v in vocabulary])
    f.close()

def save_merges(merges, filename="./merges.json"):
    '''
    merges - a list of tuples: (str, str2) representing a merge
    '''
    with open(filename, 'w') as f:
        json.dump(merges, f)