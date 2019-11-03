import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

'''
    Tokens:
        START - start token
        PAD - padding token
        END - end token
'''

def tokenize(s, arr_len=100):
    tokenizedArr = ["START"]
    tokenizedArr += list(gensim.utils.tokenize(s))
    for _ in range(arr_len - len(tokenizedArr) - 1):
        tokenizedArr += ["PAD"]
    tokenizedArr += ["END"]
    return tokenizedArr


def vectorize(arr):
    """Takes in a tokenized list of words, returns a list of word2vec vectors"""
    vectors = []
    for word in arr:
        if word != "PAD" or word != "END" or word != "START":
            dictionary = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
            vectors.append(dictionary[word])
    return vectors

def label(arr):
    return arr

vector = vectorize(["hello"])
print(vector)
