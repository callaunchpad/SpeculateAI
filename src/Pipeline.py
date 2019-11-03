import numpy as np
import os
import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

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
    words = open(os.getcwd() + "/wordList.txt", "r")
    words_list = [word.strip('\n') for word in words.readlines()]
    words.close()

    # Plus one for the END character
    all_vecs = np.eye(len(words_list) + 1)
    nonlabel = np.zeros(len(words_list) + 1)
    label_words = arr[1:] + ["END"]
    labels = [all_vecs[words_list.index(word)] if word not in ["END", "PAD"] else nonlabel for word in label_words]
    masks = [0 if word in ["END", "PAD"] else 1 for word in label_words]
    return labels, masks

