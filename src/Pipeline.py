import numpy as np
import os
import gensim

def tokenize(s, arr_len=100):
    tokenizedArr = ["START"]
    tokenizedArr += list(gensim.utils.tokenize(s))
    for _ in range(arr_len - len(tokenizedArr) - 1):
        tokenizedArr += ["PAD"]
    tokenizedArr += ["END"]
    return tokenizedArr

def vectorize(arr):
    return arr

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
