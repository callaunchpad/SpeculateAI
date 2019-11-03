import numpy as np
import os

def tokenize(s):
    return s

def vectorize(arr):
    return arr

def label(arr):
	words = open(os.getcwd() + "/wordList.txt", "r")
	words_list = [word.strip('\n') for word in words.readlines()]
	words.close()

	all_vecs = np.eye(len(words_list))
	pad_label = np.zeros(len(words_list))
	label_words = arr[1:] + ["pad"]
	labels = [all_vecs[words_list.index(word)] if word != "pad" else pad_label for word in label_words]
	masks = [1 if word == "pad" else 0 for word in label_words]
	return labels, masks
    