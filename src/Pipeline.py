import pickle
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sophiasong/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
word2vec = model.wv #holds the mapping between words and embeddings
#model.wv.save_word2vec_format('word2vec.bin', binary = True)
#vocab = model.vocab
with open('vectors.bin', 'wb') as v:
	pickle.dump(word2vec, v)
	v.close()

del model

'''
    Tokens:
        START - start token
        PAD - padding token
        END - end token
'''

def tokenize(s, arr_len=100):
    tokenizedArr = ["START"]
    tokenizedArr += list(gensim.utils.tokenize(s))
    tokenizedArr += ["END"]
    for _ in range(arr_len - len(tokenizedArr) - 1):
        tokenizedArr += ["PAD"]
    return tokenizedArr


def vectorize(arr):
	"""Takes in a tokenized list of words, returns a list of word2vec vectors"""
    file = 
    vectors = []
    for word in arr:
    	if word != 'pad' :
    		pickle_in = open('vectors.bin', "rb")
    		dictionary = pickle.load(pickle_in)
    		vectors.append(dictionary[word])
    return vectors

def label(arr):
    return arr



