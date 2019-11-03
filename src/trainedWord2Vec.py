import gensim
#import pickle

model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sophiasong/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
word2vec = model.wv #holds the mapping between words and embeddings
model.wv.save_word2vec_format('word2vec.bin', binary = True)
#vocab = model.vocab
#print(type(word2vec))

#with open('vectors.bin', 'wb') as v:
	#pickle.dump(word2vec, v)
	#v.close()

#del model