import pandas as pd 
import gensim 

model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sophiasong/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
word2vec = model.wv
del word2vec 

data = pd.read_csv('RedditNews.csv')
newsCol = data['News']
numHeadlines = len(newsCol)
words = []
for headline in range(numHeadlines):
	words.append(newsCol[headline].split()) #[['IMF', 'chief', 'backs', 'Athens', 'as', 'permanent', 'Olympic', 'host'], ['The', 'president', 'of', 'France', 'says', 'if', 'Brexit', 'won,', 'so', 'can', 'Donald', 'Trump']]

seenWords = {}

for line in words:
	for word in line:
		if word in seenWords:
			continue
		else:
			


