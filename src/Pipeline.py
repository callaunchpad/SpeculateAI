import gensim

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
    return arr

def label(arr):
    return arr
