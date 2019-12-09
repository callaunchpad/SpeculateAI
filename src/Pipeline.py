import numpy as np
import gensim
from gensim.models import KeyedVectors
import string
from nltk.corpus import stopwords
import nltk

def tokenize(s, arr_len=100):
    tokenizedArr = ["START"]
    tokenizedArr += list(gensim.utils.tokenize(s))[:97]
    tokenizedArr += ["END"]
    for _ in range(arr_len - len(tokenizedArr) - 1):
        tokenizedArr += ["PAD"]
    return tokenizedArr


def vectorize(arr, dictionary=None):
    """Takes in a tokenized list of words, returns a list of word2vec vectors"""
    vectors = []

    if dictionary is None:
        dictionary = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)

    for word in arr:
        if word != "PAD" and word != "END" and word != "START":
            vectors.append(dictionary[word])
    return vectors

def label(arr, words_list):
    # Plus one for the END character
    all_vecs = np.eye(len(words_list) + 1)
    label_words = arr[1:]
    labels = [words_list.index(word) for word in label_words]
    masks = [0 if word in ["PAD", "END"] else 1 for word in label_words]
    return labels, masks

def clean_headline(headline):
    """
    Cleans a headline for analysis
    :param headline: The headline to clean
    :return: The cleaned headline
    """
    # Replace common problems like double dash or possessives
    headline = headline.replace("--", " ")
    headline = headline.replace("'s", "")

    # Remove punctuation for tokenizer
    translator = str.maketrans("", "", string.punctuation)
    headline = headline.translate(translator)

    # Lower case all letters
    headline = [word.lower() for word in headline.split(' ') if word.isalpha()]

    # Remove stop words
    stops = set(stopwords.words('english'))

    headline = " ".join([word for word in headline if word not in stops])


    return headline

def numerized_to_text(numerized_seq, vocabulary):
    """
    Returns a generated string from a sequence of tokens
    :param numerized_seq: The sequence of tokenized words
    :param vocabulary: The lookup table
    :return: A string corresponding to the numerized sequence
    """

    translated_words = []

    for index in numerized_seq:
        if vocabulary[index] == "PAD":
            continue

        translated_words.append(vocabulary[index])

    converted_string = " ".join(translated_words)

    return converted_string


def tokenized_to_numerized(tokenized_seq, dictionary):
    """
    Takes in a cleaned, tokenized sequence and numerizes the sequence for NLP
    :param tokenized_seq: The list of tokens
    :param dictionary: The lookup map for token to index
    :return: The numerized sequence
    """

    numerized_seq = []

    for token in tokenized_seq:
        try:
            numerized_seq.append(dictionary[token])
        except KeyError:
            numerized_seq.append(dictionary['UNK'])

    return numerized_seq


def merge_headlines(headlines):
    """
    Merge all the headlines in a list
    :param headlines: A list of headlines
    :return: Return a single string of headlings
    """

    output_str = ""

    for headline in headlines[:-1]:
        output_str += clean_headline(headline) + " END "

    output_str += headlines[-1]

    return output_str
