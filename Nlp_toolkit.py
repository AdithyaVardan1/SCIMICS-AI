import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize Porter Stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """Tokenize the input sentence using nltk."""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Apply stemming to the input word using Porter Stemmer."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """Create a bag-of-words representation for the given sentence."""
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag