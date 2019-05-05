'''
CREATE THE DATASET FROM THE CSV FILES
'''

import csv
import pickle

import re, unicodedata
import nltk
import inflect
from nltk import word_tokenize

import numpy as np

data_directory = "../Data"

train_file_path = data_directory+"/train.csv"
test_file_path = data_directory+"/test.csv"
embedding_file_path = data_directory+"/glove.6B.100d.txt"
dataset_file_path = data_directory+"/dataset"

embedding_dim = 100
max_sen_len = 30

X = []
y = []

sample_submission = []

with open(train_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X.append([row[i] for i in [0, 1, 2, 5, 6]])
        y.append(row[7])

X = X[1:]
y = y[1:]

'''
Data preprocessing
'''

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def preprocessing(sample):
    words = nltk.word_tokenize(sample)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    return words

def load_embedding(embedding_file_path, wordset, embedding_dim):
    words_dict = dict()
    word_embedding = []
    index = 1
    words_dict['$EOF$'] = 0
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'r',encoding="utf-8") as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            if line[0] not in wordset: continue
            embedding = [float(s) for s in line[1:]]
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    return word_embedding, words_dict

wordset = set()

for line in X:
    line[3] = preprocessing(line[3])
    line[4] = preprocessing(line[4])
    for word in line[3]:
        wordset.add(word)
    for word in line[4]:
        wordset.add(word)

word_embedding, words_dict = load_embedding(embedding_file_path, wordset, embedding_dim)

no_word_vector = np.zeros(embedding_dim)

for line in X:

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[3]) and line[3][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[3][i]]])
        else :
            sentence.append(no_word_vector)
    line[3] = np.array(sentence)

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[4]) and line[4][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[4][i]]])
        else :
            sentence.append(no_word_vector)
    line[4] = np.array(sentence)

for i in range(len(y)):
    if y[i] == 'agreed':
        y[i] = np.array([1, 0, 0])
    elif y[i] == 'disagreed':
        y[i] = np.array([0, 1, 0])
    else :
        y[i] = np.array([0, 0, 1])

'''
Split in train and test set
'''

test_percentage = 0.25

train_X = X[int(test_percentage*len(X)):]
train_y = y[int(test_percentage*len(y)):]
test_X = X[:int(test_percentage*len(X))]
test_y = y[:int(test_percentage*len(y))]

dataset = [train_X, train_y, test_X, test_y]

'''
Save the dataset
'''

with open(dataset_file_path, 'wb') as f:
    pickle.dump(dataset, f)