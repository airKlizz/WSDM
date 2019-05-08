'''
CREATE THE DATASET FROM THE CSV FILES
'''
print("Load csv files")

import csv
import pickle

import re, unicodedata
import nltk
import inflect
from nltk import word_tokenize

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

import numpy as np

from collections import Counter

data_directory = "../Data"

train_dataset_file_path = data_directory+"/train_dataset_combine_sampling"

create_test_dataset = False

train_file_path = data_directory+"/train.csv"
test_file_path = data_directory+"/test.csv"
embedding_file_path = data_directory+"/glove.6B.100d.txt"
test_dataset_file_path = data_directory+"/test_dataset"


embedding_dim = 100
max_sen_len = 30

X_train = []
y_train = []

X_test = []

with open(train_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X_train.append([row[i] for i in [5, 6]])
        y_train.append(row[7])

X_train = X_train[1:30000]
y_train = y_train[1:30000]

with open(test_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X_test.append([row[i] for i in [5, 6]])

X_test = X_test[1:3000]

'''
Data preprocessing
'''
print("data preprocessing")

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
            embedding = np.array([float(s) for s in line[1:]])
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    return word_embedding, words_dict

wordset = set()

i = 0

for line in X_train:
    print(i, "/", len(X_train))
    i +=1
    line[0] = preprocessing(line[0])
    line[1] = preprocessing(line[1])
    for word in line[0]:
        wordset.add(word)
    for word in line[1]:
        wordset.add(word)

print("HERE2")

for line in X_test:
    line[0] = preprocessing(line[0])
    line[1] = preprocessing(line[1])
    for word in line[0]:
        wordset.add(word)
    for word in line[1]:
        wordset.add(word)

print("HERE2")

word_embedding, words_dict = load_embedding(embedding_file_path, wordset, embedding_dim)

print("HERE3")

no_word_vector = np.zeros(embedding_dim)

print("HERE4")

for line in X_train:

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[0]) and line[0][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[0][i]]])
        else :
            sentence.append(no_word_vector)
    line[0] = np.array(sentence)

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[1]) and line[1][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[1][i]]])
        else :
            sentence.append(no_word_vector)
    line[1] = np.array(sentence)

argmax_y_train = []

print("HERE5")

for i in range(len(y_train)):
    if y_train[i] == 'agreed':
        argmax_y_train.append(0)
    elif y_train[i] == 'disagreed':
        argmax_y_train.append(1)
    else :
        argmax_y_train.append(2)

if create_test_dataset:
    for line in X_test:

        sentence = []
        for i in range(max_sen_len):
            if i < len(line[0]) and line[0][i] in words_dict:
                sentence.append(word_embedding[words_dict[line[0][i]]])
            else :
                sentence.append(no_word_vector)
        line[0] = np.array(sentence)

        sentence = []
        for i in range(max_sen_len):
            if i < len(line[1]) and line[1][i] in words_dict:
                sentence.append(word_embedding[words_dict[line[1][i]]])
            else :
                sentence.append(no_word_vector)
        line[1] = np.array(sentence)

'''
DATASET RE-SAMPLING
'''

print("Original dataset shape ", Counter(argmax_y_train))
   
sm = SMOTE()
rus = RandomUnderSampler()
#sme = SMOTEENN()

X_train_reshape = np.reshape(X_train, (-1, 2*max_sen_len*embedding_dim))

batch_size = 5000
nb_batch_per_epoch = int(len(X_train_reshape)/batch_size+1)

for batch in range(nb_batch_per_epoch):
    print(batch, "/", nb_batch_per_epoch)
    idx_min = batch * batch_size
    idx_max = min((batch+1) * batch_size, len(X_train_reshape)-1)
    
    if batch == 0:
        X_train_reshape_res, y_train_res = sm.fit_resample(X_train_reshape[idx_min:idx_max], argmax_y_train[idx_min:idx_max])
    else :
        '''
        if batch%3 == 0:
            print("SMOTEENN")
            X_batch_res, y_batch_res = sme.fit_resample(X_train_reshape[idx_min:idx_max], argmax_y_train[idx_min:idx_max])
        '''
        if batch%2 == 0:
            print("RandomUnderSampler")
            X_batch_res, y_batch_res = rus.fit_resample(X_train_reshape[idx_min:idx_max], argmax_y_train[idx_min:idx_max])
        else:
            print("SMOTE")
            X_batch_res, y_batch_res = sm.fit_resample(X_train_reshape[idx_min:idx_max], argmax_y_train[idx_min:idx_max])
        X_train_reshape_res = np.concatenate((X_train_reshape_res, X_batch_res), axis=0)
        y_train_res = np.concatenate((y_train_res, y_batch_res), axis=0)

X_train = np.reshape(X_train_reshape_res, (-1, 2, max_sen_len, embedding_dim))
print("Resampled dataset shape ", Counter(y_train_res))

y_train = []
for i in range(len(argmax_y_train)):
    if argmax_y_train[i] == 0:
        y_train.append([1, 0, 0])
    elif argmax_y_train[i] == 1:
        y_train.append([0, 1, 0])
    else :
        y_train.append([0, 0, 1])

print("shape")
print(np.shape(X_train))
print(np.shape(y_train))

'''
Split in train and test set
'''

test_percentage = 0.25

train_X = X_train[int(test_percentage*len(X_train)):]
train_y = y_train[int(test_percentage*len(y_train)):]
test_X = X_train[:int(test_percentage*len(X_train))]
test_y = y_train[:int(test_percentage*len(y_train))]

train_dataset = [train_X, train_y, test_X, test_y]

if create_test_dataset:
    test_dataset = [X_test]

'''
Save dataset
'''

with open(train_dataset_file_path, 'wb') as f:
    pickle.dump(train_dataset, f)

if create_test_dataset:
    with open(test_dataset_file_path, 'wb') as f:
        pickle.dump(test_dataset, f)
