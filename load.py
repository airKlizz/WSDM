import pickle
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2"

data_directory = "../Data"

dataset_file_path = data_directory+"/dataset"

print('BEGINNING OF THE LOAD')

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

print('END OF THE LOAD')

print('Dataset length :')
print(len(dataset))
print('Dataset[0] length :')
print(len(dataset[0]))
print('Dataset[1] length :')
print(len(dataset[1]))
print('Dataset[2] length :')
print(len(dataset[2]))
print('Dataset[3] length :')
print(len(dataset[3]))

train_X = dataset[0]
train_y = dataset[1]
test_X = dataset[2]
test_y = dataset[3]

print(np.shape(train_X))
print(len(train_X[0][3]))
print(len(train_X[0][4]))
print(np.shape(train_y))
print(np.shape(test_X))
print(len(test_X[0][3]))
print(len(test_X[0][4]))
print(np.shape(test_y))
