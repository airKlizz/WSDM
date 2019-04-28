import pickle
import os

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

nb_words_max = -1
nb_words_mean = 0

for i in range(len(train_X)):
    nb_words_mean += len(train_X[i][3])
    nb_words_mean += len(train_X[i][4])
    if len(train_X[i][3]) > nb_words_max:
        nb_words_max = len(train_X[i][3])
    if len(train_X[i][4]) > nb_words_max:
        nb_words_max = len(train_X[i][4])

for i in range(len(test_X)):
    nb_words_mean += len(test_X[i][3])
    nb_words_mean += len(test_X[i][4])
    if len(test_X[i][3]) > nb_words_max:
        nb_words_max = len(test_X[i][3])
    if len(test_X[i][4]) > nb_words_max:
        nb_words_max = len(test_X[i][4])

nb_words_mean = nb_words_mean/(2*(len(train_X)+len(test_X)))

print('Number of words max :')
print(nb_words_max)
print('Number of words mean :')
print(nb_words_mean)

