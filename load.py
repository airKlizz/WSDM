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
