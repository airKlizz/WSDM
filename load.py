import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

data_directory = "../Data"

dataset_file_path = data_directory+"/dataset"

print('BEGINNING OF THE LOAD')

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

print('END OF THE LOAD')

print(len(dataset))
print(dataset[0])
