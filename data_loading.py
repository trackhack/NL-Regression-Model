import torch
import torchvision as torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataLoading:
    def __init__(self):
        self.filePath = "dataset/"   
        self.batchSize = 10
        self.num_workers = 0
    
    def load_data(self):
        file = pd.read_csv(<ADD THE PATH TO A CSV FILE CONTAINING YOUR DATA>)
        #input_data = file['input'].values.reshape(-1,1)
        #output_data = file['output'].values.reshape(-1,1)
        #my_df = pd.DataFrame(file)
        input = file.drop('y', axis=1).values
        output = file['y'].values

        #Standardize
        scaler = MinMaxScaler()
        input = scaler.fit_transform(input)

        #split dataset
        input_train, input_temp, output_train, output_temp = train_test_split(
            input, output, test_size = 0.4
        )
        input_val, input_test, output_val, output_test = train_test_split(
            input_temp, output_temp, test_size = 0.5 
        )
        #X_train, X_test, y_train, y_test = train_test_split(input, output, test_size = 0.2)

        #Convert into pytorch tensors
        input_train = torch.tensor(input_train, dtype = torch.float32)
        output_train = torch.tensor(output_train, dtype = torch.float32)
        input_val = torch.tensor(input_val, dtype = torch.float32)
        output_val = torch.tensor(output_val, dtype = torch.float32)
        input_test = torch.tensor(input_test, dtype = torch.float32)
        output_test = torch.tensor(output_test, dtype = torch.float32)

        #dataloader

        train_set = torch.utils.data.TensorDataset(input_train, output_train)
        train_loader = DataLoader(train_set, batch_size=self.batchSize, shuffle=True, num_workers=self.num_workers)

        val_set = torch.utils.data.TensorDataset(input_val, output_val)
        val_loader = DataLoader(val_set, batch_size=self.batchSize, shuffle=False, num_workers=self.num_workers)
        
        test_set = torch.utils.data.TensorDataset(input_test, output_test)
        test_loader = DataLoader(test_set, batch_size=self.batchSize, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader
