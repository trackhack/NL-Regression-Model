import os
import numpy as np
from random import *
import csv
import pandas as pd

def nonlin_func(input):
    y = 5*input[0] + input[2]*np.exp((input[0])/4) * input[2]/(input[1]+ 0.5)
    return y 

def get_x():
    x = []
    for i in range(3):
        x.append(randint(-10,10))
    return x

def fill(num_data): 
    with open(<ADD THE PATH TO A CSV FILE CONTAINING YOUR DATA>, 'w') as f:
        writer = csv.writer(f)
        for i in range(num_data):
            x = get_x()
            x_dist = []
            for j in range(3):
                dist = randint(-1,1)
                match dist:
                    case -1:
                        x_dist.append(x[j] - 0.1)
                    case 0:
                        x_dist.append(x[j])
                    case 1:
                        x_dist.append(x[j] + 0.1)
            y = nonlin_func(x_dist)
            x.append(y)
            writer.writerow(x)

fill(10000)


file = pd.read_csv(<ADD THE PATH TO A CSV FILE CONTAINING YOUR DATA>)
print("\nOriginal file:")
print(file)

headerList = ['x1', 'x2', 'x3', 'y']

file.to_csv(<ADD THE PATH AND NAME TO SAVE TO NEW FILE>', header = headerList, index = False)

file2 = pd.read_csv(<ADD THE PATH TO THE UPDATED CSV FILE CONTAINING YOUR DATA>')
print('\nModified file:')
print(file2)
