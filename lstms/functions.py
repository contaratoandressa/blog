# packages
import pandas as pd
import numpy as np
from numpy import array
import os
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# dataset
def dataset_import(local = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'):
    
    df = pd.read_csv(local)

    # descriptives

    print("Dataset`s length: ")
    print(df.shape)

    print("Dataset`s columns: ")
    print(df.columns)

    print("Dataset`s summary: ")
    print(df["Passengers"].describe())

    print("Minimum date: ")
    np.min(df.Month)

    print("Maximum date: ")
    np.max(df.Month)

    print("Verifying the length of the data: ")
    print(len((pd.date_range('1949-01-01','1960-12-12', freq='MS').strftime("%b-%y").tolist())) == df.shape[0])

    return(df)

# prepar data to apply lstms
def prepar_data(df, seeds = 10, value = 'Passengers', percent_train = 0.75, output = 1):
    
    # fix random seed
    tf.random.set_seed(seeds)

    # filter values
    df = pd.DataFrame(df[value])
    df.astype('float32')

    # normalize the data
    norm = MinMaxScaler(feature_range= (0,1))
    df = norm.fit_transform(df)

    # split train and test
    train_size = int(len(df) * percent_train)
    test_size = len(df) - train_size
    train, test = df[0:train_size,:], df[train_size:len(df),:]
    
    print("length train and test: ")
    print(len(train), len(test))

    if output == 1:
        return train
    elif output == 0:
        return test
    else:
        return norm
    
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
   
   dataX, dataY = [], []
   
   for i in range(len(dataset)-look_back-1):
     a = dataset[i:(i+look_back), 0]
     dataX.append(a)
     dataY.append(dataset[i + look_back, 0])

   return np.array(dataX), np.array(dataY)