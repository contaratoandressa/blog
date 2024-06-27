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
    
    """Download dataset

    local:
    dataset`s local

    Returns:
    df:Dataframe

   """

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
    
    """Adjust dataset

    df:
    dataset

    seeds:
    fix random seeds

    value: 
    column to analyzed

    percent_train:
    percent of dataset to train

    output:
    wich return (1 = dataset`s train, 0 = dataset`s test and other = norm value)

    Returns:
    df:Dataframe

   """
        
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
   
   """
   Create the new dataset

   local:
   dataset`s local
   
   look_back:
   lag

   """

   dataX, dataY = [], []
   
   for i in range(len(dataset)-look_back-1):
     a = dataset[i:(i+look_back), 0]
     dataX.append(a)
     dataY.append(dataset[i + look_back, 0])

   return np.array(dataX), np.array(dataY)

# create dataframe 
# adjust!!!
def simulation_lstm(trainX, testX, trainY, testY, look_back, norm):

    """
    Construct several lstm`s models and return the scores

    train and test (X and Y):
    split dataset in train and test
   
    look_back:
    lag

    norm:
    norm value

    """

    a, b = np.repeat(range(100, 1000, 100), 10), list(range(1,10))*10
    output_data = pd.DataFrame()
    output_data['density'] = b
    output_data['epochs'] = a
    output_data['score_train'] = output_data['score_test'] = np.repeat(0, output_data.shape[0])

    for elem in range(0, len(output_data.density)):

        # create and fit the lstm
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back))) # lstm = 4 number of gates
        model.add(Dense(output_data.density[elem]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=output_data.epochs[elem], batch_size=1, verbose=2)


        # make predictions
        pred_train = model.predict(trainX)
        pred_test = model.predict(testX)

        # invert predictions
        pred_train = norm.inverse_transform(pred_train)
        pred_trainY = norm.inverse_transform([trainY])
        pred_test = norm.inverse_transform(pred_test)
        pred_testY = norm.inverse_transform([testY])

        # calculate root mean squared error
        score_train = np.sqrt(mean_squared_error(pred_trainY[0], pred_train[:,0]))
        print('Train Score: %.2f RMSE' % (score_train))
        score_test = np.sqrt(mean_squared_error(pred_testY[0], pred_test[:,0]))
        print('Test Score: %.2f RMSE' % (score_test))

        # save results
        output_data['score_train'][elem], output_data['score_test'][elem] = score_train, score_test

    return(output_data)