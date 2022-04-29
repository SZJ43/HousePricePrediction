# --coding:utf-8 --


import pandas as pd

from loadData import *

DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:8, [0, 1, 2, 3, 4, -3, -2, -1]])
print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
print(train_data.size)
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
