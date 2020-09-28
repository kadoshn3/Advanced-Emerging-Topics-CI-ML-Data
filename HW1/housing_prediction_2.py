import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from d2l import tensorflow as d2l
import numpy as np

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

headers = list(all_features.columns)
for i in range(np.shape(all_features)[1]):
    for j in range(len(all_features)):
        print(all_features[headers[i]][j])

all_features[numeric_features] = all_features[numeric_features].fillna(0)