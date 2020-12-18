import pandas as pd
import numpy as np
import time

from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

start = time.time()

dataset = pd.read_csv("prova.csv")

X = dataset.iloc[:, :2]
y = dataset.iloc[:, 2].to_numpy()

from sklearn.preprocessing import MinMaxScaler

scale_x = MinMaxScaler().fit(X)
X = scale_x.transform(X)
y = y.reshape(len(y), 1)
scale_y = MinMaxScaler().fit(y)
y = scale_y.transform(y)

import joblib
joblib.dump(scale_x, "prova_x.save")
joblib.dump(scale_y, "prova_y.save")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

#print (X_train.shape, y_train.shape)
#print (X_test.shape, y_test.shape)

#create model
model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=500, batch_size=100, verbose=1)

evaluator = model.evaluate(X_train, y_train)
print('Train: {}'.format(evaluator))

evaluator = model.evaluate(X_test, y_test)
print('Test: {}'.format(evaluator))

model.save('prova.h5')
print (time.time()-start)
