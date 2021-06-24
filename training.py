import pandas as pd
import numpy as np
import time

from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import backend
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

start = time.time()
#Pandas DataFrame
dataset = pd.read_csv("heston_training.csv")
dataset = dataset.dropna()

X = dataset.iloc[:, :9]
y = dataset.iloc[:, 9].to_numpy()

from sklearn.preprocessing import MinMaxScaler

scale_x = MinMaxScaler().fit(X)
X = scale_x.transform(X)
y = y.reshape(len(y), 1)
scale_y = MinMaxScaler().fit(y)
y = scale_y.transform(y)

# ti conviene salvarti le trasformazioni cosi`
import joblib
joblib.dump(scale_x, "heston_x_scaler_matteo.save")
joblib.dump(scale_y, "heston_y_scaler_matteo.save")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#create model

model = Sequential()
model.add(Dense(60, input_dim=9, activation='relu'))
model.add(Dense(40, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])

history = model.fit(X_train, y_train, epochs=1000, batch_size=500, verbose=1)

evaluator = model.evaluate(X_train, y_train)
print('Train: {}'.format(evaluator))

evaluator = model.evaluate(X_test, y_test)
print('Test: {}'.format(evaluator))

model.save('heston_model_matteo.h5')
print (time.time()-start)
