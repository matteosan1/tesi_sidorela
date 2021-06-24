import pandas as pd
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

start = time.time()

dataset = pd.read_csv("heston_training.csv")
dataset = dataset.dropna()

X = dataset.iloc[:, 6:]
y = dataset.iloc[:, :3].to_numpy()

scale_x = MinMaxScaler().fit(X)
X = scale_x.transform(X)
scale_y = MinMaxScaler().fit(y)
y = scale_y.transform(y)

# ti conviene salvarti le trasformazioni cosi`
import joblib
joblib.dump(scale_x, "heston_x_scaler_matteo2.save")
joblib.dump(scale_y, "heston_y_scaler_matteo2.save")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(20, input_dim=4, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
history = model.fit(X_train, y_train, epochs=1000, batch_size=500, verbose=2)

evaluator = model.evaluate(X_train, y_train)
print('Train: {}'.format(evaluator))

evaluator = model.evaluate(X_test, y_test)
print('Test: {}'.format(evaluator))

model.save('heston_model_matteo2.h5')
print (time.time()-start)
