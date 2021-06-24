import pandas as pd
import numpy as np

import keras.backend as kb
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#def myloss(y_actual, y_predicted):
#    return 0.3*kb.mean(
#    print (y_actual)
#    print (y_predicted)
#    return 0.1
    

df = pd.read_csv("heston_training.csv")
df = df.dropna()

X = df.iloc[:, 3:]
y = df.iloc[:, :3].to_numpy()

scale_x = MinMaxScaler().fit(X)
X = scale_x.transform(X)
scale_y = MinMaxScaler().fit(y)
y = scale_y.transform(y)

# ti conviene salvarti le trasformazioni cosi`
import joblib
joblib.dump(scale_x, "heston_x_scaler.save")
joblib.dump(scale_y, "heston_y_scaler.save")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3))

model.compile(loss='mae', optimizer='adam', metrics=['mse'])
history = model.fit(X_train, y_train, epochs=5000, batch_size=1000, verbose=1)

evaluator = model.evaluate(X_train, y_train)
print('Train: {}'.format(evaluator))

evaluator = model.evaluate(X_test, y_test)
print('Test: {}'.format(evaluator))

#model.save('heston_model.h5')






