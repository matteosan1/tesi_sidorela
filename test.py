import pandas as pd
import time
import joblib
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

start = time.time()

dataset = pd.read_csv("heston_training_prices2.csv")
dataset = dataset.dropna()
#dataset['h'] = dataset['h'].round(decimals=4)
#dataset[dataset < 0] = 0
print (len(dataset))

X = dataset.iloc[:, :33]
X[X<0] = 0
y = dataset.iloc[:, 33:].to_numpy()
print (len(X))

scale_x = MinMaxScaler().fit(X)
X = scale_x.transform(X)
scale_y = MinMaxScaler().fit(y)
y = scale_y.transform(y)

joblib.dump(scale_x, "heston_x.save")
joblib.dump(scale_y, "heston_y.save")

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = Sequential()
model.add(Dense(64, input_dim=33, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(5))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

history = model.fit(X, y, epochs=3000, batch_size=100,
                    verbose=1, validation_split=0.2)

#evaluator = model.evaluate(X_train, y_train)
#print('Train: {}'.format(evaluator))

#evaluator = model.evaluate(X_test, y_test)
#print('Test: {}'.format(evaluator))

model.save('prova_relu.h5')
print (time.time()-start)
import json
json.dump(history.history, open('history.json', 'w'))
