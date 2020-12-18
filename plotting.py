from keras.models import load_model
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


scale_x = joblib.load("heston_x_scaler_1.save")
scale_y = joblib.load("heston_y_scaler_1.save")

dataset = pd.read_csv("heston_training.csv")
dataset = dataset.dropna()

X_test = scale_x.transform(dataset.iloc[:, :9])
y_test = np.array(dataset.iloc[:, 9].to_numpy())

model = load_model("heston_model.h5")
prediction = model.predict(X_test)

inv_prediction = np.array(scale_y.inverse_transform(prediction))

#plt.plot(inv_prediction[:100000], color="red", label="NN price")
##plt.plot(y_test, label= "HESTON price")
#plt.legend()
#plt.show()
##plt.savefig("comparison_fair.png")

delta = []
for i in range(500):
    delta.append(abs(inv_prediction[i]-y_test[i])/y_test[i])


#plt.figure(figsize = (15,10))
plt.plot(delta)#y_test, predictions)
#plt.xlabel("Actual Price")
#plt.ylabel("Predicted Price")
##plt.plot([0,1], [0,1], 'r')
plt.grid(True)
plt.show()
