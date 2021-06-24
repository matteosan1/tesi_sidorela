from heston import fourier_call_price
from keras.models import load_model
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

kappa = 3
theta = 0.04
sigma = 0.5
rho = -0.5
v0 = 0.07
T = 1
r = 0.01
S0 = 100
K = 100

print (fourier_call_price(kappa, theta, sigma, rho, v0, r, T, S0, K))
model = load_model("heston_model_matteo.h5")

X = np.array([[kappa, theta, sigma, rho, v0, r, T, S0, K]])
print (X)
scale_x = joblib.load("heston_x_scaler_matteo.save")
scale_y = joblib.load("heston_y_scaler_matteo.save")
X = scale_x.transform(X)
print (X)
print(scale_y.inverse_transform(model.predict(X)))