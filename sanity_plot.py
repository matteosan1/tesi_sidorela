import pandas as pd
import numpy as np
import csv
from heston import fourier_call_price_2

from matplotlib import pyplot as plt

params = {"kappa":np.arange(2, 3.5, 0.01),
          "theta": np.arange(0.02, 0.07, 0.001),
          "sigma": np.arange(0.1, 0.55, 0.01),
          "rho": np.arange(-0.5, -0.4, 0.01),
          "v0": np.arange(0.05,0.07,0.005),
          "r": 0.1,
          "T": np.arange(1, 2, 0.1),
          "k": np.arange(0.8, 1.4, 0.05)}

means = {k:np.mean(v) for k,v in params.items()}
index = 1

plt.figure(figsize=(15, 10))
for k, v in params.items():
    if k == "r":
        continue
    X = dict(means)
    x_plot = []
    y_plot = []
    for x in params[k]:
        X[k] = x
        y_plot.append(fourier_call_price_2(**X))
        x_plot.append(list(X.values()))
        
        sub = plt.subplot(3, 3, index)
        plt.plot(params[k], y_plot, label=k)
        sub.legend()
    index += 1
    
plt.show()
