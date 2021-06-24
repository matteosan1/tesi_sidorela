import csv
import numpy as np

from heston import fourier_call_price

r = 0.1                                                 
rho = -0.45 #np.arange(-0.5, -0.4, 0.02):
kappa = 2.75 #np.arange(2,3.5,0.25):

with open("heston_training.csv", mode='w') as f:
    writer = csv.writer(f, delimiter=",", quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    for theta in np.arange(0.02, 0.07, 0.01):
        for sigma in np.arange(0.2, 0.55, 0.1):                    
            for v0 in np.arange(0.05,0.07,0.005):                         
                for T in (0.5, 1, 1.5, 2):
                    for K in (80, 85, 90, 95, 100, 105, 110, 115, 120):
                        for St in np.arange(95, 115, 1):
                    #for moneyness in np.arange(0.8, 1.4, 0.05):
                            price = fourier_call_price(kappa, theta, sigma, rho, v0, r, T, St, K)
                            if price > 0:
                                writer.writerow([theta, sigma, v0, T, St, K, price])

                            
