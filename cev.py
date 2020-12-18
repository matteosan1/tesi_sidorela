import numpy as np
import pandas as pd
from scipy.stats import ncx2

def CEV(St, K, T, r, sigma, alpha):
    kappa = (2*r)/(sigma**2*(1-alpha)*(np.exp(2*r*(1-alpha)*T)-1))
    x = kappa*St**(2*(1-alpha))*np.exp(2*r*(1-alpha)*T)
    y = kappa*K**(2*(1-alpha))
    z = 2 + 1/(1-alpha)
                   
    V = St*(1-ncx2.cdf(y,z, x)) - K*np.exp(-r*T)*ncx2.cdf(x, z-2, y)
    return V

St = 100
K = [80, 90, 100, 110, 120]
T = 1
r = 0.05

sigma = np.arange(0.3, 0.76, 0.002)
alpha = np.arange(0.5, 0.95, 0.005)

d = {'alpha': [],
     'sigma': [],
     'price': []}

for index in K:
    for a in alpha:
        for v in sigma:            
            d["alpha"].append(a)
            d['sigma'].append(v)
            d['price'].append(CEV(St, index, T, r, v, a))
            
df = pd.DataFrame()
for k, v in d.items():
    df[k] = v
    
print (df.head())
df.to_csv('prova.csv')

    
