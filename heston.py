import numpy as np
from scipy.integrate import quad
from functools import partial

def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma*rho*u*1j
    d = np.sqrt( xi**2 + sigma**2 * (u**2 + 1j*u) )
    g1 = (xi+d)/(xi-d)
    g2 = 1/g1
    cf = np.exp( 1j*u*mu*t + (kappa*theta)/(sigma**2) * ( (xi-d)*t - 2*np.log( (1-g2*np.exp(-d*t))/(1-g2) ))\
              + (v0/sigma**2)*(xi-d) * (1-np.exp(-d*t))/(1-g2*np.exp(-d*t)) )
    return cf

def Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """
    integrand = lambda u: np.real( (np.exp(-u*k*1j) / (u*1j)) * 
                                  cf(u-1j) / cf(-1.0000000000001j) )  
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=1000)[0]

def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """
    integrand = lambda u: np.real( np.exp(-u*k*1j) /(u*1j) * cf(u) )
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=1000)[0]

def fourier_call_price(kappa, theta, sigma, rho, v0, r, T, s0, K):
    k_log = np.log(s0/K)
    cf_H_b_good = partial(cf_Heston_good, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho)  
    limit_max = 1000               
    return s0*Q1(k_log, cf_H_b_good, limit_max) - K * np.exp(-r*T) * Q2(k_log, cf_H_b_good, limit_max)

def fourier_call_price_2(kappa, theta, sigma, rho, v0 ,r ,T, k):
    k_log = np.log(k)
    cf_H_b_good = partial(cf_Heston_good, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho)  
    limit_max = 1000               
    return Q1(k_log, cf_H_b_good, limit_max) - k * np.exp(-r*T) * Q2(k_log, cf_H_b_good, limit_max)

def fourier_put_price(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K):              
    cf_H_b_good = partial(cf_Heston_good, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho)  
    limit_max = 1000              
    return K * np.exp(-r*T) * (1 - Q2(k, cf_H_b_good, limit_max)) - s0 * Q1(k, cf_H_b_good, limit_max)
