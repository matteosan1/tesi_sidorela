import math
import numpy
import pandas as pd
import numpy as np
from math import log, exp, sqrt
from math import *
from scipy.stats import norm
from scipy import *
from scipy.integrate import quad 
from dateutil.relativedelta import relativedelta


def d1(S, K, r, vol, ttm):
    num = log(S/K) + (r + 0.5*pow(vol, 2)) * ttm
    den = vol * sqrt(ttm)
    if den == 0:
        return 100000000.
    return num/den

def d2(S, K, r, vol, ttm):
    return d1(S, K, r, vol, ttm) - vol * sqrt(ttm)

def call(S, K, r, vol, ttm):
    return S * norm.cdf(d1(S, K, r, vol, ttm)) - K * exp(-r * ttm) * norm.cdf(d2(S, K, r, vol, ttm))

def put(S, K, r, vol, ttm):
    return K * exp(-r * ttm) * norm.cdf(-d2(S, K, r, vol, ttm)) - S * norm.cdf(-d1(S, K, r, vol, ttm))


# Heston model solution 1
def cf_Heston(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed in the original paper of Heston (1993)
    """
    xi = kappa - sigma*rho*u*1j
    d = np.sqrt( xi**2 + sigma**2 * (u**2 + 1j*u) )
    g1 = (xi+d)/(xi-d)
    cf = np.exp( 1j*u*mu*t + (kappa*theta)/(sigma**2) * ( (xi+d)*t - 2*np.log( (1-g1*np.exp(d*t))/(1-g1) ))\
              + (v0/sigma**2)*(xi+d) * (1-np.exp(d*t))/(1-g1*np.exp(d*t)) )
    return cf

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
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=2000 )[0]

def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """
    integrand = lambda u: np.real( np.exp(-u*k*1j) /(u*1j) * cf(u) )
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=2000 )[0]

def fourier_call_price(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K):               
    cf_H_b_good = partial(cf_Heston_good, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho )  
    limit_max = 2000               
    return s0 * Q1(k, cf_H_b_good, limit_max) - K * np.exp(-r*T) * Q2(k, cf_H_b_good, limit_max)

def fourier_put_price(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K):              
    cf_H_b_good = partial(cf_Heston_good, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa, rho=rho )  
    limit_max = 2000              
    return K * np.exp(-r*T) * (1 - Q2(k, cf_H_b_good, limit_max)) - s0 * Q1(k, cf_H_b_good, limit_max)

# Heston model solution 2
def heston_call_price(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K): 
    p1 = p(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K, 1) 
    p2 = p(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K, 2) 
    return (s0*p1-K*exp(-r*T)*p2)

def heston_put_price(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K): 
    p1 = p(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K, 1)
    p2 = p(kappa, theta, sigma, rho, v0 ,r ,T ,s0 ,K, 2)
    return (K*exp(-r*T)*(1-p2) - s0*p1 )

def p(kappa, theta, sigma, rho, v0 ,r ,T ,s0 , K, status): 
    integrand = lambda phi: (exp(-1j*phi*log(K))*f(phi, kappa, theta, sigma, rho, v0 , r, T, s0, status)/(1j*phi)).real 
    return (0.5+(1/pi)*quad(integrand,0,100)[0]) 

def f(phi, kappa, theta, sigma, rho, v0, r, T, s0, status): 
    if status==1: 
        u = 0.5 
        b = kappa-rho*sigma 
    else: 
        u = -0.5 
        b = kappa 
    a = kappa*theta 
    x = log(s0) 
    d = sqrt((rho*sigma*phi*1j-b)**2-sigma**2*(2*u*phi*1j-phi**2))  
    g = (b-rho*sigma*phi*1j+d)/(b-rho*sigma*phi*1j-d) 
    C = r*phi*1j*T+(a/sigma**2)*((b-rho*sigma*phi*1j+d)*T-2*log((1-g*exp(d*T))/( 1-g))) 
    D = (b-rho*sigma*phi*1j+d)/sigma**2*((1-exp(d*T))/(1-g*exp(d*T)))  
    return exp(C+D*v0+1j*phi*x)

# Discount curve class
class DiscountCurve:

    def __init__(self, today, pillar_dates, discount_factors):
        
        # we just store the arguments as attributes of the instance
        self.today = today
        self.pillar_dates = pillar_dates
        self.discount_factors = discount_factors
        
        self.log_discount_factors = [
            math.log(discount_factor)
            for discount_factor in self.discount_factors
        ]
        
        self.pillar_days = [
            (pillar_date - self.today).days
            for pillar_date in self.pillar_dates
        ]        
        
    def df(self, d):
        d_days = (d - self.today).days
        interpolated_log_discount_factor = numpy.interp(d_days, self.pillar_days, self.log_discount_factors)
        
        return math.exp(interpolated_log_discount_factor) 
    
    def forward_libor(self, d1, d2):
        return (
            self.df(d1) /
            self.df(d2) - 1
        ) / ((d2  - d1).days / 365)
    

class OvernightIndexSwap:
    def __init__(self, notional, payment_dates, fixed_rate):
        
        self.notional = notional
        self.payment_dates = payment_dates
        self.fixed_rate = fixed_rate
        
    def npv_floating_leg(self, discount_curve):
        
        return self.notional * (
             discount_curve.df(self.payment_dates[0]) - 
             discount_curve.df(self.payment_dates[-1])      
        )
    
    def npv_fixed_leg(self, discount_curve):
        
        npv = 0
        for i in range(1, len(self.payment_dates)):   
            
            start_date = self.payment_dates[i-1]
            end_date = self.payment_dates[i]
            
            tau = (end_date - start_date).days / 360
            df = discount_curve.df(end_date)
            
            npv = npv + df * tau
            
        return self.notional * self.fixed_rate * npv
    
    def npv(self, discount_curve):
        
        float_npv = self.npv_floating_leg(discount_curve)
        fixed_npv = self.npv_fixed_leg(discount_curve)
        
        return float_npv - fixed_npv
    
    
def generate_swap_dates(start_date, n_months, tenor_months=12):
    
    dates = []
    
    for n in range(0, n_months, tenor_months):
        dates.append(start_date + relativedelta(months=n))
    dates.append(start_date + relativedelta(months=n_months))
    
    return dates    

class ForwardRateCurve:
    
    def __init__(self, pillar_dates, pillar_rates):
        self.today = pillar_dates[0]
        self.pillar_dates = pillar_dates
        self.pillar_rates = pillar_rates
        
        self.pillar_days = [
            (pillar_date - self.today).days
            for pillar_date in self.pillar_dates
        ]

    def forward_rate(self, d):
       
        d_days = (d - self.today).days
        
        return numpy.interp(d_days, self.pillar_days, self.pillar_rates)


class InterestRateSwap:
    
    def __init__(self, start_date, notional, fixed_rate, tenor_months, maturity_years):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.fixed_leg_dates = generate_swap_dates(start_date, 12 * maturity_years)
        self.floating_leg_dates = generate_swap_dates(start_date, 12 * maturity_years,
                                                      tenor_months)
        
    def annuity(self, discount_curve):
        a = 0
        for i in range(1, len(self.fixed_leg_dates)):
            a += discount_curve.df(self.fixed_leg_dates[i])
        return a

    def swap_rate(self, discount_curve, libor_curve):
        s = 0
        for j in range(1, len(self.floating_leg_dates)):
            F = libor_curve.forward_rate(self.floating_leg_dates[j-1])
            tau = (self.floating_leg_dates[j] - self.floating_leg_dates[j-1]).days / 360
            P = discount_curve.df(self.floating_leg_dates[j])
            s += F * tau * P
        return s / self.annuity(discount_curve)
        
    def npv(self, discount_curve, libor_curve):
        S = self.swap_rate(discount_curve, libor_curve)
        A = self.annuity(discount_curve)
        return self.notional * (S - self.fixed_rate) * A

class InterestRateSwaption:
    
    def __init__(self, exercise_date, irs):
        self.exercise_date = exercise_date
        self.irs = irs
        
    def npv_bs(self, discount_curve, libor_curve, sigma):
        
        A = self.irs.annuity(discount_curve)
        S = self.irs.swap_rate(discount_curve, libor_curve)

        T = (self.exercise_date - discount_curve.today).days / 365

        d1 = (math.log(S/self.irs.fixed_rate) + 0.5 * sigma**2 * T) / (sigma * T**0.5)
        d2 = d1 - (sigma * T**0.5)

        npv = self.irs.notional * A * (S * scipy.stats.norm.cdf(d1) - 
                                       self.irs.fixed_rate * scipy.stats.norm.cdf(d2))
        
        return npv
    
    def npv_mc(self, discount_curve, libor_curve, sigma, n_scenarios=10000):
        
        A = self.irs.annuity(discount_curve)
        S = self.irs.swap_rate(discount_curve, libor_curve)

        T = (self.exercise_date - discount_curve.today).days / 365
        discounted_payoffs = []

        for i_scenario in range(n_scenarios):
            S_simulated = S * math.exp(-0.5 * sigma * sigma * T +
                                       sigma * math.sqrt(T) * numpy.random.normal())

            swap_npv = self.irs.notional * (S_simulated - self.irs.fixed_rate) * A
            discounted_payoffs.append(max(0, swap_npv))

        npv_mc = numpy.mean(discounted_payoffs)
        npv_error = 3 * numpy.std(discounted_payoffs) / math.sqrt(n_scenarios)
        
        return npv_mc, npv_error
