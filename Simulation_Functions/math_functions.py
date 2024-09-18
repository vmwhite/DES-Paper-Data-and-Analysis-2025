import scipy.stats as stats
from .math_functions import *
import numpy as np
from numpy import True_, double, log as ln
import math

############################################# Transform functions ##############################################
def Normal_MLE_sigma(sd,n):
    sigmaMLE = np.sqrt(((n-1)/n)*(sd**2))
    return sigmaMLE
def gamma_calc(x,s):
    a = (x**2)/(s**2)
    b = (s**2/x)
    return a,b
def ln_est(loc,m,q,x): #Taken from Simulation Modeling and Analysis, Law 2008 page 373 example 6.24
    #m is most likely value
    #x is the value of the q quartile. Then convert to normal(0, 1) quantile
    #loc is the location parameter (treat as the minimum value)
    invQ = stats.norm.ppf(q)
    c = ln((m-loc)/(x-loc))
    sigma_est = (math.sqrt(invQ**2 - (4*c)) - invQ)/2
    mu_est = ln(m-loc) + sigma_est**2
    return mu_est, sigma_est
def ln_est_sig_only(loc,m,mu): #Taken from Simulation Modeling and Analysis, Law 2008 page 373 example 6.24
    #m is most likely value
    #x is the value of the q quartile. Then convert to normal(0, 1) quantile
    #loc is the location parameter (treat as the minimum value)
    sigma_est = math.sqrt(mu - ln(m-loc)) 
    return sigma_est
############################################ Other funcitonss #####################################################
def dict_mean(dict, years):
    dict_list = [[d[year] for n, d in dict.items()] for year in range(years)]
    # print(dict_list)
    dict_max = [max(lit) for lit in dict_list]
    dict_min = [min(lit) for lit in dict_list]
    # print(dict_list)
    return np.mean(dict_list, axis =1), dict_max, dict_min

def dict_set_mean(dict, years):
    dict_list = [[len(d[year]) for n, d in dict.items()] for year in range(years)]
    # print(dict_set_mean)
    dict_max = [max(lit) for lit in dict_list]
    dict_min = [min(lit) for lit in dict_list]
    # print(dict_set_mean)
    return np.mean(dict_list, axis =1), dict_max, dict_min

def dict_sd(dict, years):
    dict_list = [[d[year] for n, d in dict.items()] for year in range(years)]
    # print(dict_sd)
    return np.std(dict_list, axis =1)

def dict_set_sd(dict, years):
    dict_list = [[len(d[year]) for n, d in dict.items()] for year in range(years)]
    # print(dict_set_sd)
    return np.std(dict_list, axis =1)

def final_CI(mean, sd, n):
    lower = mean - ((1.96*sd)/np.sqrt(n))
    upper = mean + ((1.96*sd)/np.sqrt(n))
    string = "(" + str(lower) + ", " + str(upper) + ")"
    return string
def timeCalculations(list):
    d = {}
    list = [x for x in list if math.isnan(x) == False]
    if len(list) == 0:
        d["Avg"] = 0
        d["SD"] = 0
        d["Max"] =0
        d["Min"] = 0
        d["n"] = 0
    else:
        d["Avg"] = np.mean(list)
        d["SD"] = np.std(list)
        d["Max"] = np.max(list)
        d["Min"] = np.min(list)
        d["n"] = len(list)
    return d