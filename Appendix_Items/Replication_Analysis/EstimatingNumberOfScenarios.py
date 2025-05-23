'''
########## 
This is an implementation of the paper Hoad, Robinson, Davies 2009. 

We check each output using thir algorithm. 

'''

import re
import os
import math
import numpy as np
import ast
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sma
# import statsmodels.formula.api as sm
import statsmodels.api as sm
import statsmodels.stats.api as sms

from statsmodels.compat import lzip

# Shapiro-Wilk Test and other normality checks
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import kstest
from statsmodels.stats.diagnostic import lilliefors
from scipy import stats
from scipy.stats import skew

#Latex tables
from tabulate import tabulate
from texttable import Texttable

import latextable
###############################################################################################################
'''' Define inputs'''
#dont change things as its not lined up along everthing and could lead to erors. You ahve to make teh table yourself from the outputs.
d_thres = 0.01
#half-widths [active, arrests, hosp, inactive, odeaths, prev, relapse, treat]  *10 for 10 year cumulative outcomes
E = [100,5,5,1000,1,100,5,5,100*25,5*25,5*25,1000*25,1*25,100*25,5*25,5*25]

###############################################################################################################
''' Special Functions'''
#https://www.youtube.com/watch?v=g0EWQV-9B_0
#https://online.stat.psu.edu/stat500/lesson/5/5.4/5.4.5
def Calc_Percision(x_bar,s,n):
    #significance level
    q = 1-(0.05/2)
    #degrees of freedom
    df = n-1
    t = stats.t.ppf(q, df)
    d = ((100*t*s)/np.sqrt(n))/x_bar
    return d

''' used to obtain a certain size CI half-width'''
def Calc_N(s,E):
    alpha = 0.05
    # Determine the z-critical value
    z = stats.norm.ppf(1-(alpha/2))
    df = n-1
    t = stats.t.ppf(1-(alpha/2), df)
    #caclulate number of required samples given sigma and half-width
    N = ((t*s)/E)**2
    return N

def Calc_E(s,N):
    alpha=0.05
    # Determine the z-critical value
    z = stats.norm.ppf(1-(alpha/2))
    #caclulate half-width
    E = ((z*s)/np.sqrt(N))
    return E

def make_latex_table(rows, col_num):
    #https://github.com/bufordtaylor/python-texttable/blob/master/texttable.py
    table = Texttable()
    table.set_cols_align(["c"] * col_num)
    # table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)
    return table

###############################################################################################################


dirname = os.path.dirname(__file__)
try: 
    os.makedirs(r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\N')
except:
    pass
Baseline_Output_Folder = r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\N'

''' Yearly Standard Outputs'''
Base_output_file_list = [r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_Active_YearEnd.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_OArrests.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_Hosp.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_Inactive_YearEnd.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_ODeaths.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_Prevalence.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_Relapses.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_Yearly_Indv_Treats.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_Active_YearEnd.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_OArrests.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_Hosp.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_Inactive_YearEnd.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_ODeaths.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_Prevalence.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_Relapses.csv',
r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\\Cum_Indv_Treats.csv']

# Scen_output_file_list = [r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_Active_YearEnd.csv',
# r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_Arrests.csv',
# r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_Hosp.csv',
# r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_Inactive_YearEnd.csv',
# r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_ODeaths.csv',
# r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_Prevalence.csv',
# r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_Relapses.csv',
# r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_400_Years_25_Time_010623_153652\summaries600\Base_Yearly_Treats.csv']

output_str = ["Active", "Arrest", "Hosp", "Inactive", 
              "ODeaths", "Relapses", "Prev", 
              "Treats", 
              "Cumulative Active", "Cumulative Arrest", "Cumulative Hosp", "Cumulative Inactive", 
              "Cumulative ODeaths", "Cumulative Relapses", "Cumulative Prev", 
              "Cumulative Treats"]

#################### Make QQplots #########################
year = [np.nan, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]
year_list = ["Year_9","Year_11", "Year_25"]
# Prepare the figure
fig, axes = plt.subplots(5, 3, figsize=(20, 20))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
ax_flat = axes.flatten()

# Track the number of plots generated
plot_count = 0

for f_idx, f in enumerate(Base_output_file_list):
    df_Base = pd.read_csv(Base_output_file_list[f_idx])
    for idx, col in enumerate(df_Base):
        if col in year_list:
            array_Base = df_Base[col]
            X_bar = np.mean(array_Base[0:600])
            residuals = X_bar - array_Base[0:600]
            ''' Q-Q PLOT '''
            ax = ax_flat[plot_count]
            sma.qqplot(residuals, line='q', ax=ax)
            ax.set_title(f'Q-Q plot of year {year[idx]}, Number of {output_str[f_idx]}, n=600', fontsize=14)
            plot_count += 1
            
            if plot_count == len(ax_flat):
                break
    if plot_count == len(ax_flat):
        break
# Save the combined image
output_file = os.path.join(Baseline_Output_Folder, 'combined_qqplots.jpeg')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
####### Normality Tests ####################
E_calc = []
N_calc = []
s_hat = []
normal_SW = []
normal_KS = []
normal_KScorr = []
skewness = []
for f_idx, f in enumerate(Base_output_file_list):
    #read in csv files 
    # df_Sens = pd.read_csv(Scen_output_file_list[f_idx])
    df_Base = pd.read_csv(Base_output_file_list[f_idx])
    #create histogram of year 2017 and 2032 for each output 
    for idx, col in enumerate(df_Base):
        if col == "Year_25":
            array_Base = df_Base[col]
            # array_Sens = df_Sens[col]
            d_n = []
            X_bar_n = []
            n_n = []
            normal_n = []
            N_n = []
            s_n = []
            for i in range(0,len(array_Base)+1):
                if i == 600:
                    break
                ############################################################################
                ''' normality test '''
                X_bar = np.mean(array_Base[0:i])
                X_bar_n.append(X_bar)
                if i < 4:
                    pass
                else:
                    stat, p = shapiro( X_bar - array_Base[0:i])
                    stat_2, p_2 = kstest( X_bar - array_Base[0:i], stats.norm.cdf)
                    stat_3, p_3 = lilliefors(X_bar - array_Base[0:i],dist='norm')
                    skewed = skew( X_bar - array_Base[0:i])
                    print('Statistics=%.3f, p=%.3f' % (stat, p))
                    # interpret
                    alpha = 0.01
                    if p > alpha:
                        print('Sample looks Gaussian (fail to reject H0)  at n= %f H0 := that the data was drawn from a normal distribtuion' % (int(i)))
                        normal_n.append(1)
                    else:
                        print('Sample does not look Gaussian (reject H0) H0 := that the data was drawn from a normal distribtuion')
                        normal_n.append(0)
                #############################################################################
                ''' find percision'''
                s = np.std(array_Base[0:i])
                s_n.append(s)
                n = int(i)
                n_n.append(n)
                d_n.append(Calc_Percision(X_bar, s, n))
                N_n.append(Calc_N(s,E[f_idx]))
    #############################################################################
    ''''' final normality check '''
    normal_SW.append(p)
    normal_KS.append(p_2)
    normal_KScorr.append(p_3)
    skewness.append(skewed)
    '''' N'''
    E_calc.append(Calc_E(s,len(n_n)))
    N_calc.append(Calc_N(s,E[f_idx]))
    s_hat.append(s)
    y_1 = np.array(N_n)
    y_2 = np.array(s_n)
    plt.scatter(y_1[300:], y_2[300:])
    plt.title(output_str[f_idx])
    # plt.show()
    print("---------------------")
    print(output_str[f_idx])
    print(y_1)
    print(y_2)
    plt.xlabel('Est. Number of Replicaitons Needed to obtain 95% confidence level (N)', fontsize=18)
    plt.xticks(fontsize=16)
    # plt.ylabel("The Number of " + name, fontsize=18)
    plt.ylabel("Standard Deviation (s)", fontsize=18)
    plt.yticks(fontsize=16)
    plt.savefig(str(Baseline_Output_Folder) + '\\' + str(output_str[f_idx])+ '_N_vs_s.png')
    plt.close()
    ''' plot: percision vs n with nortin normality range'''
    x = np.array(n_n)
    y_1 = np.array(X_bar_n)
    y_2 = np.array(d_n)
    normal_n = np.array(normal_n)
    plt.plot(x, y_2)
    plt.axhline(y=0.05, color='r', linestyle='-')
    plt.title(output_str[f_idx])
    #highlights where we fail to reject the null hypotheses
    norm = np.where(normal_n == 1)[0]
    for i in range(len(norm)):
        plt.axvline(x=norm[i], color='g', alpha=0.1,linestyle='-')
    # plt.plot(x, y_2, color=("black", "red"))
    # plt.show()
    plt.autoscale() 
    plt.savefig(str(Baseline_Output_Folder) + '\\'  + str(output_str[f_idx])+ '_n_vs_d.png')
    plt.close()
    ''' mu vs n'''
    plt.plot(x, y_1)
    plt.title(output_str[f_idx])
    norm = np.where(normal_n == 1)[0]
    for i in range(len(norm)):
        plt.axvline(x=norm[i], color='g', alpha=0.1,linestyle='-')
    # plt.show()
    plt.savefig(str(Baseline_Output_Folder) + '\\'  + str(output_str[f_idx])+ '_n_vs_mu.png')
    plt.close()
print(output_str)
print("N for 600 runs using E and current s")
print(N_calc)
print("s")
print(s_hat)
print("E_calc, Half length of N=")
print(E_calc)
print("Shapiro Wilk test pvalue, H0 := that the data was drawn from a normal distribution")
print(normal_SW)
print("KS test pvalue, H0 := that the data was drawn from a normal distribution. second is the lilliefors correction")
print(normal_KS)
print(normal_KScorr)
print("Skewness")
print(skewness)

rows = []
rows.append(['Test', "Active", "Opioid-Related Arrests", "Hospital Encounters", "Opioid-Related Deaths",  "Treatments", "Cummulative Active", "Cumulative Opioid-Related Arrests", "Cumulative Hospital Encounters","Cumulative Opioid-Related Deaths",  "Cumulative Treatments"])
rows.append(['K-S Test with Lilliefors correction (p-value)', str(round(normal_KScorr[output_str.index("Active")],2)),
             str(round(normal_KScorr[output_str.index("Arrest")],2)),str(round(normal_KScorr[output_str.index("Hosp")],2)), 
             str(round(normal_KScorr[output_str.index("ODeaths")],2)), str(round(normal_KScorr[output_str.index("Treats")],2)),
             str(round(normal_KScorr[output_str.index("Cumulative Active")],2)),str(round(normal_KScorr[output_str.index("Cumulative Arrest")],2)),
             str(round(normal_KScorr[output_str.index("Cumulative Hosp")],2)),str(round(normal_KScorr[output_str.index("Cumulative ODeaths")],2)),str(round(normal_KScorr[output_str.index("Cumulative Treats")],2))])
rows.append(['Skewness', str(round(skewness[output_str.index("Active")],2)),str(round(skewness[output_str.index("Arrest")],2)),
             str(round(skewness[output_str.index("Hosp")],2)), str(round(skewness[output_str.index("ODeaths")],2)),
             str(round(skewness[output_str.index("Treats")],2)),str(round(skewness[output_str.index("Cumulative Active")],2)),
             str(round(skewness[output_str.index("Cumulative Arrest")],2)),str(round(skewness[output_str.index("Cumulative Hosp")],2)),
             str(round(skewness[output_str.index("Cumulative ODeaths")],2)),str(round(skewness[output_str.index("Cumulative Treats")],2))])

table = make_latex_table(rows, 11)
print(latextable.draw_latex(table, caption="Normality Tests for Simulation Residual Outputs for Year 2032  N=600", label= "tab:NormTest"))

