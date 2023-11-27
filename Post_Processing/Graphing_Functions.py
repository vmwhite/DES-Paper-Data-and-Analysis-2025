'''
This file contains of the graphing functions to create graphs. 
joint confidence intervals: https://stats.stackexchange.com/questions/327560/how-to-find-joint-confidence-interval-for-a-bunch-of-normal-distributed-samples
\indv_CIlevel = (1- sqrt^years(1- joint CI level))
'''

##### libraries ######
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
from scipy.stats import t
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

def dict_diff(dict1,dict2, years):
    t_crit =scipy.stats.t.ppf(q=1-(.05/2),df=599)
    p_crit = scipy.stats.t.sf(t_crit,599-1)*2
    dict_list= [[] for n in dict1.items()]
    for n, d in dict1.items():
        dict_list[n]= abs(dict1[n] - dict2[n])
    dict_list = pd.DataFrame(data=dict_list)
    dict_list =dict_list.dropna(axis="columns")
    df_diff = dict_list.agg(['mean', 'std'], axis=0)
    df_diff.loc["diff_t_score"] = df_diff.apply((lambda row: row[0]/ (row[1]/ np.sqrt(600))), axis=0)
    df_diff.loc["diff_p_val"] = df_diff.apply((lambda row2: scipy.stats.t.sf(abs(row2[2]),n-1)*2 ))
    df_diff.loc["diff_p_val_text"] = df_diff.apply((lambda row3:  '$<0.001$' if  row3[3] < 0.001 and math.isnan(row3[3]) == False else str(round(row3[3],3))), axis=0) 
    df_diff.loc["dif_sig?"]= df_diff.apply((lambda row3: "yes" if  row3[3] < p_crit and math.isnan(row3[2]) == False else "No"))

    return df_diff
########################################## graphing funcitons ##########################################
def print_histogram(list,bins_num,s,stringx,stringy,name):
    plt.figure(figsize=(9, 6),dpi=300)
    freq, bins, patches = plt.hist(list,bins_num) 
    bin_centers = np.diff(bins)*0.5 + bins[:-1]
    n = 0
    for fr, x, patch in zip(freq, bin_centers, patches):
        height = int(freq[n])
        plt.annotate("{}".format(height),
                    xy = (x, height),             # top left corner/ of the histogram bar
                    xytext = (0,0.2),             # offsetting label position above its bar
                    textcoords = "offset points", # Offset (in points) from the *xy* value
                    ha = 'center', va = 'bottom'
                    )
        n = n+1
    plt.xlabel(stringx)
    plt.ylabel(stringy)
    plt.autoscale()
    plt.savefig('Results/Figures/Scenario'+str(s)+'/'+name+'_Hist.png')
    plt.close()

def print_barChart(data,s,stringx,stringy,name,num_years, warmup):
    ######## utilization by month ###
    plt.figure(figsize=(9, 6),dpi=300)
    plt.tight_layout()
    plt.xlabel(stringx+" (Month)")
    plt.ylabel(stringy)
    plt.tight_layout()
    # plt.set_title(name)
    plt.bar(range(warmup, len(data)), data[warmup:], align='edge', width=1.0, color='black')
    for i in range(warmup,len(data)):  
        label = "{:.0f}".format(data[i])
        plt.annotate(label, # this is the text
            (i,data[i]), # these are the coordinates to position the label
            textcoords="offset points", # how to position the text
            xytext=(0,10), # distance from text to points (x,y)
            ha='center') # horizontal alignment can be left, right or center
    plt.autoscale()
    plt.savefig('Results/Figures/Scenario'+str(s)+'/Month_'+name+'_Hist.png')
    plt.close()
    ####### utilization by year ###  NOTE TO SELF: BEINGING OF YEAR IS JUST FIRST MONTH OF THAT YEAR!!!!!!
    plt.tight_layout()
    plt.figure(figsize=(9, 6),dpi=300)
    plt.xlabel(stringx+ " (Year)")
    plt.ylabel(stringy)
    plt.tight_layout()
    j = 1
    yearly_data= [0]*(num_years+1)
    yearly_data[0]= 0
    for i in range(11, len(data),12):
        yearly_data[j]= data[i]
        label = "{:.0f}".format(yearly_data[j])
        plt.annotate(label, # this is the text
            (j,yearly_data[j]), # these are the coordinates to position the label
            textcoords="offset points", # how to position the text
            xytext=(0,10), # distance from text to points (x,y)
            ha='center') # horizontal alignment can be left, right or center
        j +=1
    plt.bar(range(warmup,len(yearly_data)), yearly_data[warmup:], align='edge', width=1.0, color='black')
    plt.autoscale()
    plt.savefig('Results/Figures/Scenario'+str(s)+'/Year_'+name+'_Hist.png')
    plt.close()

def pi_graph(sim_avg, sim_max, sim_min, sim_sd, e_year, e_lower, e_mean,name,start_year, num_years, warmup, n_runs):
    plt.figure(figsize=(11, 8),dpi=600)
    errSIM = sim_sd/ np.sqrt(n_runs)*1.96 #normal mean confidence intervals
    errSIM = [x - y for x, y in zip(sim_max, sim_min)] #non-parametric prediction intervals
    errSIM= [x / 2 for x in errSIM]
    sim_avg = [x + y for x, y in zip(errSIM, sim_min)]  #non-parametric prediction intervals
    errED = e_mean - e_lower
    # print(errED)
    new_year_list = []
    e_year = e_year.astype('int64')
    for year in range(start_year, start_year+num_years): 
        new_year_list.append(int(year))
    plt.errorbar(e_year,e_mean, errED, elinewidth = 3,  capsize=14, color="gray",ls="-.", linewidth= 5)
    plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 4, capsize=14, capthick=4, color="black" , linewidth = 4)
    plt.xlabel('Year')
    plt.autoscale(enable=True, axis='both', tight=None) 
    #plt.xticks(np.arange(2005, 2035, step=5),rotation=45)
    plt.ylabel(name+' per Year')
    plt.legend(['Observed', 'Simulated 95% Joint Prediction Interval'], bbox_to_anchor=(.5, 1.2), loc='center',fancybox=True, shadow=True)
    
    #plt.autoscale()
    plt.tight_layout(h_pad=0)
    name = name.replace("\n","")
    plt.savefig('Results/Figures/'+name+'_PI.png',bbox_inches='tight')
    plt.close()

def pi_graph_point(sim_avg, sim_max, sim_min, sim_sd, e_points,e_col,name, start_year, num_years, warmup, n_runs):
    #add min and max values.
     
    plt.figure(figsize=(11, 8),dpi=300)
    errSIM = sim_sd/ np.sqrt(n_runs)*1.96 #normal mean confidence intervals
    errSIM = [x - y for x, y in zip(sim_max, sim_min)] #non-parametric prediction intervals
    errSIM= [x / 2 for x in errSIM]
    sim_avg = [x + y for x, y in zip(errSIM, sim_min)]  #non-parametric prediction intervals
    new_year_list = []
    for year in range(start_year, start_year+num_years): 
        new_year_list.append(int(year))
    if  isinstance(e_points, pd.DataFrame):
        e_points["Year"] = e_points["Year"].astype('int64')
        plt.plot(e_points["Year"].values,e_points[e_col].values, color = "gray",ls="-.", linewidth = 5)
        plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 4, capsize=14, capthick=4, color ="black", linewidth= 4)
        plt.legend( ['Observed','Simulated 95% Joint Prediction Interval'],bbox_to_anchor=(.5, 1.25),loc='center',fancybox=True, shadow=True)     
    else:
        plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 4, capsize=14,capthick=4,  color="black", linewidth = 4)
        plt.legend(['Simulated 95% Joint Prediction Interval'],bbox_to_anchor=(.5, 1.25),loc='center',fancybox=True, shadow=True)
    plt.xlabel('Year')
    plt.xlim([2012, 2021])
    plt.autoscale(enable=True, axis='y', tight=None) 
    plt.xticks(np.arange(2012, 2023, step=2),rotation=45)
    plt.ylabel(name+' per Year')
    #plt.autoscale()
    plt.tight_layout(h_pad=0)
    name = name.replace("\n","")
    plt.savefig('Results/Figures/'+name+'_PI.png',bbox_inches='tight',)
    plt.close()
    
def ci_graph(sim_avg, sim_sd, e_year, e_lower, e_mean,name,start_year, num_years, warmup, n_runs,save_as,z_score):
    plt.figure(figsize=(9, 6),dpi=300)
    errSIM = sim_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
    errED = e_mean - e_lower
    # print(errED)
    new_year_list = []
    e_year = e_year.astype('int64')
    for year in range(start_year, start_year+num_years): 
        new_year_list.append(int(year))
    plt.errorbar(e_year,e_mean, errED, elinewidth = 3,  capsize=14, color="gray",ls="-.")
    plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 3, capsize=14, color="black" )
    plt.xlabel('Year')
    plt.xticks(np.arange(2005, 2035, step=5),rotation=45)
    plt.ylabel(name+' per Year')
    plt.legend(['Example Comparison', 'Baseline'])  
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()

def ci_graph_point(sim_avg, sim_sd, comp_avg, comp_sd,name, start_year, num_years, warmup, n_runs, save_as,z_score):
    #add min and max values. 
    plt.figure(figsize=(9, 6),dpi=300)
    errSIM = sim_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
    new_year_list = []
    for year in range(start_year, start_year+num_years): 
        new_year_list.append(int(year))
    # if  isinstance(e_points, pd.DataFrame):
    errCOMP = comp_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
    # e_points["Year"] = e_points["Year"].astype('int64')
    #plot comparison
    plt.errorbar(new_year_list[warmup:],comp_avg[warmup:],errCOMP[warmup:], elinewidth = 1, capsize=10, color = "gray",ls="-.")
    #plot baseline values
    plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 1, capsize=10, color ="black")
    plt.legend(['Example Comparison', 'Baseline'])        
    # else:
    #     plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 1, capsize=10, color="black")
    #     plt.legend(['Baseline'])
    plt.xlabel('Year')
    plt.xticks(np.arange(2005, 2035, step=5),rotation=45)
    plt.ylabel("The Number of " + name)
    plt.title("Simualted 95% Joint Confidence Interval of \n the Number of " + name )
    plt.autoscale()
    plt.savefig(save_as)
    plt.close()
