import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .math_functions import *

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
    # plt.tight_layout()
    plt.autoscale()
    plt.savefig('Results/Figures/Scenario'+str(s)+'/'+name+'_Hist.png')
    plt.close()

def print_barChart(data,s,stringx,stringy,name,num_years,warmup):
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

def ci_graph(n_runs, start_year, num_years, warmup, sim_avg, sim_max, sim_min, sim_sd, e_year, e_lower, e_mean,name):
    plt.figure(figsize=(9, 6),dpi=300)
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
    plt.errorbar(e_year,e_mean, errED, elinewidth = 1,  capsize=10, color="gray",ls="-.")
    plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 1, capsize=10, color="black" )
    plt.xlabel('Year')
    plt.xticks(np.arange(2005, 2035, step=5),rotation=45)
    plt.ylabel(name+' per Year')
    plt.legend(['Observed', 'Simulated 95% Joint Prediction Interval'])
    plt.autoscale()
    plt.savefig('Results/Figures/'+name+'_CI.png')
    plt.close()

def ci_graph_point(n_runs, start_year, num_years, warmup, sim_avg, sim_max, sim_min, sim_sd, e_points,e_col,name):
    #add min and max values. 
    plt.figure(figsize=(9, 6),dpi=300)
    errSIM = sim_sd/ np.sqrt(n_runs)*1.96 #normal mean confidence intervals
    errSIM = [x - y for x, y in zip(sim_max, sim_min)] #non-parametric prediction intervals
    errSIM= [x / 2 for x in errSIM]
    sim_avg = [x + y for x, y in zip(errSIM, sim_min)]  #non-parametric prediction intervals
    new_year_list = []
    for year in range(start_year, start_year+num_years): 
        new_year_list.append(int(year))
    if  isinstance(e_points, pd.DataFrame):
        e_points["Year"] = e_points["Year"].astype('int64')
        plt.plot(e_points["Year"],e_points[e_col], color = "gray",ls="-.")
        plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 1, capsize=10, color ="black")
        plt.legend(['Observed', 'Simulated 95% Joint Prediction Interval'])        
    else:
        plt.errorbar(new_year_list[warmup:],sim_avg[warmup:],errSIM[warmup:], elinewidth = 1, capsize=10, color="black")
        plt.legend(['Simulated 95% Joint Prediction Interval'])
    plt.xlabel('Year')
    plt.xticks(np.arange(2005, 2035, step=5),rotation=45)
    plt.ylabel(name+' per Year')
    plt.autoscale()
    plt.savefig('Results/Figures/'+name+'_CI.png')
    plt.close()