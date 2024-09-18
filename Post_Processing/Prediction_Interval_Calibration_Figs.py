
from Graphing_Functions import ci_graph,ci_graph_point, pi_graph_point, dict_mean, dict_sd, dict_set_mean, dict_set_sd, pi_graph
import sys
import os
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parallel folder
parallel_folder = os.path.join(current_dir, '..', 'Simulation_DC_OpioidPolicing')

# Add the parallel folder to the module search path
sys.path.append(parallel_folder)
import DataCleaningDCsim_NEW as inputs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 28})
# plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 25
# plt.rcParams["axes.labelsize"] = 20
# plt.rcParams['xtick.labelsize'] = 16
# plt.rcParams['ytick.labelsize'] = 16
#to change ebar and line thickness go to Graphing_Functions and change it directly in the function

##### parameters 
num_years = 13
warmup = 5
actual_warmup = 5
start_year = 2013 - actual_warmup
n_runs = n = 600
new_year_list = []
for year in range(start_year, start_year+num_years): 
    new_year_list.append(int(year))

Baseline_Folder =r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries'+str(n)

csv_files = ['\Base_Yearly_Arrivals.csv','\Base_Yearly_ODeaths.csv','\Base_Yearly_OArrests.csv', '\Base_Yearly_Indv_Treats.csv', '\Base_Yearly_Hosp.csv', '\Base_Yearly_Prevalence.csv']
comp_inputs = [ inputs.df_initiation, inputs.df_DCdeaths, inputs.df_Yarrests, inputs.df_treat, inputs.df_HE, inputs.df_prev]
x_lab = ["Arrival", "Deaths", "Arrests","Individuals Treated\n", "Hospital Encounters\n", "Prevalence"]
ecol_lab = ["Dane County Opioid Initiation estimate (number of people)", "Deaths", "ArrestCount", "Total Individuals", "Number of Discharges", "Prevalence"]
for idx, item in enumerate(comp_inputs):
    df = pd.read_csv(r''+Baseline_Folder + csv_files[idx])
    df= df[0:n]
    df= df.T[1:num_years+1]
    dict_mu, dict_max, dict_min = dict_mean(df, num_years)
    if item == inputs.df_prev:
        pi_graph(Baseline_Folder,dict_mu, dict_max, dict_min, dict_sd(df,num_years),inputs.df_prev["Year"], inputs.df_prev["Dane County use estimate (number of people) LOWER CI"].astype(float),inputs.df_prev["Dane County use estimate (number of people)"].astype(float),"Prevalence", start_year, num_years, warmup, n_runs)
    else:
        pi_graph_point(Baseline_Folder,dict_mu, dict_max, dict_min, dict_sd(df,num_years),item,ecol_lab[idx], x_lab[idx], start_year, num_years, warmup, n)

