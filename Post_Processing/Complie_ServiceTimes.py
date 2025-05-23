
import os
import ast
import csv
import pandas as pd
import numpy as np
n= 600
cwd = os.getcwd() 

mainf=r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600'
mainfile=r'Revison2_Results\Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_021425_082048\summaries600\Base_ServiceTimes.csv'
pd.read_csv(mainfile)

Service_Times = []
Avg_Service_Times = []
#     ''' Global Calculations'''
# Convert all values to float
Service_Times[1:4] = [[float(val) for val in sublist] for sublist in Service_Times[1:4]]
Avg_Service_Times.append([sum(sub_list[1:]) / len(sub_list[1:]) for sub_list in zip(*Service_Times[1:4])][1:])
Avg_Service_Times[0].insert(0,0)

list_of_outcomes = [ 
Avg_Service_Times

]
outcome_str = [ 
"Avg_Service_Times"
]  

first_line = ["Run","First Active State","Treatment State","Jail State","Hospital State","InactiveAll State","InactiveOnly State","InactiveTreat State","InactiveCrime State","InactiveHosp State","Enter Age"]  
Avg_Service_Times.insert(0,first_line)
with open(mainf + "\ " + str(outcome_str[0])+".csv", "w", newline="") as f: 
    writer = csv.writer(f)
    writer.writerows(list_of_outcomes[0])
f.close()