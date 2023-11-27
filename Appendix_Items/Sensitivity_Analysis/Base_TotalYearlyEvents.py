import os
import sys
import glob
import re
import csv

mainfolder = r'Results_Baseline_Scen_400_Years_25_Time_100422_210149'

with open(mainfolder+"\summaries\Base_MainOutputs.csv", "w", newline="") as f: #for baseline analyis
    csv_columns = ['Run','Output'] + list(range(0,25))
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for file in glob.glob(mainfolder + "/*Scenario.txt"):
        fin = open(file, 'r')

        start_parsing = False
        parsed_variables = {}

        for line in fin.readlines():
            if "----------- Time Summary DF ---------------" in line:
                break

            if start_parsing:
                r_match = re.search(r'[a-zA-Z- ]+:,? +', line)
                variable_name = r_match.group().split(":")[0]
                # print(variable_name)
                dict1= {'Output' : variable_name}
                # Lists 
                list_match = re.search(r'\[[\d: ,]+\]', line)

                # Dictionaries 
                dict_match = re.search(r'[\{][\d: ,]+[\}]', line)

                # Run
                run = re.findall(r'\d+', str(file))
                dict0 = {"Run": run[5]}  
                
                if list_match is not None: # if list
                    parsed_variable = list_match.group().strip('][').split(', ')
                    parsed_variable = [int(a) for a in parsed_variable]
                    dict2 = {id: item for id,item in enumerate(parsed_variable)}
                elif dict_match is not None: # if dict
                    parsed_variable = {}
                    dict_list = dict_match.group().strip('}{').split(', ')
                    for entry in dict_list:
                        parsed_variable[int(entry.split(":")[0])] = int(entry.split(":")[1])
                    parsed_variables[variable_name] = parsed_variable
                    dict2 = {id:item for id,item in parsed_variables[variable_name].items()}
                row = {**dict0, **dict1, **dict2}
                writer.writerow(row)

            if "---------------------- Total Yearly Events ----------------------------" in line:
                start_parsing = True
f.close()