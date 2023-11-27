'''' This file takes in the raw text files from a results folder and converts them to summary CSV files'''
import re
import os
import ast
import csv
import pandas as pd
import numpy as np

cwd = os.getcwd() 


flist = [r'Results_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023844',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_20_Scen_1000_Years_25_Time_033023_092508',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_40_Scen_1000_Years_25_Time_033023_024503',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_60_Scen_1000_Years_25_Time_033023_042120',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_80_Scen_1000_Years_25_Time_033023_031124',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_100_Scen_1000_Years_25_Time_033023_021852',
r'Results_ED2RVal_22270_MARIVal_20_CMVal_0_Scen_1000_Years_25_Time_033023_022830',
r'Results_ED2RVal_22270_MARIVal_40_CMVal_0_Scen_1000_Years_25_Time_033023_040647',
r'Results_ED2RVal_22270_MARIVal_60_CMVal_0_Scen_1000_Years_25_Time_033023_014303',
r'Results_ED2RVal_22270_MARIVal_80_CMVal_0_Scen_1000_Years_25_Time_033023_023019',
r'Results_ED2RVal_22270_MARIVal_100_CMVal_0_Scen_1000_Years_25_Time_033023_025006',
r'Results_ED2RVal_30000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_131333',
r'Results_ED2RVal_40000_MARIVal_20_CMVal_20_Scen_1000_Years_25_Time_033023_015145',
r'Results_ED2RVal_45000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023133',
r'Results_ED2RVal_60000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_043938',
r'Results_ED2RVal_60000_MARIVal_40_CMVal_40_Scen_1000_Years_25_Time_033023_021800',
r'Results_ED2RVal_75000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_050610',
r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_1000_Years_25_Time_033023_030711',
r'Results_ED2RVal_90000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_031549']

for f_idx,f in enumerate(flist):
    mainfolder = f #ED2Recovery Scenario
    #
    try: 
        os.makedirs(mainfolder + "\summaries600")
        os.remove(mainfolder + "\Test_SimulationsMovement.txt")
    except:
        pass

    ''' The following is for calculating Service Time Values'''
    Service_Times = []
    file_list = os.listdir(mainfolder)
    for idx, file in enumerate(file_list):
        scen = 0
        if file.endswith('Scenario.txt'):
            fin = open(mainfolder+"/"+file, 'r')
            lines = fin.readlines()
            run = re.findall(r'\d+', str(file_list[idx]))
            if int(run[1]) == 0:
                skipsA = -1
                for line in lines: #combines Service times, other outcomes not currently combined
                    skipsA += 1
                    if "Avg_Time" in line:
                        break
                data = pd.read_csv(mainfolder+"/"+file, header = 0, sep="  +",skiprows= skipsA)
                Service_Times.append(data["State"].to_list())
                Service_Times[0].insert(0,"Run") 
            elif int(run[1]) == 101:
                skipsB = -1
                for line in lines: #combines Service times, other outcomes not currently combined
                    skipsB += 1
                    if "Avg_Time" in line:
                        break
            if (int(run[1]) % 50) == 0:
                skips = skipsA
            else:
                skips = skipsB
            data = pd.read_csv(mainfolder+"/"+file, header = 0, sep="  +",skiprows= skips)
            Service_Times.append(data["Avg_Time"].to_list())
            Service_Times[scen+1].insert(0,run[1])
            scen += 1     
            fin.close()

    # print("-------- Service Times ---------")
    # print(Service_Times) 
    with open(mainfolder+"\Summaries600\Base_ServiceTimes.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Service_Times)
    f.close()

    ''' The following is for calculating Main Output Values'''
    Main_Outputs = []
    file_list = os.listdir(mainfolder)
    n= 1000
    outputs_mainlist = [[] for i in range(n)]
    startdic = 41
    dic_num = 0
    readin = False
    for idx, file in enumerate(file_list):
        if file.endswith('Scenario.txt'):
            with open(mainfolder+"/"+file, 'r') as file:
                for line in file:
                    if "----------- Time Summary DF ---------------" in line:
                        readin = False
                    elif "---------------------- Total Yearly Events ----------------------------" in line:
                        readin = False
                    elif "Monthly Number of arrivals" in line:
                        readin = False
                    if readin == True:
                        dumb, run = re.findall(r'\d+', str(file_list[idx]))
                        try:
                            startdic = line.index('{')
                            line_dict = ast.literal_eval(line[startdic:])
                            outputs_mainlist[int(run)].append(list(line_dict.values()))
                        except ValueError:
                            startdic = line.index('[')
                            line_dict = ast.literal_eval(line[startdic:])
                            outputs_mainlist[int(run)].append(line_dict)
                    if "---------------------- Total Yearly Events ----------------------------" in line:
                        readin = True
                    elif "arrivals in final month:" in line:
                        readin = True
                    
    years = len(outputs_mainlist[0][0])
    active_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    active_ut[0].insert(0,"Run")
    inactive_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    inactive_ut[0].insert(0,"Run")
    Odeath_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    Odeath_ut[0].insert(0,"Run")
    Non_Odeath_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    Non_Odeath_ut[0].insert(0,"Run")
    Arrest_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    Arrest_ut[0].insert(0,"Run")
    Hosp_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    Hosp_ut[0].insert(0,"Run")
    Treat_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    Treat_ut[0].insert(0,"Run")
    arrivals = [["Year_" + str(i) for i in range(1,years+1)]]
    arrivals[0].insert(0,"Run")
    ODeaths = [["Year_" + str(i) for i in range(1,years+1)]]
    ODeaths[0].insert(0,"Run")
    Non_ODeaths = [["Year_" + str(i) for i in range(1,years+1)]]
    Non_ODeaths[0].insert(0,"Run")
    Arrests = [["Year_" + str(i) for i in range(1,years+1)]]
    Arrests[0].insert(0,"Run")
    Indv_Arrests = [["Year_" + str(i) for i in range(1,years+1)]]
    Indv_Arrests[0].insert(0,"Run")
    Treats = [["Year_" + str(i) for i in range(1,years+1)]]
    Treats[0].insert(0,"Run")
    Indv_Treats = [["Year_" + str(i) for i in range(1,years+1)]]
    Indv_Treats[0].insert(0,"Run")
    Hosp = [["Year_" + str(i) for i in range(1,years+1)]]
    Hosp[0].insert(0,"Run")
    Indv_Hosp = [["Year_" + str(i) for i in range(1,years+1)]]
    Indv_Hosp[0].insert(0,"Run")
    Prevalence = [["Year_" + str(i) for i in range(1,years+1)]]
    Prevalence[0].insert(0,"Run")
    Active_YearEnd = [["Year_" + str(i) for i in range(1,years+1)]]
    Active_YearEnd[0].insert(0,"Run")
    Inactive_YearEnd =  [["Year_" + str(i) for i in range(1,years+1)]]
    Inactive_YearEnd[0].insert(0,"Run")
    Relapses = [["Year_" + str(i) for i in range(1,years+1)]]
    Relapses[0].insert(0,"Run")

    for idx,run_list in enumerate(outputs_mainlist):
        active_ut.append(run_list[0])
        active_ut[idx+1].insert(0,idx)
        inactive_ut.append(run_list[1])
        inactive_ut[idx+1].insert(0,idx)
        Odeath_ut.append(run_list[2])
        Odeath_ut[idx+1].insert(0,idx)
        Non_Odeath_ut.append(run_list[3])
        Non_Odeath_ut[idx+1].insert(0,idx)
        Arrest_ut.append(run_list[4])
        Arrest_ut[idx+1].insert(0,idx)
        Hosp_ut.append(run_list[5])
        Hosp_ut[idx+1].insert(0,idx)
        Treat_ut.append(run_list[6])
        Treat_ut[idx+1].insert(0,idx)
        arrivals.append(run_list[7])
        arrivals[idx+1].insert(0,idx)
        ODeaths.append(run_list[8])
        ODeaths[idx+1].insert(0,idx)
        Non_ODeaths.append(run_list[9])
        Non_ODeaths[idx+1].insert(0,idx)
        Arrests.append(run_list[10])
        Arrests[idx+1].insert(0,idx)
        Indv_Arrests.append(run_list[11])
        Indv_Arrests[idx+1].insert(0,idx)
        Treats.append(run_list[12])
        Treats[idx+1].insert(0,idx)
        Indv_Treats.append(run_list[13])
        Indv_Treats[idx+1].insert(0,idx)
        Hosp.append(run_list[14])
        Hosp[idx+1].insert(0,idx)
        Indv_Hosp.append(run_list[15])
        Indv_Hosp[idx+1].insert(0,idx)
        Prevalence.append(run_list[16])
        Prevalence[idx+1].insert(0,idx)
        Active_YearEnd.append(run_list[17])
        Active_YearEnd[idx+1].insert(0,idx)
        Inactive_YearEnd.append(run_list[18])
        Inactive_YearEnd[idx+1].insert(0,idx)
        Relapses.append(run_list[19])
        Relapses[idx+1].insert(0,idx)
    print(arrivals)
    with open(mainfolder+"\Summaries600\Base_MonthEnd_Active_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(active_ut)
    f.close()
    with open(mainfolder+"\Summaries600\Base_MonthEnd_Inactive_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(inactive_ut)
    f.close()
    with open(mainfolder+"\Summaries600\Base_MonthEnd_Odeath_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Odeath_ut)
    f.close()
    with open(mainfolder+"\Summaries600\Base_MonthEnd_Non_Odeath_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Non_Odeath_ut)
    f.close()
    with open(mainfolder+"\Summaries600\Base_MonthEnd_Arrest_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Arrest_ut)
    f.close()
    with open(mainfolder+"\Summaries600\Base_MonthEnd_Hosp_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Hosp_ut)
    f.close()
    with open(mainfolder+"\Summaries600\Base_MonthEnd_Treat_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Treat_ut)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Arrivals.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(arrivals)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_ODeaths.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(ODeaths)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_NonODeaths.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Non_ODeaths)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Arrests.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Arrests)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Indv_Arrests.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_Arrests)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Treats.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Treats)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Indv_Treats.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_Treats)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Hosp.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Hosp)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Indv_Hosp.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_Hosp)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Prevalence.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Prevalence)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Active_YearEnd.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Active_YearEnd)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Inactive_YearEnd.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Inactive_YearEnd)
    f.close()
    with open(mainfolder+"\Summaries600\Base_Yearly_Relapses.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Relapses)
    f.close()

    ''' Cumulative calculations'''
    list_of_outcomes = [ 
        arrivals, 
        ODeaths, 
        Non_ODeaths, 
        Arrests, 
        Indv_Arrests, 
        Treats, 
        Indv_Treats, 
        Hosp, 
        Indv_Hosp, 
        Prevalence, 
        Active_YearEnd , 
        Inactive_YearEnd , 
        Relapses
    ]
    outcome_str = [ 
        "arrivals", 
        "ODeaths", 
        "Non_ODeaths", 
        "Arrests", 
        "Indv_Arrests", 
        "Treats", 
        "Indv_Treats", 
        "Hosp", 
        "Indv_Hosp", 
        "Prevalence", 
        "Active_YearEnd" , 
        "Inactive_YearEnd" , 
        "Relapses"
    ]  
    cumulative_outcomes =[]
    for idx, outcome in enumerate(list_of_outcomes):
        cumulative_outcomes.append([])
        for run, value_list in enumerate(outcome):
            del value_list[0]
            if run == 0:
                pass
            else:
                cumulative_value = 0
                cumulative_outcomes[idx].append([])
                for year, year_val in enumerate(value_list):
                    cumulative_value += year_val    
                    cumulative_outcomes[idx][run-1].append(cumulative_value)
                cumulative_outcomes[idx][run-1].insert(0,run-1)
        first_line = ["Year_" + str(i) for i in range(1,years+1)]
        first_line.insert(0,"Run")
        cumulative_outcomes[idx].insert(0,first_line)
        with open(mainfolder+"\Summaries600\Cum_"+str(outcome_str[idx])+".csv", "w", newline="") as f: #for baseline analyis
            writer = csv.writer(f)
            writer.writerows(cumulative_outcomes[idx])
        f.close()

    # try: 
    #     os.makedirs(mainfolder + "\Comparison_ED2R")
    #     os.makedirs(mainfolder + "\Comparison_MARI")
    #     os.makedirs(mainfolder + "\Comparison_CM")
    # except:
    #     pass