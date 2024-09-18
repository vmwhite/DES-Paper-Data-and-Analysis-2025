'''' This file takes in the raw text files from a results folder and converts them to summary CSV files'''
import re
import os
import ast
import csv
import pandas as pd
import numpy as np
n= 600
cwd = os.getcwd() 


flist = [r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606',
r'Results_1Process_ED2RVal_30000_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_194511',
r'Results_2Process_ED2RVal_45000_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_210127',
r'Results_3Process_ED2RVal_60000_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_151544',
r'Results_4Process_ED2RVal_75000_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_201600',
r'Results_5Process_ED2RVal_90000_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_143934',
r'Results_6Process_ED2RVal_22270_MARIVal_20_CMVal_0_Scen_600_Years_25_Time_051724_194618',
r'Results_7Process_ED2RVal_22270_MARIVal_40_CMVal_0_Scen_600_Years_25_Time_051724_143736',
r'Results_8Process_ED2RVal_22270_MARIVal_60_CMVal_0_Scen_600_Years_25_Time_051724_192330',
r'Results_9Process_ED2RVal_22270_MARIVal_80_CMVal_0_Scen_600_Years_25_Time_051724_150123',
r'Results_10Process_ED2RVal_22270_MARIVal_100_CMVal_0_Scen_600_Years_25_Time_051724_141226',
r'Results_11Process_ED2RVal_22270_MARIVal_0_CMVal_20_Scen_600_Years_25_Time_051724_135815',
r'Results_12Process_ED2RVal_22270_MARIVal_0_CMVal_40_Scen_600_Years_25_Time_051724_135714',
r'Results_13Process_ED2RVal_22270_MARIVal_0_CMVal_60_Scen_600_Years_25_Time_051724_140819',
r'Results_14Process_ED2RVal_22270_MARIVal_0_CMVal_80_Scen_600_Years_25_Time_051724_140527',
r'Results_15Process_ED2RVal_22270_MARIVal_0_CMVal_100_Scen_600_Years_25_Time_051724_142528',
r'Results_16Process_ED2RVal_40000_MARIVal_20_CMVal_20_Scen_600_Years_25_Time_051724_150754',
r'Results_17Process_ED2RVal_60000_MARIVal_40_CMVal_40_Scen_600_Years_25_Time_051724_145529',
r'Results_18Process_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_600_Years_25_Time_051724_164846']

for f_idx,f in enumerate(flist):
    mainfolder = f #ED2Recovery Scenario
    #
    try: 
        os.makedirs(mainfolder + "\summaries" +str(n))
        os.remove(mainfolder + "\Test_SimulationsMovement.txt")
    except:
        pass

    ''' The following is for calculating Service Time Values'''
    Service_Times = []
    file_list = os.listdir(mainfolder)
    scen = 0
    for idx, file in enumerate(file_list):
        if file.endswith('Scenario.txt'):
            fin = open(mainfolder+"/"+file, 'r')
            lines = fin.readlines()
            run = re.findall(r'\d+', str(file_list[idx]))
            if int(run[1]) == 0:
                skipsA = -1
                skipsB = 46
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
    with open(mainfolder+"\Summaries"+str(n)+"\Base_ServiceTimes.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Service_Times)
    f.close()

    ''' The following is for calculating Main Output Values'''
    Main_Outputs = []
    file_list = os.listdir(mainfolder)
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
    OArrest_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    OArrest_ut[0].insert(0,"Run")
    anyArrest_ut = [["Month_" + str(i) for i in range(1,years+1)]]
    anyArrest_ut[0].insert(0,"Run")
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
    OCrimes = [["Year_" + str(i) for i in range(1,years+1)]]
    OCrimes[0].insert(0,"Run")
    Indv_OCrimes = [["Year_" + str(i) for i in range(1,years+1)]]
    Indv_OCrimes[0].insert(0,"Run")
    OArrests = [["Year_" + str(i) for i in range(1,years+1)]]
    OArrests[0].insert(0,"Run")
    Indv_OArrests = [["Year_" + str(i) for i in range(1,years+1)]]
    Indv_OArrests[0].insert(0,"Run")
    anyCrimes = [["Year_" + str(i) for i in range(1,years+1)]]
    anyCrimes[0].insert(0,"Run")
    Indv_anyCrimes = [["Year_" + str(i) for i in range(1,years+1)]]
    Indv_anyCrimes[0].insert(0,"Run")
    anyArrests = [["Year_" + str(i) for i in range(1,years+1)]]
    anyArrests[0].insert(0,"Run")
    Indv_anyArrests = [["Year_" + str(i) for i in range(1,years+1)]]
    Indv_anyArrests[0].insert(0,"Run")
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
        OArrest_ut.append(run_list[4])
        OArrest_ut[idx+1].insert(0,idx)
        anyArrest_ut.append(run_list[5])
        anyArrest_ut[idx+1].insert(0,idx)
        Hosp_ut.append(run_list[6])
        Hosp_ut[idx+1].insert(0,idx)
        Treat_ut.append(run_list[7])
        Treat_ut[idx+1].insert(0,idx)
        arrivals.append(run_list[8])
        arrivals[idx+1].insert(0,idx)
        ODeaths.append(run_list[9])
        ODeaths[idx+1].insert(0,idx)
        Non_ODeaths.append(run_list[10])
        Non_ODeaths[idx+1].insert(0,idx)
        OCrimes.append(run_list[11])
        OCrimes[idx+1].insert(0,idx)
        Indv_OCrimes.append(run_list[12])
        Indv_OCrimes[idx+1].insert(0,idx)
        OArrests.append(run_list[13])
        OArrests[idx+1].insert(0,idx)
        Indv_OArrests.append(run_list[14])
        Indv_OArrests[idx+1].insert(0,idx)
        anyCrimes.append(run_list[15])
        anyCrimes[idx+1].insert(0,idx)
        Indv_anyCrimes.append(run_list[16])
        Indv_anyCrimes[idx+1].insert(0,idx)
        anyArrests.append(run_list[17])
        anyArrests[idx+1].insert(0,idx)
        Indv_anyArrests.append(run_list[18])
        Indv_anyArrests[idx+1].insert(0,idx)
        Treats.append(run_list[19])
        Treats[idx+1].insert(0,idx)
        Indv_Treats.append(run_list[20])
        Indv_Treats[idx+1].insert(0,idx)
        Hosp.append(run_list[21])
        Hosp[idx+1].insert(0,idx)
        Indv_Hosp.append(run_list[22])
        Indv_Hosp[idx+1].insert(0,idx)
        Prevalence.append(run_list[23])
        Prevalence[idx+1].insert(0,idx)
        Active_YearEnd.append(run_list[24])
        Active_YearEnd[idx+1].insert(0,idx)
        Inactive_YearEnd.append(run_list[25])
        Inactive_YearEnd[idx+1].insert(0,idx)
        Relapses.append(run_list[26])
        Relapses[idx+1].insert(0,idx)
    print(arrivals)
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_Active_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(active_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_Inactive_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(inactive_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_Odeath_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Odeath_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_Non_Odeath_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Non_Odeath_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_OArrest_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(OArrest_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_anyArrest_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(anyArrest_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_Hosp_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Hosp_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_MonthEnd_Treat_ut.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Treat_ut)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Arrivals.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(arrivals)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_ODeaths.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(ODeaths)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_NonODeaths.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Non_ODeaths)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_OArrests.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(OArrests)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Indv_OArrests.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_OArrests)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_OCrimes.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(OCrimes)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Indv_OCrimes.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_OCrimes)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_anyCrimes.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(anyCrimes)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Indv_anyCrimes.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_anyCrimes)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_anyArrests.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(anyArrests)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Indv_anyArrests.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_anyArrests)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Treats.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Treats)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Indv_Treats.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_Treats)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Hosp.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Hosp)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Indv_Hosp.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Indv_Hosp)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Prevalence.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Prevalence)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Active_YearEnd.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Active_YearEnd)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Inactive_YearEnd.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Inactive_YearEnd)
    f.close()
    with open(mainfolder+"\Summaries"+str(n)+"\Base_Yearly_Relapses.csv", "w", newline="") as f: #for baseline analyis
        writer = csv.writer(f)
        writer.writerows(Relapses)
    f.close()

    ''' Cumulative calculations'''
    list_of_outcomes = [ 
        arrivals, 
        ODeaths, 
        Non_ODeaths, 
        OCrimes, 
        Indv_OCrimes, 
        OArrests, 
        Indv_OArrests, 
        anyCrimes, 
        Indv_anyCrimes, 
        anyArrests, 
        Indv_anyArrests, 
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
        "OCrimes", 
        "Indv_OCrimes", 
        "OArrests", 
        "Indv_OArrests", 
        "anyCrimes", 
        "Indv_anyCrimes", 
        "anyArrests", 
        "Indv_anyArrests", 
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
        with open(mainfolder+"\Summaries"+str(n)+"\Cum_"+str(outcome_str[idx])+".csv", "w", newline="") as f: #for baseline analyis
            writer = csv.writer(f)
            writer.writerows(cumulative_outcomes[idx])
        f.close()
