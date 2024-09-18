########################################
''' 
Main Simulation for Dane County Opioid use Model
last update: 4/1/24
Built in Python 3.9.0, using SimPY
To do make generators gobal so that the seeds are workking as expected and can be called. 
''' 
'''
#1 of 2 for finding longest code processes
import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()
'''
######################### Resources #########################
#resource: https://simpy.readthedocs.io/en/latest/examples/movie_renege.html 
# https://pythonhosted.org/SimPy/Manuals/SManual.html

#########################  libraries #########################
import time as ti
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import simpy
import pandas as pd
import os
import multiprocessing
from Simulation_Functions.graph_functions import *   
from Simulation_Functions.math_functions import *
from Simulation_Functions.inputs import *
from Simulation_Functions.sim_outputs_condor import *
from Simulation_Functions.DES_functions import *
from Simulation_Functions.person_class import *
#### Globals for Simualtion ######
global starttime 
global original_stdout
global warmup
##################################
starttime = ti.time()
original_stdout = sys.stdout
warmup = 5

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

######################### Single Simulation Run ######################
def simulation_run(stuff):
    (s, seeds, params) = stuff

    #################### Start of simulation run loop ###############################
    print("STARTING SIMULATION...")

    ######## Creating seperate random seeds per variable ###############
    #https://docs.python.org/3/library/random.html
    #set the desired seed numbers for each random number generator below
    gen_dict = {}
    gen_dict["arrival_gen"] = random.Random(seeds[s]['arrival'])
    gen_dict["death_gen"] = random.Random(seeds[s]['death'])
    gen_dict["relapse_gen"] = random.Random(seeds[s]['relapse'])
    gen_dict["Ocrime_gen"] = random.Random(seeds[s]['crime'])
    gen_dict["treat_gen"] = random.Random(seeds[s]['treat'])
    gen_dict["hosp_gen"] = random.Random(seeds[s]['hosp'])
    gen_dict["hosp_sub_gen"] = random.Random(seeds[s]['hosp_sub'])
    gen_dict["service_gen"] = random.Random(seeds[s]['service'])
    gen_dict["inactive_gen"] = random.Random(seeds[s]['inactive'])
    gen_dict["alldeath_gen"] = random.Random(seeds[s]['alldeath'])
    gen_dict["start_gen"] = random.Random(seeds[s]['start'])
    gen_dict["MARI_gen"] = random.Random(seeds[s]['MARI'])
    gen_dict["CM_gen"] = random.Random(seeds[s]['CM'])
    gen_dict["nonOcrime_gen"] = random.Random(seeds[s]['crime_non'])
    #print("1:", arrival_gen.randint(0,10) , " = ", treat_gen.randint(0,10), " (2): ", arrival_gen.randint(0,10), " = ", treat_gen.randint(0,10) )
   
    ########### Parameters for Run ###############
    n_runs = params["n_runs"]
    num_years = params["num_years"]
    days_per_year = params["days_per_year"]
    days_per_month = params["days_per_month"]
    
    #################### set up empty lists ###############################
    #Time in states
    Mean_enter_Age = {}
    Mean_Prev_Age = {}
    Time_FirstActive = {}
    Time_Active ={}
    Time_Treat ={}
    Time_Hosp ={}
    Time_Jail ={}
    Time_InactiveAll = {}
    Time_InactiveOnly ={}
    Time_InactiveTreat ={}
    Time_InactiveCrime ={}
    Time_InactiveHosp ={}
    #dataframes
    df_times = pd.DataFrame()

    ### RUNNING THE SIMULAITON ####
    sys.stdout = original_stdout
    print("--------------------  Scenario "+ str(s) + " Simulation Start ------------------------") 
    #set up and start the simulation 
    env = simpy.Environment()

    #intitial population in simualtion
    initial_done = False
    print(".....printing arrivas.....")
    #print("main: ",gen_dict["arrival_gen"].triangular(27298.81,34224.21,43260.59))
    [Persons, params["starting_probs"]]= generate_starting_population(gen_dict, env, initial_done,params)
    #print("main: ",gen_dict["arrival_gen"].triangular(27298.81,34224.21,43260.59))
    initial_done = True
    
    write_file = open("Results/Test_SimulationsMovement.txt", "w+")
    sys.stdout = write_file
    
    #start process and run
    i=len(Persons["dict"])
    env.process(user_arrivals(env, i, initial_done,gen_dict,params, Persons))
    #env.run(until=10)
    env.run(until=days_per_year*num_years)
    # pickle.dump(Person_Dict, open("PersonDictMARI.p", "wb"))
    # Full_Per_Dic[s] = Person_Dict
    ##################################################################### PRINTING / ANALYSIS #####################################################################################
    sys.stdout = original_stdout
    print(" SIMULATION END - ", num_years," Years - ", env.now)
    write_file = open("Results/Test_SimulationsStats_" + str(n_runs) + "Runs_" + str(s) + "Scenario.txt", "w+")
    sys.stdout = write_file
    print("----------------------- Example individual  ---------------------------")
    #print(Person_Dict[i])

    ############## Initalizing ##################
    
    #Totals 
    Total_newUsers_run = {}
    Total_InactiveUsers_run = {}
    Total_ActiveUsers_run = {}
    Total_Relapses_run = {}
    Total_OD_Deaths_run = {}
    Total_nOD_Deaths_run = {}
    Total_OCrimes_run = {}
    Total_OCrimes_uniq_run = {}
    Total_OArrests_run = {}
    Total_OArrests_uniq_run = {}
    Total_AllCrimes_run = {}
    Total_AllCrimes_uniq_run = {}
    Total_AllArrests_run = {}
    Total_AllArrests_uniq_run = {}
    Total_Treatments_run = {}
    Total_Treatments_uniq_run = {}
    Total_Hosp_run = {}
    Total_Hosp_uniq_run = {}
    Total_Prev_run ={}
    
    
    for year in range(num_years):
        Total_newUsers_run[year] = 0
        Total_InactiveUsers_run[year] = 0
        Total_ActiveUsers_run[year] = 0
        Total_Relapses_run[year] = 0
        Total_OD_Deaths_run[year] = 0
        Total_nOD_Deaths_run[year] = 0
        Total_OCrimes_run[year] = 0
        Total_OCrimes_uniq_run[year] = set()
        Total_OArrests_run[year] = 0
        Total_OArrests_uniq_run[year] =set()
        Total_AllCrimes_run[year] = 0
        Total_AllCrimes_uniq_run[year] = set()
        Total_AllArrests_run[year] = 0
        Total_AllArrests_uniq_run[year] = set()
        Total_Treatments_run[year] = 0
        Total_Treatments_uniq_run[year] = set()
        Total_Hosp_run[year] = 0
        Total_Hosp_uniq_run[year] = set()
        Total_Prev_run[year] = 0
        
    # ServiceTimes
    Mean_enter_Age[s] = {}
    Time_FirstActive[s] = {}
    Time_Active[s] ={}
    Time_Treat[s] = {}
    Time_Hosp[s] ={}
    Time_Jail[s] ={}
    Time_InactiveAll[s] = {}
    Time_InactiveOnly[s] ={}
    Time_InactiveTreat[s] ={}
    Time_InactiveCrime[s] ={}
    Time_InactiveHosp[s] ={}
    ##################### gathering event times ################################
    print("----------------------- Single Simulation Summary ---------------------------")
    #times list creation
    Time_FirstActive_list = []
    Time_inTreat_list = []
    Time_Hosp_list = []
    Time_Jail_list = []
    Time_InactiveOnly_list = []
    Time_InactiveTreat_list = []
    Time_InactiveCrime_list = []
    Time_InactiveHosp_list = []
    enter_age_list = []
    prev_age_list = []

    #### list of all death times, crimes, hospt encounters, and treatment starts  ####
    arrivals_list = []
    relapse_list = []
    OD_deaths_list = []
    nOD_death_list = []
    Ocrimes_list = []
    Allcrimes_list = []
    hosp_list = [] 
    treatments_list = []
    ####### Utilization arrays ######
    #monthly - current simulation state at the start of the month
    months = 12*num_years
    arrivals_ut = [0]*months
    hosp_ut = [0]*months
    Ocrime_ut = [0]*months
    Allcrime_ut = [0]*months
    treat_ut = [0]*months
    ODdeath_ut = [0]*months
    nOD_death_ut = [0]*months
    active_ut = [0]*months
    inactive_ut = [0]*months
    total_indivs = [0]*months

    ######## start of list creation and array counts ###### 
    ## might need a minsu 1 on the ceilings so months and year counts match up, ok since max month isnt done via range funciton
    print("..Perons with errors..")
    for d,v in Persons["dict"].items():
        prev_count = [0]*num_years
        if "arrivalT" in v:
            arrivals_list.append(v["arrivalT"])
            arrival_year = math.floor(v["arrivalT"] / days_per_year)
            if arrival_year > (num_years-0.5) :
                print("Person: ", d, ", Arrival Time: ", v, ", in Year: ", arrival_year)
                pass     
            else:
                Total_newUsers_run[arrival_year] =  Total_newUsers_run[arrival_year]+1
                for y in range(arrival_year,num_years): #add one to current and future years after arrival
                    prev_count[y] += 1
            #active utlization start arrival
            j = math.floor(v["arrivalT"]/days_per_month)
            for i in range(j,months): #average number of days in a month
                active_ut[i] += 1
                total_indivs[i] += 1
                if i == j: arrivals_ut[i] += 1
        if "OpioidDeathT" in v and not v["OpioidDeathT"] == None:
            OD_deaths_list.append(v["OpioidDeathT"])
            year = math.floor(v["OpioidDeathT"] / days_per_year)
            if year < num_years:
                Total_OD_Deaths_run[year] =  Total_OD_Deaths_run[year]+1
            for i in range(math.floor(v["OpioidDeathT"] /days_per_month),months):
                active_ut[i] -= 1
                ODdeath_ut[i] += 1
            for y in range(year+1,num_years): #subtract one from all future years after death, because cannot use in future year and will not relapse.
                prev_count[y] -= 1
        if "NonOpioidDeathT" in v and not v["NonOpioidDeathT"] == None:
            if math.floor(v["NonOpioidDeathT"] /days_per_year) > num_years: 
                pass
            else:
                nOD_death_list.append(v["NonOpioidDeathT"])
                year = math.floor(v["NonOpioidDeathT"] / days_per_year)
                if year < num_years:
                    Total_nOD_Deaths_run[year] =  Total_nOD_Deaths_run[year]+1
                for i in range(math.floor(v["NonOpioidDeathT"] /days_per_month),months):
                    if v["PrevState"] == "Ocrime":
                        Ocrime_ut[i] -= 1
                        Allcrime_ut[i] -= 1
                        nOD_death_ut[i] += 1
                    elif v["PrevState"] == "nonOcrime":
                        Allcrime_ut[i] -= 1
                        nOD_death_ut[i] += 1
                    elif v["PrevState"]  == "treat":
                        treat_ut[i] -= 1
                        nOD_death_ut[i] += 1
                    elif v["PrevState"]  == "hosp":
                        hosp_ut[i] -= 1
                        nOD_death_ut[i] += 1
                    elif v["PrevState"] == "inactive":
                        inactive_ut[i] -= 1
                        nOD_death_ut[i] += 1
                    else:
                        active_ut[i] -= 1
                        nOD_death_ut[i] += 1
                if v["PrevState"] == "active":
                    for y in range(year+1,num_years): #subtract one from all future years after death, because cannot use in future year and will not relapse.
                        prev_count[y] -= 1
        if "List_Times_inTreatment" in v and not v["List_Times_inTreatment"] == None:
            for n in v["List_Times_inTreatment"]:
                treatments_list.append(n)
                year = math.floor(n / days_per_year)
                if year < num_years:
                    Total_Treatments_uniq_run[year].add(d) #If the element already exists, the add() method does not add the element.
                    Total_Treatments_run[year] =   Total_Treatments_run[year]+1
                for i in range(math.floor(n/days_per_month),months):
                    active_ut[i] -= 1
                    treat_ut[i] += 1
                for y in range(year+1,num_years): #subtract one from all future years
                    prev_count[y] -= 1
        if "List_Times_ofCrime" in v and not v["List_Times_ofCrime"] == None:
            ###### HAVE ANOTHER INDICATIOR OF WHAT KIND OF CRIME. Then depending on Type do accoring
            for idx,n in enumerate(v["List_Times_ofCrime"]):
                Allcrimes_list.append(n)
                year = math.floor(n / days_per_year)
                if year < num_years:
                        Total_AllCrimes_uniq_run[year].add(d) # If the element already exists, the add() method does not add the element.
                        Total_AllCrimes_run[year] =  Total_OCrimes_run[year]+1
                if v["List_Crime_Type"][idx] == "Ocrime":
                    Ocrimes_list.append(n)
                    if year < num_years:
                        Total_OCrimes_uniq_run[year].add(d) # If the element already exists, the add() method does not add the element.
                        Total_OCrimes_run[year] =  Total_OCrimes_run[year]+1
                        if v["List_Crime_ExitNext"][idx] == "treatMARI":
                            pass
                        else:
                            Total_OArrests_uniq_run[year].add(d) # If the element already exists, the add() method does not add the element.
                            Total_OArrests_run[year] =  Total_OArrests_run[year]+1
                            Total_AllArrests_uniq_run[year].add(d) # If the element already exists, the add() method does not add the element.
                            Total_AllArrests_run[year] =  Total_OArrests_run[year]+1
                else:
                    if year < num_years:
                        Total_AllArrests_uniq_run[year].add(d) # If the element already exists, the add() method does not add the element.
                        Total_AllArrests_run[year] =  Total_OArrests_run[year]+1
                for i in range(math.floor(n/days_per_month),months):
                    active_ut[i] -= 1
                    Allcrime_ut[i] += 1
                    if v["List_Crime_Type"][idx] == "Ocrime":
                        Ocrime_ut[i] += 1
                for y in range(year+1,num_years): #subtract one from all future years
                    prev_count[y] -= 1
        if "List_Times_inHosp" in v and not v["List_Times_inHosp"] == None:
            for n in v["List_Times_inHosp"]:
                hosp_list.append(n)
                year = math.floor(n / days_per_year)
                if year < num_years:
                    Total_Hosp_uniq_run[year].add(d) #If the element already exists, the add() method does not add the element.
                    Total_Hosp_run[year] =  Total_Hosp_run[year]+1
                #hosp utilization enter
                for i in range(math.floor(n/days_per_month),months):
                    active_ut[i] -= 1
                    hosp_ut[i] += 1
                for y in range(year+1,num_years): #subtract one from all future years
                    prev_count[y] -= 1
        ################ utilization exit times #######################
        if "List_Treat_ExitTimes" in v and not v["List_Treat_ExitTimes"] == None:
            for n in v["List_Treat_ExitTimes"]:
                for i in range(math.floor(n/days_per_month),months):
                    treat_ut[i] -= 1
                    inactive_ut[i] +=1
        if "List_Crime_ExitTimes" in v and not v["List_Crime_ExitTimes"] == None:
            for idx, n in enumerate(v["List_Crime_ExitTimes"]):
                if v["List_Crime_ExitNext"][idx] == "inactive":
                    for i in range(math.floor(n/days_per_month),months):
                        Allcrime_ut[i] -= 1
                        inactive_ut[i] +=1
                        if  v["List_Crime_Type"][idx] == "Ocrime":
                            Ocrime_ut[i] -= 1
                elif v["List_Crime_ExitNext"][idx] == "treatCM":
                    for i in range(math.floor(n/days_per_month),months):
                        Allcrime_ut[i] -= 1
                        active_ut[i] +=1 #since going to crime next(offset)
                        if  v["List_Crime_Type"][idx] == "Ocrime":
                            Ocrime_ut[i] -= 1
                    year = math.floor(n/days_per_year)
                    for y in range(year+1,num_years): #add one to all future years to offset going to treat next
                        prev_count[y] += 1
                elif v["List_Crime_ExitNext"][idx] == "treatMARI":
                    for i in range(math.floor(n/days_per_month),months):
                        Allcrime_ut[i] -= 1
                        active_ut[i] +=1 #since going to treat next (offset)
                        if  v["List_Crime_Type"][idx] == "Ocrime":
                            Ocrime_ut[i] -= 1
                    year = math.floor(n/days_per_year)
                    for y in range(year+1,num_years):  #add one to all future years to offset going to treat next
                        prev_count[y] += 1
                else:
                    print("No index fount for next locaiton")
        if "List_Hosp_ExitTimes" in v and not v["List_Hosp_ExitTimes"] == None:
            for idx, n in enumerate(v["List_Hosp_ExitTimes"]):
                if v["List_Hosp_ExitNext"][idx] == "crime":
                    for i in range(math.floor(n/days_per_month),months):
                        hosp_ut[i] -= 1
                        #since going to crime next
                        active_ut[i] +=1 
                    year = math.floor(n/days_per_year)
                    for y in range(year+1,num_years): #add one to all future years to offset going to crime next
                        prev_count[y] += 1
                elif v["List_Hosp_ExitNext"][idx] == "treat":
                    for i in range(math.floor(n/days_per_month),months):
                        hosp_ut[i] -= 1
                        #since going to treat next
                        active_ut[i] +=1 
                    year = math.floor(n/days_per_year)
                    for y in range(year+1,num_years): #add one to all future years to offset going to treat next
                        prev_count[y] += 1
                elif v["List_Hosp_ExitNext"][idx] == "fatal":
                    for i in range(math.floor(n/days_per_month),months):
                        hosp_ut[i] -= 1
                        active_ut[i] +=1
                    year = math.floor(n/days_per_year)
                    for y in range(year+1,num_years): #add one to all future years to offset going to OPIOID!! death next  
                        prev_count[y] += 1
                elif v["List_Hosp_ExitNext"][idx] == "inactive":
                    for i in range(math.floor(n/days_per_month),months):
                        hosp_ut[i] -= 1
                        inactive_ut[i] +=1
                else:
                    print("No index fount for next locaiton")
        if "List_Times_Inactive_only" in v and not v["List_Times_Inactive_only"] == None:
            for n in v["List_Times_Inactive_only"]:
                for i in range(math.floor(n/days_per_month),months):
                    inactive_ut[i] += 1
                    active_ut[i] -=1
                for y in range(math.floor(n/ days_per_year)+1,num_years):
                    prev_count[y] -= 1
        if "List_Relapse_Time" in v and not v["List_Relapse_Time"] == None:
            for n in v["List_Relapse_Time"]:
                year = math.floor(n/days_per_year)
                if year < num_years:
                    Total_Relapses_run[year] += 1
                    relapse_list.append(n)
                for i in range(math.floor(n/days_per_month),months):
                    inactive_ut[i] -= 1
                    active_ut[i] +=1
                for y in range(math.floor(n/days_per_year),num_years): #add one to current and future years if relapse occured
                    prev_count[y] += 1

        ############### list of all time in nodes  #######################
        if "Time_FirstActive" in v and not v["Time_FirstActive"] == None:
            Time_FirstActive_list.append(v["Time_FirstActive"])
        if "List_Treat_ServiceTimes" in v and not v["List_Treat_ServiceTimes"] == None:
            for n in v["List_Treat_ServiceTimes"]:
                Time_inTreat_list.append(n)
        if "List_Crime_ServiceTimes" in v and not v["List_Crime_ServiceTimes"] == None:
            for n in v["List_Crime_ServiceTimes"]:
                Time_Jail_list.append(n)
        if "List_Hosp_ServiceTimes" in v and not v["List_Hosp_ServiceTimes"] == None:
            for n in v["List_Hosp_ServiceTimes"]:
                Time_Hosp_list.append(n)
        if "List_InactiveOnly_ServiceTimes" in v and not v["List_InactiveOnly_ServiceTimes"] == None:
            for n in v["List_InactiveOnly_ServiceTimes"]:
                Time_InactiveOnly_list.append(n)
        if "List_InactiveTreat_ServiceTimes" in v and not v["List_InactiveTreat_ServiceTimes"] == None:
            for n in v["List_InactiveTreat_ServiceTimes"]:
                Time_InactiveTreat_list.append(n)
        if "List_InactiveCrime_ServiceTimes" in v and not v["List_InactiveCrime_ServiceTimes"] == None:
            for n in v["List_InactiveCrime_ServiceTimes"]:
                Time_InactiveCrime_list.append(n)
        if "List_InactiveHosp_ServiceTimes" in v and not v["List_InactiveHosp_ServiceTimes"] == None:
            for n in v["List_InactiveHosp_ServiceTimes"]:
                Time_InactiveHosp_list.append(n)
        if "EnterAge" in v:
            if int(v["arrivalT"]) == 0:
                prev_age_list.append(v["EnterAge"])
            else:
                enter_age_list.append(v["EnterAge"])
        for y, value in enumerate(prev_count): #if the indidivudal used at all in a given year they are added to yearly prevalence. If they didn't use for a whole year an indidivduals prev_count should equal 0
            if value > 0 :
                Total_Prev_run[y] = Total_Prev_run[y] + 1

    sys.stdout = original_stdout
    print("---- Finished Dictionary Calculations for Scenario", s,"-----")
    sys.stdout = write_file
    i = 11
    for year in range(num_years):
        Total_InactiveUsers_run[year] = inactive_ut[i]
        Total_ActiveUsers_run[year] = active_ut[i]
        i +=12
    sys.stdout = original_stdout
    print("---- Finished new lists for Scenario", s,"-----")
    sys.stdout = write_file
    print("..End of error list..")
    ############## Calculate Avg, SD, MIN, MAX times ######################
    Time_FirstActive[s] = timeCalculations(Time_FirstActive_list)
    Time_Treat[s] = timeCalculations(Time_inTreat_list)
    Time_Jail[s] = timeCalculations(Time_Jail_list)
    Time_Hosp[s] = timeCalculations(Time_Hosp_list)
    Time_InactiveAll[s] = timeCalculations(Time_InactiveOnly_list+ Time_InactiveCrime_list + Time_InactiveHosp_list + Time_InactiveTreat_list)
    Time_InactiveOnly[s] = timeCalculations(Time_InactiveOnly_list)
    Time_InactiveTreat[s] = timeCalculations(Time_InactiveTreat_list)
    Time_InactiveCrime[s] = timeCalculations(Time_InactiveCrime_list)
    Time_InactiveHosp[s] = timeCalculations(Time_InactiveHosp_list)
    Mean_enter_Age[s] = timeCalculations(enter_age_list)
    Mean_Prev_Age[s] = timeCalculations(prev_age_list)
    ##################### Prints individual scenario Histograms ################################
    os.makedirs('Results/Figures', exist_ok=True)
    if (s % 50) == 0: 
        sys.stdout = original_stdout
        print("---- Creating Figures for Scenario", s,"-----")
        sys.stdout = write_file
        os.makedirs('Results/Figures/Scenario'+str(s), exist_ok=True)
        #### Hisograms of Totals ###########
        print_histogram(enter_age_list,18,s,"Age at Sim","Number of Individuals", "Age_Init")
        print_histogram(prev_age_list,18,s,"Age of Inidivduals at beginning simualtion", "Number of Individuals", "Age_Prev")
        #''' 
        # can comment out to save time
        #monthly
        print_histogram(arrivals_list,(int(months)),s,"Time of Arrival (Months)","Number of Arrivals","Arrivals_Month")  #new user arrivals in each month
        print_histogram(treatments_list,(int(months)),s,"Time of Treatment Start (Months)","Number of Treatment Starts","Treatment_Month") #number of treatment starts in each month
        print_histogram(Ocrimes_list,(int(months)),s,"Time of Crime (Months)","Number of Crimes","Crimes_Month") # number of crimes in each month
        print_histogram(hosp_list,(int(months)),s,"Time of Hospital Encounter (Months)","Number of Hospital Encounters","Hospital_Month") #number of Hospital Encounters in each month ####
        print_histogram(OD_deaths_list,(int(months)),s,"Time of Death (Months)","Number of Deaths","Deaths_Month") #number of deaths in each month ###
        #'''
        #Yearly
        # print(OD_deaths_list)
        print_histogram(arrivals_list,(int(num_years)),s,"Time of Arrival (Years)","Number of Arrivals","Arrivals_Year")  #new user arrivals in each month
        print_histogram(treatments_list,(int(num_years)),s,"Time of Treatment Start (Years)","Number of Treatment Starts","Treatment_Year") #number of treatment starts in each num_year
        print_histogram(Ocrimes_list,(int(num_years)),s,"Time of Crime (Years)","Number of Crimes","Crimes_Year") # number of crimes in each num_year
        print_histogram(hosp_list,(int(num_years)),s,"Time of Hospital Encounter (Years)","NUmber of Hospital Encounters","Hospital_Year") #number of Hospital Encounters in each num_year ####
        print_histogram(OD_deaths_list,(int(num_years)),s,"Time of Death (Years)","Number of Opioid-Related Deaths","OD_Deaths_Year") #number of Opioid deaths in each num_year ###
        print_histogram(nOD_death_list,(int(num_years)),s,"Time of Death (Years)","Number of nonOpioid-Related Deaths","nonOD_Deaths_Year") #number of non-Opioid Related deaths in each num_year ###
        print_histogram(relapse_list,(int(num_years)),s,"Time of Relapse (Years)","Number of Individuals","Relapse_Year") #number of non-Opioid Related deaths in each num_year ###
        #### Hisograms of Utilization per month ###########
        print_barChart(active_ut,s,"Time","Number of OUD Active Individuals","Ut_Active", num_years,warmup) 
        print_barChart(inactive_ut,s,"Time","Number of OUD Inactive Individuals","Ut_Inactive", num_years,warmup) 
        print_barChart(treat_ut,s,"Time","Number of Individuals in Treatment","Ut_Treatment", num_years,warmup) 
        print_barChart(Ocrime_ut,s,"Time","Number of Individuals in Criminal Justice System for Opioid-related","Ut_OCrimes", num_years,warmup) 
        print_barChart(Allcrime_ut,s,"Time","Number of Individuals in Criminal Justice System","Ut_AllCrimes", num_years,warmup) 
        print_barChart(hosp_ut,s,"Time","Number of Individuals in Hospital","Ut_Hospital", num_years,warmup)
        print_barChart(ODdeath_ut,s,"Time","Number of Deceased Individuals from Opioid-Related Causes","Ut_OD_Deaths", num_years,warmup) 
        print_barChart(nOD_death_ut,s,"Time","Number of Deceased Individuals from Non-Opioid-Related Causes","Ut_nOD_Deaths", num_years,warmup) 
        
        plt.plot(range(0,int(months)),active_ut,marker=".", color="k")
        plt.plot(range(0,int(months)),Allcrime_ut,marker=".", color="r")
        plt.plot(range(0,int(months)),ODdeath_ut,marker=".", color="b")
        plt.plot(range(0,int(months)),nOD_death_ut,marker=".", color="#a0522d")
        plt.plot(range(0,int(months)),treat_ut,marker=".", color="g")
        plt.plot(range(0,int(months)),hosp_ut,marker=".", color="m") 
        plt.legend(['Active',"Arrests",'Opioid Deaths',"non-Opioid Deaths", "Treatment", 'Hospital Encounters'])
        plt.xlabel("Month")
        plt.title("Number of people in each state at the beginning of the Month")
        plt.ylabel("Number of People")
        plt.tight_layout()
        plt.savefig('Results/Figures/Scenario'+str(s)+'/Ut_noInactive.png')
        plt.plot(range(0,int(months)),inactive_ut,marker=".", color="y")
        plt.legend(['Active', "Arrests",'Opioid Deaths',"non-Opioid Deaths", "Treatment", 'Hospital Encounters','In-active'])
        plt.savefig('Results/Figures/Scenario'+str(s)+'/Ut_all.png')
        plt.close()
        #### Hisograms of Times ###########
        #''' 
        # can comment out to save time
        print_histogram(Time_FirstActive_list,(int(months)),s,"Amount of Time in first Active State (Days)","Number of Individuals","Time_FirstActive") #Amount of time spent in active state for first time####
        print_histogram(Time_inTreat_list,(int(months)),s,"Amount of Time in Treatment State per episode (Days)","Number of Individuals","Time_inTreat")
        print_histogram(Time_Jail_list,(int(months)),s,"Amount of Time in Jail\Prison State per episode (Days)","Number of Individuals","Time_inJail")
        print_histogram(Time_Hosp_list,(int(months)),s,"Amount of Time in Hospital State per episode (Days)","Number of Individuals","Time_inHosp")
        #'''
        sys.stdout = original_stdout
        print("---- Finished Figures for Scenario", s,"-- Calculating Yearly Lists -----")
    sys.stdout = write_file
    ##################### Prints individual scenario summary ################################ 
    print("Simulation End time: ", float(days_per_year*num_years))
    # print("ODDeathTime_List: ", OD_deaths_list)
    # print("ODDeathTime_YearList: ", [math.floor(v/days_per_year) for v in OD_deaths_list])
    #is everyone accounted for, yes :)
    print("Total number of individuals:",  total_indivs[months-1])
    print("Number in active state in final month:", active_ut[months-1])
    print("Number in inactive state in final month:", inactive_ut[months-1])
    print("Number in opioid death state in final month:", ODdeath_ut[months-1])
    print("Number in non-opioid death state in final month:", nOD_death_ut[months-1])
    print("Number in Opioid-Arrest state in final month:", Ocrime_ut[months-1])
    print("Number in Any Arrest state in final month:", Allcrime_ut[months-1])
    print("Number in Hospital state in final month:", hosp_ut[months-1])
    print("Number in Treatment State in final month:", treat_ut[months-1])
    print("Number of arrivals in final month:", arrivals_ut[months-1])
    #Monthly Totals   
    print("Monthly Number in active state:", active_ut)
    print("Monthly Number in inactive state:", inactive_ut)
    print("Monthly Number in opioid death state:", ODdeath_ut)
    print("Monthly Number in non-opioid death state:", nOD_death_ut)
    print("Monthly Number in Opioid-Arrest state:", Ocrime_ut)
    print("Monthly Number in Any Arrest state:", Allcrime_ut)
    print("Monthly Number in Hospital state:", hosp_ut)
    print("Monthly Number in Treatment State:", treat_ut)
    print("Monthly Number of arrivals:", arrivals_ut)
    # Total of yearly events
    print("---------------------- Total Yearly Events ----------------------------")   
    print("Number of new user arrivals each year:, ", Total_newUsers_run)
    print("Total number of Opioid Deaths:, ", Total_OD_Deaths_run)
    print("Total number of Non-Opioid Deaths:, ", Total_nOD_Deaths_run)
    print("Total number of opioid-realted Crimes:, ", Total_OCrimes_run)
    print("Total number of Individuals that committed an opioid-related crime:, ", [len(Total_OCrimes_uniq_run[year]) for year in range(num_years)])
    print("Total number of opioid-realted Arrests:, ", Total_OArrests_run)
    print("Total number of Individuals Arrested for opioid-related crime:, ", [len(Total_OArrests_uniq_run[year]) for year in range(num_years)])
    print("Total number of Crimes:, ", Total_AllCrimes_run)
    print("Total number of Individuals that committed a crime:, ", [len(Total_AllCrimes_uniq_run[year]) for year in range(num_years)])
    print("Total number of Arrests:, ", Total_AllArrests_run)
    print("Total number of Individuals Arrested for a crime:, ", [len(Total_AllArrests_uniq_run[year]) for year in range(num_years)])
    print("Total number of Treatment Starting Episodes:, ", Total_Treatments_run)
    print("Total number of Individuals sent to Treatment:, ", [len(Total_Treatments_uniq_run[year]) for year in range(num_years)])
    print("Total number of Hospital Encounters:, ", Total_Hosp_run)
    print("Total number of Individuals with Hospital Encounters:, ", [len(Total_Hosp_uniq_run[year]) for year in range(num_years)])
    print("Total number of Yearly Prevalence: ", Total_Prev_run)
    print("Number of Inidivdual in active state at the end of each year: ", Total_ActiveUsers_run)
    print("Number of Inidivdual in inactive state at the end of each year: ", Total_InactiveUsers_run)
    print("Total Number of Yearly Relapses:", Total_Relapses_run )

    ''' WORKING CODE to change above print totals to print a dataframe instead
    Type = ["Users", "Arrivals", "Deaths",  "Arrests", "Indv_Arrests", "Treat_Starts", "Indv_Treat", "Hosp", "Indv Hosp"]
    Total = [len(Person_Dict), sum(Total_newUsers_run[year] for year in range(num_years)), sum(Total_OD_Deaths[s]), sum(Total_Crimes[s]), sum(len(Total_Crimes_uniq[s][year]) for year in range(num_years)),sum(Total_Treatments[s]), sum(Total_Treatments_uniq)]
    year0 = [0,0,0,0,0,0,0,0,0]
    df = {'Item':Type,"Total":Total}
    df_Totals = pd.DataFrame(df)
    '''
    sys.stdout = original_stdout
    print("---- Starting Time Summaries for Scenario ", s,"-----")
    sys.stdout = write_file
    ##### Time First Active #####
    Type = ["First Active State"]
    Avg = [Time_FirstActive[s]["Avg"]]
    sd = [Time_FirstActive[s]["SD"]]
    length_n = [Time_FirstActive[s]["n"]]
    CI = [final_CI(Time_FirstActive[s]["Avg"],Time_FirstActive[s]["SD"],length_n)]
    minn = [Time_FirstActive[s]["Min"]]
    maxx = [Time_FirstActive[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)
    # print("----------- Time First Active Summary DF ---------------")
    # print(df_times.loc[s,:].to_string())
    
    ##### Time in Treatment #####
    Type = ["Treatment State"]
    Avg = [Time_Treat[s]["Avg"]]
    sd = [Time_Treat[s]["SD"]]
    CI = [final_CI(Time_Treat[s]["Avg"],Time_Treat[s]["SD"],len(Time_inTreat_list))]
    minn = [Time_Treat[s]["Min"]]
    maxx = [Time_Treat[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)
    
    ##### Time in Jail #####
    Type = ["Jail State"]
    Avg = [Time_Jail[s]["Avg"]]
    sd = [Time_Jail[s]["SD"]]
    CI = [final_CI(Time_Jail[s]["Avg"],Time_Jail[s]["SD"],len(Time_Jail_list))]
    minn = [Time_Jail[s]["Min"]]
    maxx = [Time_Jail[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    ##### Time in Hosp #####
    Type = ["Hospital State"]
    Avg = [Time_Hosp[s]["Avg"]]
    sd = [Time_Hosp[s]["SD"]]
    CI = [final_CI(Time_Hosp[s]["Avg"],Time_Hosp[s]["SD"],len(Time_Hosp_list))]
    minn = [Time_Hosp[s]["Min"]]
    maxx = [Time_Hosp[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    ##### Time in inactive only #####
    Type = ["InactiveAll State"]
    Avg = [Time_InactiveAll[s]["Avg"]]
    sd = [Time_InactiveAll[s]["SD"]]
    CI = [final_CI(Time_InactiveAll[s]["Avg"],Time_InactiveAll[s]["SD"],len(Time_InactiveOnly_list + Time_InactiveHosp_list + Time_InactiveTreat_list + Time_InactiveCrime_list))]
    minn = [Time_InactiveAll[s]["Min"]]
    maxx = [Time_InactiveAll[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    ##### Time in inactive only #####
    Type = ["InactiveOnly State"]
    Avg = [Time_InactiveOnly[s]["Avg"]]
    sd = [Time_InactiveOnly[s]["SD"]]
    CI = [final_CI(Time_InactiveOnly[s]["Avg"],Time_InactiveOnly[s]["SD"],len(Time_InactiveOnly_list))]
    minn = [Time_InactiveOnly[s]["Min"]]
    maxx = [Time_InactiveOnly[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    ##### Time in inactive after treatment #####
    Type = ["InactiveTreat State"]
    Avg = [Time_InactiveTreat[s]["Avg"]]
    sd = [Time_InactiveTreat[s]["SD"]]
    CI = [final_CI(Time_InactiveTreat[s]["Avg"],Time_InactiveTreat[s]["SD"],len(Time_InactiveTreat_list))]
    minn = [Time_InactiveTreat[s]["Min"]]
    maxx = [Time_InactiveTreat[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    ##### Time in inactive after Crime #####
    Type = ["InactiveCrime State"]
    Avg = [Time_InactiveCrime[s]["Avg"]]
    sd = [Time_InactiveCrime[s]["SD"]]
    CI = [final_CI(Time_InactiveCrime[s]["Avg"],Time_InactiveCrime[s]["SD"],len(Time_InactiveCrime_list))]
    minn = [Time_InactiveCrime[s]["Min"]]
    maxx = [Time_InactiveCrime[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    ##### Time in inactive after Hosp #####
    Type = ["InactiveHosp State"]
    Avg = [Time_InactiveHosp[s]["Avg"]]
    sd = [Time_InactiveHosp[s]["SD"]]
    CI = [final_CI(Time_InactiveHosp[s]["Avg"],Time_InactiveHosp[s]["SD"],len(Time_InactiveHosp_list))]
    minn = [Time_InactiveHosp[s]["Min"]]
    maxx = [Time_InactiveHosp[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    ##### Enter Age #####
    Type = ["Enter Age"]
    Avg = [Mean_enter_Age[s]["Avg"]]
    sd = [Mean_enter_Age[s]["SD"]]
    CI = [final_CI(Mean_enter_Age[s]["Avg"],Mean_enter_Age[s]["SD"],len(enter_age_list))]
    minn = [Mean_enter_Age[s]["Min"]]
    maxx = [Mean_enter_Age[s]["Max"]]
    df = pd.DataFrame(np.column_stack([Type,Avg,sd,CI,minn,maxx]), columns=['State','Avg_Time','sd_Time','CI','Min_Time','MaxTime'])
    df.round(decimals = 5)
    #append to larger dataframe
    df_times = pd.concat([df_times, df], ignore_index=True)

    print("----------- Time Summary DF ---------------")
    #print(df_times.loc[9*s:(9*s)+9,:].to_string())
    print(df_times.to_string())
    
    sys.stdout = original_stdout
    write_file.close()
    print("---- Finished Sumary of Scenario ", s,"-----")
    return Total_newUsers_run,Total_Relapses_run, Total_ActiveUsers_run, Total_Prev_run, Total_OD_Deaths_run, Total_nOD_Deaths_run, Total_OCrimes_run, Total_OCrimes_uniq_run, Total_Treatments_run, Total_Treatments_uniq_run, Total_Hosp_run, Total_Hosp_uniq_run, Total_InactiveUsers_run,df_times,Total_AllCrimes_run
    # breakpoint()
    
if __name__ == '__main__':
    os.makedirs('Results', exist_ok=True)
    # original_stdout = sys.stdout
    params = sim_params(warmup)
    
    #################### generate event times from LN distribution ###############################
    ########################################## Ru,,02n for n_runs ######################################################################
    seeds = {0: {'arrival': 35484, "death": 568, "relapse":25677, "crime":8, "treat":103, "hosp": 6944, "hosp_sub": 6935, "service": 53215, "inactive": 234235, "alldeath" : 9289, "start" : 9274, "MARI": 321548, "CM": 226948, "crime_non" : 12345 } }

    for s in range(params["n_runs"]):
        seeds[s+1] ={}
        for t in seeds[s]:
            seeds[s+1][t] = ((seeds[s][t]*5) + (s+1)) % 100000   

    if params["n_runs"] <= 8:
        CPUs = params["n_runs"]
    else:
        CPUs = 8

    debug = True
    if debug:
        results = []
        for s in range(params["n_runs"]):
            results.append(simulation_run([s, seeds, params]))
    else:
        pool = multiprocessing.Pool(CPUs) #number of CPUs requested
        args = []
        for s in range(params["n_runs"]):
            args.append([s, seeds, params])
        results = pool.map(simulation_run, args)

    sys.stdout = original_stdout#FullSim_file    
    endtime = ti.time()
    print ('total time elapsed:', endtime - starttime)
    print("...Simulation Finished. Starting Summary Statistics...")
    print_model_outputs(warmup, seeds, starttime, original_stdout, results, params)

    '''
    #Working code...

    breakpoint()

    '''
    '''
    #2 of 2 for finding longest code processes
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    '''