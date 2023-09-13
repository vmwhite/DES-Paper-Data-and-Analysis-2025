########################################
''' 
Main Simulation for Dane County Opioid user Model
last update: 1/4/23
Built in Pythin 3.9.0
Using SimPY
Scenario:
    A community has limited number of treatment capacity that share common funds. 
    Using individuals arrive randomly to treatment, request one type of treatment, and
    go into into recovery and cannot overdose while in recovery.

    Individuals in recovery relapse randomly. Any using individuals fataly overdose randomly.
    Additionally using individuals randomly commit drug related crimes. Th

interesting to look at use of impact of better and more treeatment also look at police encounters.  
''' 
'''
#1 of 2 for finding longest code processes
import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()
'''
########################################
#resource: https://simpy.readthedocs.io/en/latest/examples/movie_renege.html 
# https://pythonhosted.org/SimPy/Manuals/SManual.html
##### libraries ######
# from pickle import TRUE
# from re import L
# from fcntl import F_SEAL_SEAL
# from codecs import oem_decode
import pickle
import time as ti
import sys
import random
import math
import numpy as np
from numpy import True_, double, log as ln
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from pandas.core.arrays.sparse import dtype
from pandas.core.tools.numeric import to_numeric
import scipy as scipy
import scipy.stats as stats
import simpy
import pandas as pd
from functools import partial, wraps
import os
from datetime import datetime
import multiprocessing
starttime = ti.time()
original_stdout = sys.stdout
warmup = 5

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

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

########################################## graphing funcitons ##########################################
def print_histogram(list,bins_num,s,stringx,stringy,name):
    plt.figure(figsize=(9, 6),dpi=300)
    #plt.hist(list,num_years) 
    #plt.xlabel("Years")
    # print(name)
    freq, bins, patches = plt.hist(list,bins_num) 
    # print("list length: ",len(list))
    # print("num bins:", bins_num)
    # print("freq: ", freq)
    # print("bins: ", bins )
    # print("patches: ", patches)
    # x coordinate for labels
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

def print_barChart(data,s,stringx,stringy,name,num_years):
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

def ci_graph(sim_avg, sim_max, sim_min, sim_sd, e_year, e_lower, e_mean,name):
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

def ci_graph_point(sim_avg, sim_max, sim_min, sim_sd, e_points,e_col,name):
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
    
        
############# IMPORTING DATA #################################
def simulation_run(stuff):

    (s, n_runs, num_years, days_per_year, days_per_month, start_year, seeds, params) = stuff
    lam_user_arrival = params["lam_user_arrival"]
    LNmean_deathdays = params["LNmean_deathdays"]
    LNsigma_deathdays = params["LNsigma_deathdays"]
    LNmean_hospdays = params["LNmean_hospdays"]
    LNsigma_hospdays = params["LNsigma_hospdays"]
    LNmean_arrestdays = params["LNmean_arrestdays"]
    LNsigma_arrestdays = params["LNsigma_arrestdays"]
    LNmean_treatdays = params["LNmean_treatdays"]
    LNsigma_treatdays = params["LNsigma_treatdays"]
    hospital_encounter_thres = params["hospital_encounter_thres"]
    LNmean_iadays = params["LNmean_iadays"]
    LNsig_iadays = params["LNsig_iadays"]
    dup_prev_age_mean = params["dup_prev_age_mean"]
    dup_prev_age_sig = params["dup_prev_age_sig"]
    dup_init_age_mean =  params["dup_init_age_mean"] 
    dup_init_age_sig = params["dup_init_age_sig"]
    LNmean_hospservice = params["LNmean_hospservice"]
    LNsig_hospservice = params["LNsig_hospservice"]
    LNmean_crimeservice = params["LNmean_crimeservice"]
    LNsig_crimeservice = params["LNsig_crimeservice"]
    LNmean_treatservice = params["LNmean_treatservice"]
    LNsig_treatservice = params["LNsig_treatservice"]
    LNmean_crimerel = params["LNmean_crimerel"]
    LNsig_crimerel = params["LNsig_crimerel"]
    LNmean_treatrel = params["LNmean_treatrel"]
    LNsig_treatrel = params["LNsig_treatrel"]
    LNmean_hosprel = params["LNmean_hosprel"]
    LNsig_hosprel = params["LNsig_hosprel"]
    LNmean_iarel = params["LNmean_iarel"]
    LNsig_iarel = params["LNsig_iarel"]
    hospital_encounter_thres_base = params["HE_thres_baseline"]
    ED2R_start_time = params["ED2R_start_time"] 
    MARI_threshold = params["MARI_thres"]
    MARI_start_time = params["MARI_start_time"]
    CM_threshold = params["CM_thres"]
    CM_start_time = params["CM_start_time"]
    
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

    #################### Start of simulation run loop ###############################
    print("STARTING SIMULATION...")

    #FullSim_file = open("Results/Test_FullSimulation_"+ str(n_runs)+"Runs.txt", "w+")
    #sys.stdout = FullSim_file
    # sys.stdout = original_stdout
    print("Dane County Simulation")

#################### Start of simulation run loop ###############################

# for s in range(n_runs):

################################################## Creating seperate random seeds per variable ####################################
    #https://docs.python.org/3/library/random.html
    #set the desired seed numbers for each random number generator below
    arrival_gen = random.Random(seeds[s]['arrival'])
    death_gen = random.Random(seeds[s]['death'])
    relapse_gen = random.Random(seeds[s]['relapse'])
    crime_gen = random.Random(seeds[s]['crime'])
    treat_gen = random.Random(seeds[s]['treat'])
    hosp_gen = random.Random(seeds[s]['hosp'])
    hosp_sub_gen = random.Random(seeds[s]['hosp_sub'])
    service_gen = random.Random(seeds[s]['service'])
    inactive_gen = random.Random(seeds[s]['inactive'])
    alldeath_gen = random.Random(seeds[s]['alldeath'])
    start_gen = random.Random(seeds[s]['start'])
    MARI_gen = random.Random(seeds[s]['MARI'])
    CM_gen = random.Random(seeds[s]['CM'])
    #print("1:", arrival_gen.randint(0,10) , " = ", treat_gen.randint(0,10), " (2): ", arrival_gen.randint(0,10), " = ", treat_gen.randint(0,10) )

    ######################################################## Setting Parameters #################################################################
    #Note: Triangular funciton is (low, high, mode)
    def arrival_time():
        time = arrival_gen.expovariate(lam_user_arrival)
        return time 
    
    def OR_death_time(now):
        # time = max_deathdays*death_gen.betavariate(BetaA1_deathdays,BetaA2_deathdays)
        year_shift = 2019
        if now <= days_per_year*(year_shift - start_year):
            time = death_gen.lognormvariate(LNmean_deathdays[0],LNsigma_deathdays[0])
        else:
            time = death_gen.lognormvariate(LNmean_deathdays[1],LNsigma_deathdays[1])
        return time

    def crime_time():
        time = crime_gen.lognormvariate(LNmean_arrestdays,LNsigma_arrestdays)
        # time = max_arrestdays*crime_gen.betavariate(BetaA1_arrestdays,BetaA2_arrestdays)
        return time

    def treat_time():
        time = treat_gen.lognormvariate(LNmean_treatdays,LNsigma_treatdays)
        # time = max_treatdays*treat_gen.betavariate(BetaA1_treatdays,BetaA2_treatdays)
        return time

    def hosp_time():
        time = hosp_gen.lognormvariate(LNmean_hospdays,LNsigma_hospdays)
        # time = max_hospdays*hosp_gen.betavariate(BetaA1_hospdays,BetaA2_hospdays)
        return time
    def alldeath_time(loopdone):
        #Based on Lewer 2020 age histogram ofdrug using cohort in australia
        if loopdone == False: #prevalence age distribtuion
            age = 12+ alldeath_gen.lognormvariate(dup_prev_age_mean,dup_prev_age_sig)
        else: #initiation age distribtuion
            age = 12 + alldeath_gen.lognormvariate(dup_init_age_mean, dup_init_age_sig)
        if age >101:
            age = 100
        index_age = math.floor((age/5)-2)
        deathAge_ranges = [x for x in range(math.floor(age/5)*5,101,5)]
        #based on adjusted death rates for non-drug abuse 1999-2001
        num_surviving = [99098,99000,98677,98245,97830,97330,96620,95556,93956,91626,88047,82727,75263,64974,51153,34705,18602,6921,1489]
        num_die= [98, 323, 432, 414,500,710,1065,1600,2330,3579,5320,7464,10289,13821,16449,16103,11681,5432,1489]
        LIFE_exp_probs = []
        for x in range(index_age, len(num_surviving)):
            LIFE_exp_probs.append(num_die[x]/num_surviving[index_age])
        rand_deathAgegroup = alldeath_gen.choices(deathAge_ranges, weights=LIFE_exp_probs, k=1)
        rand_deathAgegroup= float(rand_deathAgegroup[0])
        if age > rand_deathAgegroup:
            time = alldeath_gen.uniform(age*365.25,(rand_deathAgegroup+5)*365.25) - age*365.25
        else:
            time = alldeath_gen.uniform(rand_deathAgegroup*365.25,(rand_deathAgegroup+5)*365.25) - age*365.25
        return time, age
    

    def inactive_time(): 
        time = inactive_gen.lognormvariate(LNmean_iadays,LNsig_iadays)
        return time
    def service_time(Itype):
        if Itype == 'crime':
            time = service_gen.lognormvariate(LNmean_crimeservice,LNsig_crimeservice) 
        elif Itype == 'treatment':    
            time = service_gen.lognormvariate(LNmean_treatservice,LNsig_treatservice) 
        elif Itype == 'hosp':
            time = service_gen.lognormvariate(LNmean_hospservice,LNsig_hospservice) 
        elif Itype == 'inactive': #no service time for individuals not actively using only set a relapse time (aka time in inactive state)
            time = 0
        return time

    def relapse_time(Itype, ArrivalTime, CurrentTime, NatDeath): #estimate of time until next use
        if ArrivalTime == 0 and CurrentTime == 0:
            if  Itype == 'inactive': #split up starting inactive distribtuions
                # breakpoint()
                RN = start_gen.random()
                if RN <    (starting_probs[1] / starting_probs[4]): #arrest
                    Itype = "crime"
                elif RN <    ((starting_probs[1] / starting_probs[4]) + (starting_probs[2]/ starting_probs[4])): #hospital
                    Itype = "hosp"
                elif RN <    ((starting_probs[1] / starting_probs[4]) + (starting_probs[2]/ starting_probs[4])+(starting_probs[3]/ starting_probs[4])): #treatment
                    Itype = "treatment"
            if Itype == 'crime': 
                time = relapse_gen.lognormvariate(LNmean_crimerel,LNsig_crimerel) 
            elif Itype == 'treatment':
                time = relapse_gen.lognormvariate(LNmean_treatrel,LNsig_treatrel) 
                # alternative option to break up - percent of people that complete treatment:
                # if they complete treatment
                # if they don't complete treatment    
            elif Itype == 'hosp':
                time = relapse_gen.lognormvariate(LNmean_hosprel,LNsig_hosprel) 
            else:
                time = relapse_gen.lognormvariate(LNmean_iarel,LNsig_iarel) 
        else:
            if Itype == 'crime': 
                time = relapse_gen.lognormvariate(LNmean_crimerel,LNsig_crimerel) 
            elif Itype == 'treatment':
                time = relapse_gen.lognormvariate(LNmean_treatrel,LNsig_treatrel)   
            elif Itype == 'hosp':
                time = relapse_gen.lognormvariate(LNmean_hosprel,LNsig_hosprel)
            elif  Itype == 'inactive': 
                time = relapse_gen.lognormvariate(LNmean_iarel,LNsig_iarel)
        return time
    
    

    ################ Creating Performance Metrics ##########s###############

    Person = {} #Simpy dict of Users
    Person_Dict = {} #Dictionary of Users: arrivalT, nextCrimeT, nextTreatT
    i = 0


    ################################################################# MODEL #####################################################################################
    class User(object):
        def __init__(self, env,i):
            #creates a new user as process that
            self.env = env
            self.action = env.process(self.setUser())
            self.user_type = 'nru' #User types: nru -non-relapsed users, rnu - recovering non-user, ru - relapsed users, d - deceased
            self.isAlive = True #status type: alive = True, dead=False, used to exit user loop
            self.num = i #numbered person
            self.timeofNonOpioidDeath, self.EnterAge = alldeath_time(intital_done) 
            self.timeofNonOpioidDeath = self.timeofNonOpioidDeath + self.env.now
            self.timeofFatalOD = OR_death_time(self.env.now) + self.env.now
            # print('New user %d Created. %s at time %d.' % (self.num, self.getUserType(), self.env.now))
        
        def setUser(self):
            Person_Dict[self.num]["isAlive"] = True
            Person_Dict[self.num]["EnterAge"] = self.EnterAge
            Person_Dict[self.num]["Num_inTreatment"] = 0
            Person_Dict[self.num]["List_Times_inTreatment"] = []
            Person_Dict[self.num]["List_Treat_ServiceTimes"] = []
            Person_Dict[self.num]["List_Treat_ExitTimes"] = []
            Person_Dict[self.num]["List_InactiveTreat_ServiceTimes"] = []
            Person_Dict[self.num]["Num_Crimes"] = 0 
            Person_Dict[self.num]["List_Times_ofCrime"] = []
            Person_Dict[self.num]["List_Crime_ServiceTimes"] = []
            Person_Dict[self.num]["List_Crime_ExitTimes"] = []
            Person_Dict[self.num]["List_InactiveCrime_ServiceTimes"] = []
            Person_Dict[self.num]["Num_Hosp"] = 0
            Person_Dict[self.num]["List_Times_inHosp"] = []
            Person_Dict[self.num]["List_Hosp_ServiceTimes"] = []
            Person_Dict[self.num]["List_Hosp_ExitTimes"] = []
            Person_Dict[self.num]["List_Hosp_ExitNext"] = []
            Person_Dict[self.num]["List_InactiveHosp_ServiceTimes"] = []
            Person_Dict[self.num]["Num_Inactive_only"] = 0
            Person_Dict[self.num]["List_Times_Inactive_only"] = []
            Person_Dict[self.num]["List_InactiveOnly_ServiceTimes"] = []
            Person_Dict[self.num]["List_Relapse_Time"] = []
            Person_Dict[self.num]["OpioidDeathT"] = None
            Person_Dict[self.num]["NonOpioidDeathT"] = None
            Person_Dict[self.num]["OpioidDeath"] = None
            Person_Dict[self.num]["List_Crime_ExitNext"] = []
            if "Next_Interrupt" in Person_Dict[self.num]:
                Person_Dict[self.num]["PrevState"] = Person_Dict[self.num]["Next_Interrupt"]
                Person_Dict[self.num]["Time_FirstActive"] = float('nan')
            else:
                Person_Dict[self.num]["Next_Interrupt"] = ''
                Person_Dict[self.num]["PrevState"] = "active"
            while self.isAlive: #continues until individual exits the simlation via overdose death.           
                # Non-Relapsed User
                if self.user_type == 'nru':
                    Person_Dict[self.num]["TimeofFatalD"] = self.timeofNonOpioidDeath 
                    Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD 
                    # print('person %d has become a %s and activated thier disease at time %f and has a scheduled death time at time %f' % (self.num, self.getUserType(), self.env.now,self.timeofFatalOD + env.now))
                # Relapsed User
                else:
                    self.user_type = 'ru'
                    self.timeofFatalOD = OR_death_time(self.env.now) + self.env.now # see fxn def #
                    Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD 
                    Person_Dict[self.num]["PrevState"] = "active"
                    # print('person %d has become a %s and activated thier disease at time %f and has a scheduled death time at time %f' % (self.num, self.getUserType(), self.env.now,self.timeofFatalOD + env.now))
                # Process to set Deceased
                try:
                    yield self.env.process(self.NextEvent(env,Person[self.num],Person_Dict[self.num],self.num))
                    yield self.env.timeout(0.00001) 
                    if self.user_type == 'd':
                        sys.exit("Error: This person is already deceased")
                    else:
                        if self.env.now > float(days_per_year*num_years) or min(self.timeofFatalOD, self.timeofNonOpioidDeath) > float(days_per_year*num_years):
                            print(self.env.now, ">?", float(days_per_year*num_years))
                            print(self.timeofFatalOD, ">?", float(days_per_year*num_years))
                            print(self.timeofNonOpioidDeath, ">?",float(days_per_year*num_years))
                            print("")
                            pass
                        else:
                            self.user_type = 'd'
                            # print('person %d is %s and decesed time %f.' % (self.num,self.getUserType(), self.env.now))
                            if self.timeofFatalOD < self.timeofNonOpioidDeath:
                                Person_Dict[self.num]["OpioidDeath"] = True
                                Person_Dict[self.num]["OpioidDeathT"] = self.env.now
                            else:
                                Person_Dict[self.num]["OpioidDeath"] = False
                                Person_Dict[self.num]["NonOpioidDeathT"] = self.env.now
                            Person_Dict[self.num]["isAlive"] = False
                            Person_Dict[self.num]["State"] = "death"
                            self.isAlive= False
                # Interuptions from setting Deceased
                except simpy.Interrupt as interrupt:
                    if self.user_type == 'nru' or self.user_type == 'ru':
                        if self.user_type == 'nru':
                            #Amount of time person is originally active until first state
                            if "Time_FirstActive" in Person_Dict[self.num]:
                                pass
                            else:
                                Person_Dict[self.num]["Time_FirstActive"] = self.env.now - Person_Dict[self.num]["arrivalT"]
                        self.user_type = 'rnu' #set user type to a recovering non-user. Person goes to treatment either via treatment or jail
                        enter_time = self.env.now
                        # Treatment Interruption
                        if interrupt.cause == 'treatment':
                            s_time = service_time(interrupt.cause) # Service time depending on interruption type, see fxn def 
                            r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now, self.timeofNonOpioidDeath) # see fxn def 
                            self.sobriety_duration = s_time + r_time
                            self.timeofFatalOD = 10000000 #cannot die from opioid related cause while in treatment
                            Person_Dict[self.num]["State"] = "Treatment"
                            Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD + self.env.now
                            Person_Dict[self.num]["Num_inTreatment"] = Person_Dict[self.num]["Num_inTreatment"] + 1
                            Person_Dict[self.num]["List_Times_inTreatment"].append(enter_time)
                            Person_Dict[self.num]["List_Treat_ServiceTimes"].append(s_time)
                            Person_Dict[self.num]["List_Treat_ExitTimes"].append(enter_time+s_time)
                            Person_Dict[self.num]["List_InactiveTreat_ServiceTimes"].append(r_time)
                            Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                            # print('person %d is a %s at time %f and is in treatment. They will be in treatment for %f time and stop treatment at time %f. After treatment, their opioid use will remain inactive for %f time and will reactivate at time %f' % (self.num,self.getUserType(), enter_time,s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                        # Crime Interruption
                        elif interrupt.cause == 'crime':
                            self.timeofFatalOD = 10000000 #cannot die from opioid related cause until relapse
                            Person_Dict[self.num]["State"] = "crime"
                            Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD + self.env.now
                            Person_Dict[self.num]["Num_Crimes"] = Person_Dict[self.num]["Num_Crimes"] + 1
                            Person_Dict[self.num]["List_Times_ofCrime"].append(enter_time)
                            rand_number = MARI_gen.random()
                            if rand_number <= MARI_threshold and env.now > MARI_start_time:
                                Person_Dict[self.num]["List_Crime_ExitNext"].append("treatMARI")
                                Person_Dict[self.num]["List_Crime_ServiceTimes"].append(0.001)
                                Person_Dict[self.num]["List_Crime_ExitTimes"].append(enter_time+ 0.001)
                                Person_Dict[self.num]["nextTreatT"] = enter_time + 0.001
                                print('person %d has had an Arrest at time %f, and will go stright to treatment after a service time of %f at time %f.' % (self.num, enter_time, 0.001, Person_Dict[self.num]["nextTreatT"]))
                                #recaluclate service and relapse times as the individual is now in treatment
                                s_time = service_time('treatment') # Service time depending on interruption type, see fxn def 
                                r_time = relapse_time('treatment', Person_Dict[self.num]["arrivalT"], env.now, self.timeofNonOpioidDeath) # see fxn def
                                print('person %d is a %s at time %f and is in treatment. They will be in treatment for %f time and stop treatment at time %f. After treatment, their opioid use will remain inactive for %f time and will reactivate at time %f' % (self.num,self.getUserType(), enter_time,s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                                Person_Dict[self.num]["State"] = "Treatment"
                                Person_Dict[self.num]["Num_inTreatment"] = Person_Dict[self.num]["Num_inTreatment"] + 1
                                Person_Dict[self.num]["List_Times_inTreatment"].append(enter_time + 0.001)
                                Person_Dict[self.num]["List_Treat_ServiceTimes"].append(s_time)
                                Person_Dict[self.num]["List_Treat_ExitTimes"].append(enter_time+s_time)
                                Person_Dict[self.num]["List_InactiveTreat_ServiceTimes"].append(r_time)
                                self.sobriety_duration = s_time + r_time
                                Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                            else:
                                rand_num2 = CM_gen.random()
                                s_time = service_time(interrupt.cause) # Service time depending on interruption type, see fxn def 
                                if rand_num2 <= CM_threshold and (env.now + s_time) > CM_start_time:
                                    Person_Dict[self.num]["List_Crime_ExitNext"].append("treatCM")
                                    Person_Dict[self.num]["nextTreatT"] = enter_time + s_time + 0.001
                                    Person_Dict[self.num]["List_Crime_ServiceTimes"].append(s_time)
                                    Person_Dict[self.num]["List_Crime_ExitTimes"].append(enter_time+s_time)
                                    Person_Dict[self.num]["Next_Interrupt"] = 'treat' 
                                    self.sobriety_duration = s_time 
                                else:      
                                    Person_Dict[self.num]["List_Crime_ExitNext"].append("inactive")
                                    r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now, self.timeofNonOpioidDeath) # see fxn def 
                                    self.sobriety_duration = s_time + r_time
                                    Person_Dict[self.num]["List_Crime_ServiceTimes"].append(s_time)
                                    Person_Dict[self.num]["List_Crime_ExitTimes"].append(enter_time+s_time)
                                    Person_Dict[self.num]["List_InactiveCrime_ServiceTimes"].append(r_time)
                                    Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                                # print('person %d is a %s at time %f and has been arrested. They will serve %f time and be relased at time %f. After CJ release, their opioid use will remain inactive for %f time and will reactivate at time %f.' % (self.num,self.getUserType(), enter_time,s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                        # Hospital Encounter Interruption
                        elif interrupt.cause == 'hosp':
                            self.timeofFatalOD = 10000000 #cannot die from opioid related cause until relapse
                            s_time = service_time(interrupt.cause) # Service time depending on interruption type, see fxn def 
                            Person_Dict[self.num]["State"] = "hosp"
                            Person_Dict[self.num]["Num_Hosp"] = Person_Dict[self.num]["Num_Hosp"] + 1
                            Person_Dict[self.num]["List_Times_inHosp"].append(enter_time)
                            Person_Dict[self.num]["List_Hosp_ServiceTimes"].append(s_time) 
                            Person_Dict[self.num]["List_Hosp_ExitTimes"].append(enter_time+s_time)
                            rand_number = hosp_sub_gen.random()
                            if env.now > ED2R_start_time:
                                HE_thres = hospital_encounter_thres
                            else:
                                HE_thres = hospital_encounter_thres_base
                            # print('person %d has had a Hospital Encounter at time %f, and had a random number of %f.' % (self.num, enter_time, rand_number))
                            if rand_number < HE_thres[0]: # probability of getting arrested after hosp encounter
                                Person_Dict[self.num]["nextCrimeT"] = enter_time+ s_time + 0.001
                                Person_Dict[self.num]["List_Hosp_ExitNext"].append("crime")
                                Person_Dict[self.num]["Next_Interrupt"] = 'crime'
                                self.sobriety_duration = s_time
                                # print('person %d has had a Hospital Encounter at time %f, and will go stright to arrest after a service time of %f at time %f.' % (self.num, enter_time, s_time, Person_Dict[self.num]["nextCrimeT"])) 
                            elif rand_number < HE_thres[1]: # probability of starting treatment after hosp encounter
                                Person_Dict[self.num]["nextTreatT"] = enter_time + s_time + 0.001
                                Person_Dict[self.num]["Next_Interrupt"] = 'treat' 
                                Person_Dict[self.num]["List_Hosp_ExitNext"].append("treat")
                                self.sobriety_duration = s_time 
                                # print('person %d has had a Hospital Encounter at time %f, and will go stright to treatment after a service time of %f at time %f.' % (self.num, enter_time, s_time, Person_Dict[self.num]["nextTreatT"])) 
                            elif rand_number < HE_thres[2]:  # probability of dying of fatal overdose after hosp encounter
                                Person_Dict[self.num]["TimeofFatalOD"] = enter_time + s_time + 0.001
                                Person_Dict[self.num]["PrevState"] = "hosp"
                                Person_Dict[self.num]["List_Hosp_ExitNext"].append("fatal")
                                self.sobriety_duration = s_time 
                                # print('person %d has had a Hospital Encounter at time %f, and will die after a service time of %f at time %f.' % (self.num, enter_time, s_time, Person_Dict[self.num]["TimeofFatalOD"])) 
                            else:
                                Person_Dict[self.num]["List_Hosp_ExitNext"].append("inactive")
                                r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now, self.timeofNonOpioidDeath) # see fxn def 
                                self.sobriety_duration = s_time + r_time
                                Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + s_time + r_time)
                                Person_Dict[self.num]["List_InactiveHosp_ServiceTimes"].append(r_time)
                                # print('person %d has had a Hospital Encounter at time %f. They have a service time of %f and will be released at time %f. After Hospital release, their opioid use will remain inactive for %f time and will reactivate at time %f.' % (self.num, enter_time, s_time,(enter_time+s_time), r_time, (enter_time+s_time+r_time)))
                        # Inactive Interruption
                        elif interrupt.cause == 'inactive':
                            s_time = service_time(interrupt.cause) # Service time depending on interruption type, see fxn def 
                            r_time = relapse_time(interrupt.cause, Person_Dict[self.num]["arrivalT"], env.now, self.timeofNonOpioidDeath) # see fxn def 
                            self.sobriety_duration = r_time
                            self.timeofFatalOD = 10000000 #cannot die temporarily until relapse
                            Person_Dict[self.num]["State"] = "inactive"
                            Person_Dict[self.num]["TimeofFatalOD"] = self.timeofFatalOD + self.env.now
                            Person_Dict[self.num]["Num_Inactive_only"] = Person_Dict[self.num]["Num_Inactive_only"] + 1
                            Person_Dict[self.num]["List_Times_Inactive_only"].append(enter_time)
                            Person_Dict[self.num]["List_Relapse_Time"].append(enter_time + r_time)
                            Person_Dict[self.num]["List_InactiveOnly_ServiceTimes"].append(r_time)
                            # print('person %d is a %s at time %f and stoped opioid use. Their opioid use will remain inactive for %f time, and will reactivate at time %f.' % (self.num,self.getUserType(), enter_time,r_time,(enter_time+r_time)))
                        # Other Interruption
                        else:
                            print("unknown interuption cause: ",interrupt.cause)
                        
                        if self.timeofNonOpioidDeath < (self.env.now + self.sobriety_duration) and self.timeofNonOpioidDeath < days_per_year*num_years :
                            timeleft = self.timeofNonOpioidDeath - self.env.now
                            if self.timeofNonOpioidDeath < (self.env.now + s_time): #inactive has a service time of 0
                                if interrupt.cause == "crime":
                                    if Person_Dict[self.num]["List_Crime_ExitNext"][-1] == "treatMARI": #should be if
                                        Person_Dict[self.num]["PrevState"] = "treat"
                                        del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                        del Person_Dict[self.num]["List_Treat_ExitTimes"][-1] 
                                        Person_Dict[self.num]["List_Treat_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_inTreatment"][-1]
                                    elif Person_Dict[self.num]["List_Crime_ExitNext"][-1] == "treatCM":
                                        Person_Dict[self.num]["PrevState"] = "crime"
                                        del Person_Dict[self.num]["List_Crime_ExitTimes"][-1]
                                        Person_Dict[self.num]["List_Crime_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_ofCrime"][-1]
                                    else:    
                                        Person_Dict[self.num]["PrevState"] = "crime"
                                        del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                        del Person_Dict[self.num]["List_Crime_ExitTimes"][-1]
                                        del Person_Dict[self.num]['List_InactiveCrime_ServiceTimes'][-1]
                                        Person_Dict[self.num]["List_Crime_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_ofCrime"][-1]
                                elif interrupt.cause == "treatment":
                                    Person_Dict[self.num]["PrevState"] = "treat"
                                    del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                    del Person_Dict[self.num]["List_Treat_ExitTimes"][-1] 
                                    del Person_Dict[self.num]['List_InactiveTreat_ServiceTimes'][-1]
                                    Person_Dict[self.num]["List_Treat_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_inTreatment"][-1]
                                elif interrupt.cause == "hosp":
                                    Person_Dict[self.num]["PrevState"] = "hosp"
                                    del Person_Dict[self.num]["List_Hosp_ExitTimes"][-1]
                                    Person_Dict[self.num]["List_Hosp_ServiceTimes"][-1] =  self.timeofNonOpioidDeath - Person_Dict[self.num]["List_Times_inHosp"][-1]
                                    if Person_Dict[self.num]["List_Hosp_ExitNext"][-1] == "inactive":
                                        del Person_Dict[self.num]["List_Relapse_Time"][-1]
                                        del Person_Dict[self.num]['List_InactiveHosp_ServiceTimes'][-1]
                                    # elif Person_Dict[self.num]["List_Hosp_ExitNext"][-1] == "fatal": if someone has a comorbiity and dies from fatal even through they wer ein teh hpsial for opioid reason cause of death is non-opioid.
                                else:
                                    print("OTHER?!?!?")
                            else:
                                Person_Dict[self.num]["PrevState"] = "inactive"
                                del Person_Dict[self.num]["List_Relapse_Time"][-1]
                            timeleft = max(timeleft,0)
                            yield self.env.timeout(timeleft)
                            Person_Dict[self.num]["OpioidDeath"] = False
                            Person_Dict[self.num]["NonOpioidDeathT"] = self.env.now
                            Person_Dict[self.num]["isAlive"] = False
                            Person_Dict[self.num]["State"] = "death"
                            self.isAlive=False
                            # print('person %d has died at time %f.' % (self.num, self.env.now))
                        else:
                            # yield self.env.timeout(self.sobriety_duration)
                            yield self.env.timeout(self.sobriety_duration) #interupts process to go to death during sobriety.                    
                            Person_Dict[self.num]["State"] = "active"
                            if interrupt.cause == 'hosp':
                                if Person_Dict[self.num]["List_Hosp_ExitNext"][-1] == "fatal":
                                    Person_Dict[self.num]["OpioidDeath"] = True
                                    Person_Dict[self.num]["isAlive"] = False
                                    # print('person %d is %s at time %f.' % (self.num,self.getUserType(), self.env.now))
                                    Person_Dict[self.num]["OpioidDeathT"] = self.env.now
                                    Person_Dict[self.num]["State"] = "death"
                                    self.isAlive= False
                            if Person_Dict[self.num]["Next_Interrupt"] == '':
                                self.user_type = 'ru'
                                # print('person %d has become a %s and activated their disease at time %f.' % (self.num, self.getUserType(), self.env.now))
                    else:
                        sys.exit("Error: This person is already a recovering non-user")

        def setDead(self):
            time = min(self.timeofNonOpioidDeath,self.timeofFatalOD)
            time = max(time,0)
            yield self.env.timeout(time)
        
        def getUserType(self):
            if self.user_type == 'nru':
                return 'non-relapsed user'
            elif self.user_type == 'rnu':
                return 'recovering non-user'
            elif self.user_type == 'ru':
                return 'relapsed user'
            else:
                return 'deceased'
        def NextEvent(self, env, user, Person_Dict,i):
            if Person_Dict["Next_Interrupt"] == 'treat':
                Person_Dict["Next_Interrupt"] = ''
                user.action.interrupt('treatment')
                yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
                yield env.timeout(user.sobriety_duration) #interupts next event process 
            elif Person_Dict["Next_Interrupt"] == 'crime':
                Person_Dict["Next_Interrupt"] = ''
                user.action.interrupt('crime')
                yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
                yield env.timeout(user.sobriety_duration) #interupts next event process 
            elif Person_Dict["Next_Interrupt"] == 'hosp':
                Person_Dict["Next_Interrupt"] = ''
                user.action.interrupt('hosp')
                yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
                yield env.timeout(user.sobriety_duration) #interupts next event process 
            elif Person_Dict["Next_Interrupt"] == 'inactive':
                Person_Dict["Next_Interrupt"] = ''
                user.action.interrupt('inactive')       
                yield self.env.timeout(0.00001)   #allows sobriety_duration time to be calculated
                yield env.timeout(user.sobriety_duration) #interupts next event process  
            else:
                #Calculate all next possible event times
                Ctime = crime_time() #see fx def #
                Person_Dict["nextCrimeT"] = Ctime + env.now
                # print('person %d has a scheduled next Crime at time %f' % (i, Ctime+env.now))
                Ttime = treat_time() #see fxn def #
                Person_Dict["nextTreatT"] = Ttime + env.now
                # print('person %d has a scheduled next treatment at time %f' % (i, Ttime+env.now))
                Htime = hosp_time() #see fxn def #
                Person_Dict["nextHospT"] = Htime + env.now
                # print('person %d has a scheduled next Hospital Encounter at time %f' % (i, Htime+env.now))
                Itime = inactive_time() 
                Person_Dict["NextInactiveT"] = Itime + env.now
                # print('person %d has a scheduled next Inactive Stage at time %f' % (i, Itime+env.now))
                Dtime = user.timeofNonOpioidDeath - env.now
                ODtime = user.timeofFatalOD - env.now
                #select the minimum time to be the next event
                time = min(Ctime,Ttime,Htime,Itime, ODtime,Dtime)            
                # get index of smallest item in list
                list_times = (Ctime,Ttime,Htime,Itime,ODtime,Dtime)
                X = list_times.index(min(list_times))
                Person_Dict["NextEventType"] = X
                Person_Dict["TimeUnitNextEvent"] = time
                # print('person %d has a scheduled next Event at time %f of Type %f' % (i, time+env.now, X))
                time = max(time,0)
                yield self.env.timeout(time)
                # if user.user_type == 'nru' or user.user_type=='ru' and user.isAlive:
                if X == 0:
                    user.action.interrupt('crime')        
                    # print('person %d is now being arrested at time %f of Type %f' % (i, env.now, X))            
                elif X == 1:
                    user.action.interrupt('treatment')
                    # print('person %d is now starting treatment at time %f of Type %f' % (i, env.now, X))
                elif X == 2:
                    user.action.interrupt('hosp')
                    # print('person %d is now going to hosp at time %f of Type %f' % (i, env.now, X))
                elif X == 3:
                    user.action.interrupt('inactive')
                    # print('person %d in now inactive at time %f of Type %f' % (i, env.now, X))

    
    def user_arrivals(env,i):
        #Create new users until the sim end time is reached
        while True:
            time = arrival_time() #see fxn def # 
            #print("time until next arrival", expo)
            Person_Dict[i] = {}
            Person_Dict[i]["arrivalT"] = time + env.now
            time = max(time,0)
            yield env.timeout(time)
            #can add conditions to delay person arrival or assign them to services / groups see  #https://simpy.readthedocs.io/en/latest/examples/movie_renege.html?highlight=arrivals#movie-renege,
            Person[i] = User(env,i)
            i = i + 1


    
    ### RUNNING THE SIMULAITON ####
    #set up adn start the simulation 
    env = simpy.Environment()
    #trace(env, monitor)

    #intitial population in simualtion
    intital_done = False
    ''' Average number of people in use'''
    starting_probs= []
    starting_probs.append(arrival_gen.triangular(27298.81,34224.21,43260.59)) #number of indibiuals in starting population
    starting_probs.append(crime_gen.triangular(15,25,50) / starting_probs[0])
    starting_probs.append(hosp_gen.triangular(5,11,15) / starting_probs[0])
    starting_probs.append(treat_gen.triangular(300,450,500) / starting_probs[0])
    starting_probs.append((arrival_gen.triangular(starting_probs[0]/5, 2*(starting_probs[0]/5),4*(starting_probs[0]/5)))/starting_probs[0])
    for i in range(0, math.floor(starting_probs[0])):
        start_RN = start_gen.random()
        Person_Dict[i] = {}
        Person[i] = User(env,i)
        Person_Dict[i]["arrivalT"] = 0
        #test B lowered percent starting in arrest adn decrease inactive state
        if start_RN <    starting_probs[1]: #arrest
            Person_Dict[i]["Next_Interrupt"] = "crime"
        elif start_RN <    (starting_probs[2] + starting_probs[1]): #hospital
            Person_Dict[i]["Next_Interrupt"] = "hosp"
        elif start_RN <    (starting_probs[3]+ starting_probs[2] + starting_probs[1]): #treatment
            Person_Dict[i]["Next_Interrupt"] = "treat"
        elif start_RN <   (starting_probs[4]+ starting_probs[3] + starting_probs[2] + starting_probs[1]): # active
            pass #as this is how one typically enters the simulation
        else: #inactive 
            Person_Dict[i]["Next_Interrupt"] = "inactive"
    intital_done = True
    sys.stdout = original_stdout
    print("--------------------  Scenario "+ str(s) + " Simulation Start ------------------------") 
    i = i + 1
    write_file = open("Results/Test_SimulationsMovement.txt", "w+")
    sys.stdout = write_file
    #start process and run
    env.process(user_arrivals(env,i))
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
    print(Person_Dict[i])

    ############## Initalizing ##################
    
    #Totals 
    Total_newUsers_run = {}
    Total_InactiveUsers_run = {}
    Total_ActiveUsers_run = {}
    Total_Relapses_run = {}
    Total_OD_Deaths_run = {}
    Total_nOD_Deaths_run = {}
    Total_Crimes_run = {}
    Total_Crimes_uniq_run = {}
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
        Total_Crimes_run[year] = 0
        Total_Crimes_uniq_run[year] = set()
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
    crimes_list = []
    hosp_list = [] 
    treatments_list = []
    ####### Utilization arrays ######
    #monthly - current simulation state at the start of the month
    months = 12*num_years
    arrivals_ut = [0]*months
    hosp_ut = [0]*months
    crime_ut = [0]*months
    treat_ut = [0]*months
    ODdeath_ut = [0]*months
    nOD_death_ut = [0]*months
    active_ut = [0]*months
    inactive_ut = [0]*months
    total_indivs = [0]*months

    ######## start of list creation and array counts ###### 
    ## might need a minsu 1 on the ceilings so months and year counts match up, ok since max month isnt done via range funciton
    print("..Perons with errors..")
    for d,v in Person_Dict.items():
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
                    if v["PrevState"] == "crime":
                        crime_ut[i] -= 1
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
            for n in v["List_Times_ofCrime"]:
                crimes_list.append(n)
                year = math.floor(n / days_per_year)
                if year < num_years:
                    Total_Crimes_uniq_run[year].add(d) # If the element already exists, the add() method does not add the element.
                    Total_Crimes_run[year] =  Total_Crimes_run[year]+1
                for i in range(math.floor(n/days_per_month),months):
                    active_ut[i] -= 1
                    crime_ut[i] += 1
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
                        crime_ut[i] -= 1
                        inactive_ut[i] +=1
                elif v["List_Crime_ExitNext"][idx] == "treatCM":
                    for i in range(math.floor(n/days_per_month),months):
                        crime_ut[i] -= 1
                        active_ut[i] +=1 #since going to treat next
                    year = math.floor(n/days_per_year)
                    for y in range(year+1,num_years): #add one to all future years to offset going to treat next
                        prev_count[y] += 1
                elif v["List_Crime_ExitNext"][idx] == "treatMARI":
                    for i in range(math.floor(n/days_per_month),months):
                        crime_ut[i] -= 1
                        active_ut[i] +=1 #since going to treat next
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
        print_histogram(crimes_list,(int(months)),s,"Time of Crime (Months)","Number of Crimes","Crimes_Month") # number of crimes in each month
        print_histogram(hosp_list,(int(months)),s,"Time of Hospital Encounter (Months)","Number of Hospital Encounters","Hospital_Month") #number of Hospital Encounters in each month ####
        print_histogram(OD_deaths_list,(int(months)),s,"Time of Death (Months)","Number of Deaths","Deaths_Month") #number of deaths in each month ###
        #'''
        #Yearly
        # print(OD_deaths_list)
        print_histogram(arrivals_list,(int(num_years)),s,"Time of Arrival (Years)","Number of Arrivals","Arrivals_Year")  #new user arrivals in each month
        print_histogram(treatments_list,(int(num_years)),s,"Time of Treatment Start (Years)","Number of Treatment Starts","Treatment_Year") #number of treatment starts in each num_year
        print_histogram(crimes_list,(int(num_years)),s,"Time of Crime (Years)","Number of Crimes","Crimes_Year") # number of crimes in each num_year
        print_histogram(hosp_list,(int(num_years)),s,"Time of Hospital Encounter (Years)","NUmber of Hospital Encounters","Hospital_Year") #number of Hospital Encounters in each num_year ####
        print_histogram(OD_deaths_list,(int(num_years)),s,"Time of Death (Years)","Number of Opioid-Related Deaths","OD_Deaths_Year") #number of Opioid deaths in each num_year ###
        print_histogram(nOD_death_list,(int(num_years)),s,"Time of Death (Years)","Number of nonOpioid-Related Deaths","nonOD_Deaths_Year") #number of non-Opioid Related deaths in each num_year ###
        print_histogram(relapse_list,(int(num_years)),s,"Time of Relapse (Years)","Number of Individuals","Relapse_Year") #number of non-Opioid Related deaths in each num_year ###
        #### Hisograms of Utilization per month ###########
        print_barChart(active_ut,s,"Time","Number of OUD Active Individuals","Ut_Active", num_years) 
        print_barChart(inactive_ut,s,"Time","Number of OUD Inactive Individuals","Ut_Inactive", num_years) 
        print_barChart(treat_ut,s,"Time","Number of Individuals in Treatment","Ut_Treatment", num_years) 
        print_barChart(crime_ut,s,"Time","Number of Individuals in Criminal Justice System","Ut_Crimes", num_years) 
        print_barChart(hosp_ut,s,"Time","Number of Individuals in Hospital","Ut_Hospital", num_years)
        print_barChart(ODdeath_ut,s,"Time","Number of Deceased Individuals from Opioid-Related Causes","Ut_OD_Deaths", num_years) 
        print_barChart(nOD_death_ut,s,"Time","Number of Deceased Individuals from Non-Opioid-Related Causes","Ut_nOD_Deaths", num_years) 
        
        plt.plot(range(0,int(months)),active_ut,marker=".", color="k")
        plt.plot(range(0,int(months)),crime_ut,marker=".", color="r")
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
    print("Number in Arrest state in final month:", crime_ut[months-1])
    print("Number in Hospital state in final month:", hosp_ut[months-1])
    print("Number in Treatment State in final month:", treat_ut[months-1])
    print("Number of arrivals in final month:", arrivals_ut[months-1])
    #Monthly Totals   
    print("Monthly Number in active state:", active_ut)
    print("Monthly Number in inactive state:", inactive_ut)
    print("Monthly Number in opioid death state:", ODdeath_ut)
    print("Monthly Number in non-opioid death state:", nOD_death_ut)
    print("Monthly Number in Arrest state:", crime_ut)
    print("Monthly Number in Hospital state:", hosp_ut)
    print("Monthly Number in Treatment State:", treat_ut)
    print("Monthly Number of arrivals:", arrivals_ut)
    # Total of yearly events
    print("---------------------- Total Yearly Events ----------------------------")   
    print("Number of new user arrivals each year:, ", Total_newUsers_run)
    print("Total number of Opioid Deaths:, ", Total_OD_Deaths_run)
    print("Total number of Non-Opioid Deaths:, ", Total_nOD_Deaths_run)
    print("Total number of Arrests:, ", Total_Crimes_run)
    print("Total number of Individuals Arrested:, ", [len(Total_Crimes_uniq_run[year]) for year in range(num_years)])
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
    df_times = df_times.append(df, ignore_index=True)
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
    df_times = df_times.append(df, ignore_index=True)
    
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
    df_times = df_times.append(df, ignore_index=True)

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
    df_times = df_times.append(df, ignore_index=True)

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
    df_times = df_times.append(df, ignore_index=True)

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
    df_times = df_times.append(df, ignore_index=True)

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
    df_times = df_times.append(df, ignore_index=True)

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
    df_times = df_times.append(df, ignore_index=True)

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
    df_times = df_times.append(df, ignore_index=True)

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
    df_times = df_times.append(df, ignore_index=True)

    print("----------- Time Summary DF ---------------")
    #print(df_times.loc[9*s:(9*s)+9,:].to_string())
    print(df_times.to_string())
    
    sys.stdout = original_stdout
    write_file.close()
    print("---- Finished Sumary of Scenario ", s,"-----")
    return Total_newUsers_run,Total_Relapses_run, Total_ActiveUsers_run, Total_Prev_run, Total_OD_Deaths_run, Total_nOD_Deaths_run, Total_Crimes_run, Total_Crimes_uniq_run, Total_Treatments_run, Total_Treatments_uniq_run, Total_Hosp_run, Total_Hosp_uniq_run, Total_InactiveUsers_run,df_times 
    # breakpoint()
    
if __name__ == '__main__':
    os.makedirs('Results', exist_ok=True)
    # original_stdout = sys.stdout
    params = []
    import DataCleaningDCsim_NEW as inputs
    ### arrival rate ######
    lam_user_arrival = inputs.init_rate #Rate of initiation, i.e. Number of person that use opioids for the first time per day.
    
    # breakpoint()
    '''
    A variable follows a Poisson distribution if the following conditions are met:
        Data are counts of events (nonnegative integers with no upper bound).
        All events are independent. (Note: deaths are likely correlated with "bad batches" )
        Average rate does not change over the period of interest. (Note: death and initiation rates do change over time...)
    '''
    ################################################ Used for Calibration #################################################
    ''''
    Event times estimated in LNdist_Double Check.xlsx
    Mu chosen using data sources, sigma chosen using Law 2008
    '''
    #ARC 2
    # LNmean_deathdays = 12.32844619 #number currently in Table 1
    # LNmean_deathdays = 12.48367713 #number currently in Excel and Sensitivity Table
    # ODdeathdays_est = {"l":0, "m":1, "mu":LNmean_deathdays}
    # LNsigma_deathdays = ln_est_sig_only(ODdeathdays_est["l"],ODdeathdays_est["m"],ODdeathdays_est["mu"])
    #Current baseline 60922_030838
    # ODdeathdays_est = {"l":0, "m":1, "q":0.002425914, "x":365.25} #<- q is average yearly OD deaths / average yealry prevalence est
    
    #102222 results
    ODdeathdays_est = []
    # ODdeathdays_est.append({"l":0, "m":1, "q":0.002386761, "x":365.25}) #<- q is average yearly OD deaths / average yealry prevalence est
    # ODdeathdays_est.append({"l":0, "m":1, "q":0.002771925, "x":365.25}) #<- q is average yearly OD deaths / average yealry prevalence est  
    #Current baseline 102822_030838 RESULTS
    ODdeathdays_est.append({"l":0, "m":1, "q":0.002651957, "x":365.25}) #<- q is average yearly OD deaths / average yealry prevalence est
    ODdeathdays_est.append({"l":0, "m":1, "q":0.003326891, "x":365.25}) #<- q is average yearly OD deaths / average yealry prevalence est   
    LNmean_deathdays = []
    LNsigma_deathdays = []
    for idx, i in enumerate(ODdeathdays_est):
        mu, sig =  ln_est(ODdeathdays_est[idx]["l"],ODdeathdays_est[idx]["m"],ODdeathdays_est[idx]["q"],ODdeathdays_est[idx]["x"])
        LNmean_deathdays.append(mu)
        LNsigma_deathdays.append(sig)
    #OLD
    # ODdeathdays_est = {"l":0, "m":1, "q":0.001169358, "x":7} # <- q is average OD deaths / starting population. 
    # LNmean_deathdays, LNsigma_deathdays =  ln_est(ODdeathdays_est["l"],ODdeathdays_est["m"],ODdeathdays_est["q"],ODdeathdays_est["x"])

    #ARC 3
    LNmean_hospdays = 9.072755796855650 #<- only last twpo years, all four -> 8.9407719434103400
    hospdays_est = {"l":0, "m":1, "mu":LNmean_hospdays}
    LNsigma_hospdays = ln_est_sig_only(hospdays_est["l"],hospdays_est["m"],hospdays_est["mu"])
    #OLD
    # hospdays_est = {"l":0, "m":1, "q":0.04753527, "x":365.25} #<- q is average  hosps / starting population. 
    # LNmean_hospdays, LNsigma_hospdays =  ln_est(hospdays_est["l"],hospdays_est["m"],hospdays_est["q"],hospdays_est["x"])
    
    #ARC 4
    LNmean_arrestdays = 10.0436515056944000
    arrestdays_est = {"l":0, "m":90, "mu":LNmean_arrestdays}
    LNsigma_arrestdays = ln_est_sig_only(arrestdays_est["l"],arrestdays_est["m"],arrestdays_est["mu"])
    # LNsigma_arrestdays = 3.16917205365919000 #-.75 #sig_c: -1 sig_b & sig_a: -1.25
    # arrestdays_est = {"l":0, "m":1, "q":0.015451051, "x":365.25} #<- q is average  arrests / starting population. 
    # LNmean_arrestdays, LNsigma_arrestdays =  ln_est(arrestdays_est["l"],arrestdays_est["m"],arrestdays_est["q"],arrestdays_est["x"])
    
    #ARC 5
    #Numbers in Excel document
    # LNmean_treatdays = 8.6869256502269700
    # treatdays_est = {"l":0, "m":30, "mu":LNmean_treatdays}
    # LNsigma_treatdays = ln_est_sig_only(treatdays_est["l"],treatdays_est["m"],treatdays_est["mu"])

    #Current Baseline 60922_030838
    treatdays_est = {"l":0, "m":630, "q":0.060289179, "x":365.25} #<- q is average yearly treats / average yealry prevalence est 
    LNmean_treatdays, LNsigma_treatdays =  ln_est(treatdays_est["l"],treatdays_est["m"],treatdays_est["q"],treatdays_est["x"])
    
    
    #ARCS 6-8
    #after a hostpial encounter: prob get arrested, prob start treatment, prob die of fatal overdose:
    #https://www.health.state.mn.us/communities/opioids/prevention/followup.html --> citing study https://pubmed.ncbi.nlm.nih.gov/31229387/
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5802515/ 7.3%-9.81 % of ICU Overdse patients died
    #https://pubmed.ncbi.nlm.nih.gov/30198926/ nationally 4.7% to 2.3% of indviduals with IOD and POD died after admission
    #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229174 larger study 2.36% all 2.18% in Midwest  died during hospitalization.23.66% all and 22.27% midwest discharged to inpatent 
    process = sys.argv[3] #process is the index row we use for our scenario
    with open("AllScenarioMatrix.csv", 'r', encoding='utf-8-sig') as fp:  #The first column in the ED2Recovery Numbers, the second is the MARI numbers
        Matrix = fp.readlines()
    fp.close()
    Matrixrow = Matrix[int(process)].split(",")
    MatrixVal = float(Matrixrow[0])
    str_MatrixVal = MatrixVal*100000 #are in large values so that the file name wont have a decimal
    #arrest is 0.01, death is 0.0318, treatment is between (baseline:0.2227, max:0.9582)
    arrest_per = 0.01
    death_per = 0.0218
    # death_per = 0.03540161 #test not from past results
    baseline_treat_per = 0.2227
    ED2R_start_year = 2017
    hospital_encounter_thres_base = {0:arrest_per, 1:(baseline_treat_per+arrest_per), 2:(baseline_treat_per+arrest_per+death_per)}
    hospital_encounter_thres = {0:arrest_per, 1:(MatrixVal+arrest_per), 2:(MatrixVal+arrest_per+death_per)} #arrest, treatment, death

    #ARC 9
    #Current Baseline 60922_030838
    iadays_est = {"l":0, "m":1, "q":0.494, "x":120}
    LNmean_iadays, LNsig_iadays =  ln_est(iadays_est["l"],iadays_est["m"],iadays_est["q"],iadays_est["x"])
    #Current Excel Spreedsheet Nubmers
    # iadays_est = {"l":0, "m":1, "q":0.688, "x":120}
    # LNmean_iadays, LNsig_iadays =  ln_est(iadays_est["l"],iadays_est["m"],iadays_est["q"],iadays_est["x"])


    #ARC 10 - now calcualted based on entering age. adjust enter age probabilities here
    dup_prev_age_mean, dup_prev_age_sig = ln_est(12,30,.75,42.4)
    dup_init_age_mean, dup_init_age_sig = ln_est(12,25,.75,37.9)
    # DC_agePer = [.1400,.2094,.1651,.1340,.1363,.1161,.0640,.0351]
    # DC_agegroup = [10,20,30,40,50,60,70,80]
    ''''
    Service times estimated from fuction ln_est from LAW 2008
    '''
    #ARC A
    #amount of time in hospital
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5802515/
    #https://journals-sagepub-com.ezproxy.library.wisc.edu/doi/full/10.1177/8755122519860081 n=101 in Ohio. 49/101 were addimitted and had a average LOS of 4.39 (range=0-22). average 1.91 days in ICU and 2.48 in general floor
    #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229174 larger study mean 3.6 days, 73% <=3 days all 3.2 days mean 1.8 median and 72.3% <=3 in Midwest
    #time = service_gen.triangular(.1,25,1.8)
    #time = service_gen.lognormvariate(.5,1.2) #using eyeball estimate
    hospservice_est = {"l":0, "m":1.8, "q":0.723, "x":3}
    LNmean_hospservice, LNsig_hospservice = ln_est(hospservice_est["l"],hospservice_est["m"],hospservice_est["q"],hospservice_est["x"])
    #ARC B
    #estimate of crime service time
    #low level crime - MARI jail data HC median&mode is 1 average 20 max is 321, 90% in jail < 57 days after 1st offense
    #time = service_gen.triangular(.1,365.250,1)
    #time = service_gen.lognormvariate(1,2) #sd decided to get an E(x) around 20 days and mode of 1 day
    crimeservice_est = {"l":0, "m":1, "q":0.9, "x":57}
    LNmean_crimeservice, LNsig_crimeservice = ln_est(crimeservice_est["l"],crimeservice_est["m"],crimeservice_est["q"],crimeservice_est["x"])
    #ARC C
    #estimate of treatmetn service time
    #table 51: from 2015 Wisconsin DHS report. Source Program Participation System
    #51.3% in dane county completed SUD treatment, 59.5% recvied at least 90 days of treatment
    #try triangular with mean a 120 and max of 10 years
    #time = service_gen.lognormvariate(4.75,.9) #eyball decided based on 40% prob less than 90
    treatservice_est = {"l":0, "m":30, "q":0.405, "x":90}
    LNmean_treatservice, LNsig_treatservice = ln_est(treatservice_est["l"],treatservice_est["m"],treatservice_est["q"],treatservice_est["x"])
    ''''
    Relapse times estimated from fuction ln_est from LAW 2008
    '''
    #ARC D
    # crimerel_est = {"l":0, "m":2, "q":48/62, "x":90}
    crimerel_est = {"l":0, "m":2, "q":48/62, "x":90}
    LNmean_crimerel, LNsig_crimerel = ln_est(crimerel_est["l"],crimerel_est["m"],crimerel_est["q"],crimerel_est["x"]) #kinlicock study
    #ARC E
    treatrel_est = {"l":0, "m":28, "q":226/308, "x":182}
    LNmean_treatrel, LNsig_treatrel = ln_est(treatrel_est["l"],treatrel_est["m"],treatrel_est["q"],treatrel_est["x"]) # nunes
    #ARC F
    hosprel_est = {"l":0, "m":1, "q":0.85, "x":30} #chutuape 2001
    LNmean_hosprel, LNsig_hosprel = ln_est(hosprel_est["l"],hosprel_est["m"],hosprel_est["q"],hosprel_est["x"]) 
    # ARC G
    # iarel_est = {"l":0, "m":7, "q":0.2, "x":120} #10-12-22 BASELINE
    iarel_est = {"l":0, "m":1, "q":0.7, "x":20125.275/10} # 10/28/22 RUN
    #90 percent use within their lifetime? i.e. estimated death range #test 1: 90% lifetime day 30, #test 2: 90% lifetime day 1,#test 3: 90% 1/2 lifetime day 1,#test 4: 90% 1/3 lifetime day 1
    #test 5: 7days with 60% within a quarter of lifetime
    #test 6: 7 days 20% within 120 days
    LNmean_iarel,LNsig_iarel =  ln_est(iarel_est["l"],iarel_est["m"],iarel_est["q"],iarel_est["x"]) 

    #### MARI Threshold ######
    MARI_thres = float(Matrixrow[1]) 
    str_MARIVal = MARI_thres*100 #are in large values so that the file name wont have a decimal
    MARI_start_year = 2017
    ##### Case Management Threshold #####
    CM_thres = float(Matrixrow[2])
    str_CMVal = CM_thres*100 #are in large values so that the file name wont have a decimal
    CM_start_year = 2023

    #### time inputs
    n_runs = int(sys.argv[1])
    num_years = int(sys.argv[2]) + warmup
    days_per_year = 365.25
    days_per_month = 365.25/12
    start_year = 2013 - warmup

    params = {}
    params["lam_user_arrival"] = lam_user_arrival
    params["LNmean_deathdays"] = LNmean_deathdays
    params["LNsigma_deathdays"] = LNsigma_deathdays
    params["LNmean_hospdays"] = LNmean_hospdays
    params["LNsigma_hospdays"] = LNsigma_hospdays
    params["LNmean_arrestdays"] = LNmean_arrestdays
    params["LNsigma_arrestdays"] = LNsigma_arrestdays
    params["LNmean_treatdays"] = LNmean_treatdays
    params["LNsigma_treatdays"] = LNsigma_treatdays
    params["hospital_encounter_thres"] = hospital_encounter_thres
    params["LNmean_iadays"] = LNmean_iadays
    params["LNsig_iadays"] = LNsig_iadays
    params["dup_prev_age_mean"] = dup_prev_age_mean
    params["dup_prev_age_sig"] = dup_prev_age_sig
    params["dup_init_age_mean"] = dup_init_age_mean
    params["dup_init_age_sig"] = dup_init_age_sig
    params["LNmean_hospservice"] = LNmean_hospservice
    params["LNsig_hospservice"] = LNsig_hospservice
    params["LNmean_crimeservice"] = LNmean_crimeservice
    params["LNsig_crimeservice"] = LNsig_crimeservice
    params["LNmean_treatservice"] = LNmean_treatservice
    params["LNsig_treatservice"] = LNsig_treatservice
    params["LNmean_crimerel"] = LNmean_crimerel
    params["LNsig_crimerel"] = LNsig_crimerel
    params["LNmean_treatrel"] = LNmean_treatrel
    params["LNsig_treatrel"] = LNsig_treatrel
    params["LNmean_hosprel"] = LNmean_hosprel
    params["LNsig_hosprel"] = LNsig_hosprel
    params["LNmean_iarel"] = LNmean_iarel
    params["LNsig_iarel"] = LNsig_iarel
    params["HE_thres_baseline"] = hospital_encounter_thres_base
    params["ED2R_start_time"] = (ED2R_start_year - start_year) * 365.25
    params["MARI_thres"] = MARI_thres
    params["MARI_start_time"] = (MARI_start_year - start_year) * 365.25
    params["CM_thres"] = CM_thres
    params["CM_start_time"] = (CM_start_year - start_year) * 365.25
    #################### generate event times from LN distribution ###############################

    #################### set up enmpty lists ###############################
    #Total Numbers
    Total_OD_Deaths = {}
    Total_nOD_Deaths = {}
    Total_Crimes = {}
    Total_Crimes_uniq = {} 
    Total_Treatments = {}
    Total_Treatments_uniq = {}
    Total_Hosp = {}
    Total_Hosp_uniq = {}
    Full_Per_Dic = {} #full dictionaly of all simulation runs
    Total_newUsers = {}
    Total_Relapse = {}
    Total_InactiveUsers = {}
    Total_ActiveUsers = {}
    Total_Prev = {}
    Mean_enter_Age = {}

    #dataframes
    df_times = pd.DataFrame()
########################################## Ru,,02n for n_runs ######################################################################

    
    seeds = {0: {'arrival': 35484, "death": 568, "relapse":25677, "crime":8, "treat":103, "hosp": 6944, "hosp_sub": 6935, "service": 53215, "inactive": 234235, "alldeath" : 9289, "start" : 9274, "MARI": 321548, "CM": 226948} }

    for s in range(n_runs):
        seeds[s+1] ={}
        for t in seeds[s]:
            seeds[s+1][t] = ((seeds[s][t]*5) + (s+1)) % 100000   

    if n_runs <= 8:
        CPUs = n_runs
    else:
        CPUs = 8

    debug = False
    if debug:
        results = []
        for s in range(n_runs):
            results.append(simulation_run([s, n_runs, num_years, days_per_year, days_per_month, start_year, seeds, params]))
    else:
        pool = multiprocessing.Pool(CPUs) #number of CPUs requested
        args = []
        for s in range(n_runs):
            args.append([s, n_runs, num_years, days_per_year, days_per_month, start_year, seeds, params])
        results = pool.map(simulation_run, args)

    for index, s in enumerate(results):
        Total_newUsers[index] = s[0]
        Total_Relapse[index] = s[1]
        Total_ActiveUsers[index] = s[2]
        Total_Prev[index] = s[3]
        Total_OD_Deaths[index] = s[4]
        Total_nOD_Deaths[index] = s[5]
        Total_Crimes[index] = s[6]
        Total_Crimes_uniq[index] = s[7]
        Total_Treatments[index] = s[8]
        Total_Treatments_uniq[index] = s[9]
        Total_Hosp[index] = s[10]
        Total_Hosp_uniq[index] = s[11]
        Total_InactiveUsers[index] = s[12]
        s[13].insert(0, "Scenerio", index, True)
        df_times = pd.concat([df_times, s[13]], ignore_index=True, sort=False)

    sys.stdout = original_stdout#FullSim_file    
    endtime = ti.time()
    print ('total time elapsed:', endtime - starttime)
    print("...Simulation Finished. Starting Summary Statistics...")

    write_file2 = open("Results/Test_SimulationsStatsSUMMARY_"+str(n_runs)+"Runs.txt", "w+")
    sys.stdout = write_file2
    print("-----------------------Seeds used --------------------------------------")
    print(seeds)
    print("----------------------- Parameters used --------------------------------------")
    print("Warm-up period: ", warmup, " years")
    print("Initiation Age: mu = %f, sigma = %f " %(dup_init_age_mean, dup_init_age_sig))
    print("Prevalence Age: mu = %f, sigma = %f " %(dup_prev_age_mean, dup_prev_age_sig))
    # print("Starting Population: Low = %f, Median= %f, High= %f", start_pop)
    print("ARC 1 Arrival Rate: lambda = ", lam_user_arrival)
    print("ARC 2 Fatal Overdose Next Event Time: mu_pre2019 = %f, sigma_pre_2019 = %f,  mu_post_2019 = %f, sigma_post_2019 = %f " %(LNmean_deathdays[0], LNsigma_deathdays[0],LNmean_deathdays[1], LNsigma_deathdays[1]))
    print(ODdeathdays_est)
    print("ARC 3 Hospital Encounter Next Event Time: mu = %f, sigma = %f " %(LNmean_hospdays, LNsigma_hospdays))
    print(hospdays_est)
    print("ARC 4 Opioid-Related Arrest Next Event Time: mu = %f, sigma = %f " %(LNmean_arrestdays, LNsigma_arrestdays))
    print(arrestdays_est)
    print("ARC 5 Treatment Next Event Time: mu = %f, sigma = %f " %(LNmean_treatdays, LNsigma_treatdays))
    print(treatdays_est)
    print("ARC 6 HE to Fatal Next Event Time: Probability = %f " %(hospital_encounter_thres[2]-hospital_encounter_thres[1]))
    print("ARC 7 HE to Arrest Next Event Time: Probability =%f " %(hospital_encounter_thres[0]))
    print("ARC 8 HE to Treatment Next Event Time: Probability =%f " %(hospital_encounter_thres[1] - hospital_encounter_thres[0]))
    print("ARC 9 Inactive Next Event Time: mu = %f, sigma = %f " %(LNmean_iadays, LNsig_iadays))
    print(iadays_est)
    # print("ARC 10 All Death Next Event Rate:, alpha = %f , beta = %f, b = %f" %(Beta1_NODdays,Beta2_NODdays,Beta_maxb))
    print("ARC A Hospital Encounter Service Time: mu = %f, sigma = %f with," %(LNmean_hospservice, LNsig_hospservice))
    print(hospservice_est)
    print("ARC B Arrest Service Time: mu = %f, sigma = %f " %(LNmean_crimeservice, LNsig_crimeservice))
    print(crimeservice_est)
    print("ARC C Treatment Service Time: mu = %f, sigma = %f " %(LNmean_treatservice, LNsig_treatservice))
    print(treatservice_est)
    print("ARC D Inactive Service time from Arrest: mu = %f, sigma = %f " %(LNmean_crimerel, LNsig_crimerel))
    print(crimerel_est)
    print("ARC E Inactive Service time from Treatment: mu = %f, sigma = %f " %(LNmean_treatrel, LNsig_treatrel))
    print(treatrel_est)
    print("ARC F Inactive Service time from HE: mu = %f, sigma = %f " %(LNmean_hosprel, LNsig_hosprel))
    print(hosprel_est)
    print("ARC G Inactive Service time from Active: mu = %f, sigma = %f " %(LNmean_iarel, LNsig_iarel))
    print(iarel_est)
    
    ################################## Summary output ########################################
    print("----------------------- Simulation Runs Summary ------------------------")
    print("----------------------- Users ------------------------")
    # print("Avg number of Users over 6 years: ", np.mean([len(Full_Per_Dic[s]) for s in range(n_runs)]))
    dict_mu, dict_max, dict_min = dict_mean(Total_newUsers, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_newUsers,num_years), inputs.df_initiation,"Dane County Opioid Initiation estimate (number of people)","Arrival")
    print("Avg number of Opioid Initations: ", dict_mu)
    print("95\% PI number of Opioid Initations: ", dict_min, dict_max)
    dict_mu, dict_max, dict_min = dict_mean(Total_ActiveUsers, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_ActiveUsers,num_years), None, "Number of Active Users at the end of the year","Active")
    print("Avg number of Individuals with Active Use: ", dict_mu)
    print("95\% PI number of Individuals with Active Use: ", dict_min, dict_max)
    dict_mu, dict_max, dict_min = dict_mean(Total_InactiveUsers, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_InactiveUsers,num_years), None, "Number of Active Users at the end of the year","Inactive")
    print("Avg number of Individuals with Inactive Use: ", dict_mu)
    print("95\% PI number of Individuals with Inactive Use: ", dict_min, dict_max)
    
    print("----------------------- Opioid Deaths ------------------------")
    dict_mu, dict_max, dict_min=dict_mean(Total_OD_Deaths, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_OD_Deaths,num_years), inputs.df_DCdeaths,'Deaths',"OR_Death")
    print("Avg number of Opioid Deaths: ", dict_mu)
    print("95\% PI number of Opioid Deaths: ", dict_min, dict_max)
    # print(Total_OD_Deaths)
    print("----------------------- non-Opioid Deaths ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_nOD_Deaths, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_nOD_Deaths,num_years), None,'Non-Opioid-Related Deaths',"nOR_Death")
    print("Avg number of non-Opioid Deaths: ", dict_mu )
    print("95\% PI number of non-Opioid Deaths: ", dict_min, dict_max)
    # print(Total_nOD_Deaths)
    print("----------------------- Crimes ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Crimes, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_Crimes,num_years), inputs.df_Yarrests,'ArrestCount',"Arrests")
    print("Avg number of Arrests: ", dict_mu)
    print("95\% PI number of Arrests: ", dict_min, dict_max)
    # print(Total_Crimes)
    dict_mu, dict_max, dict_min = dict_set_mean(Total_Crimes_uniq, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_set_sd(Total_Crimes_uniq,num_years), None,'Unique Individuals Arrested',"Indv_Arrested")
    print("Avg number of Individuals Arrested: ",  dict_mu )
    print("95\% PI number of Individuals Arrested: ", dict_min, dict_max)
    # print(Total_Crimes_uniq)
    print("----------------------- Treatment ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Treatments, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_Treatments,num_years), None,'Treatment Starts',"Treatments")
    print("Avg number of Treatment Starting Episodes: ",  dict_mu )
    print("95\% PI number of Treatment Starting Episodes: ", dict_min, dict_max)
    # print(Total_Treatments)
    dict_mu, dict_max, dict_min = dict_set_mean(Total_Treatments_uniq,num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_set_sd(Total_Treatments_uniq,num_years), inputs.df_treat,'Total Individuals',"Indv_Treated")
    print("Avg number of Individuals sent to Treatment: ", dict_set_mean(Total_Treatments_uniq,num_years))
    print("95\% PI number of Individuals sent to Treatment: ", dict_min, dict_max)
    # print(Total_Treatments_uniq)
    print("----------------------- Hospital Encounters ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Hosp, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_Hosp,num_years), inputs.df_HE,"Number of Discharges","HospEncounters")
    print("Avg number of Hospital Encounters: ", dict_mu )
    print("95\% PI number of Hospital Encounters: ", dict_min, dict_max)
    # print(Total_Hosp)
    dict_mu, dict_max, dict_min = dict_set_mean(Total_Hosp_uniq, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_set_sd(Total_Hosp_uniq,num_years), None,"Number of Individuals Discharged","Indv_HospEncounters")
    print("Avg number of Individuals with Hospital Encounters: ",  dict_mu )
    print("95\% PI number of Individuals with Hospital Encounters: ", dict_min, dict_max)
    # print(Total_Hosp_uniq)
    print("----------------------- Prevalence ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Prev, num_years)
    ci_graph(dict_mu, dict_max, dict_min, dict_sd(Total_Prev,num_years),inputs.df_prev["Year"], inputs.df_prev["Dane County use estimate (number of people) LOWER CI"].astype(float),inputs.df_prev["Dane County use estimate (number of people)"].astype(float),"Prevalence")
    print("Avg Opioid Prevalence: ", dict_mu )
    print("95\% PI Opioid Prevalence: ", dict_min, dict_max)
    # print(Total_Prev)
    print("----------------------- Relapse ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Relapse, num_years)
    ci_graph_point(dict_mu, dict_max, dict_min, dict_sd(Total_Relapse,num_years), None, "Number of Relapses","Relapse")
    print("Avg Opioid Relapses: ", dict_mu )
    print("95\% PI Opioid Relapses: ", dict_min, dict_max)
    #Print Times summary
    print("----------------------- Times Summary Table ------------------------")
    print(df_times.to_string())

    endtime = ti.time()
    print("----------------------- Total Simulation Time to Run ------------------------")
    print ('total time elapsed:', endtime - starttime)

    sys.stdout = original_stdout

    write_file2.close()
    print("Done writing Summary file")
    today = datetime.now()
    dt_string = today.strftime("%m%d%y_%H%M%S")
    
    dst = 'Results_ED2RVal_'+ str(int(str_MatrixVal))+ '_MARIVal_' + str(int(str_MARIVal)) + '_CMVal_' + str(int(str_CMVal)) +'_Scen_'+ str(n_runs) + '_Years_'+str(num_years)+ '_Time_'+ dt_string
    os.rename('Results', dst)
    print(" renamed results folder to: ", dst)
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