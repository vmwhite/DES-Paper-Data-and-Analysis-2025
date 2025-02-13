from Simulation_Functions.math_functions import *
import sys
from Simulation_Functions.person_class import User
def sim_params(warmup):
    params = []
    ### arrival rate ######
    lam_user_arrival = 10.87 #Rate of initiation, i.e. Number of person that use opioids for the first time per day.
    
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
    ODdeathdays_est = []
    #Current baseline 102822_030838 RESULTS
    ODdeathdays_est.append({"l":0, "m":1, "q":0.002651957, "x":365.25}) #<- q is average yearly OD deaths / average yealry prevalence est
    ODdeathdays_est.append({"l":0, "m":1, "q":0.003326891, "x":365.25}) #<- q is average yearly OD deaths / average yealry prevalence est   
    LNmean_deathdays = []
    LNsigma_deathdays = []
    for idx, i in enumerate(ODdeathdays_est):
        mu, sig =  ln_est(ODdeathdays_est[idx]["l"],ODdeathdays_est[idx]["m"],ODdeathdays_est[idx]["q"],ODdeathdays_est[idx]["x"])
        LNmean_deathdays.append(mu)
        LNsigma_deathdays.append(sig)

    #ARC 3
    LNmean_hospdays = 9.072755796855650 #<- only last twpo years, all four -> 8.9407719434103400
    hospdays_est = {"l":0, "m":1, "mu":LNmean_hospdays}
    LNsigma_hospdays = ln_est_sig_only(hospdays_est["l"],hospdays_est["m"],hospdays_est["mu"])

    #ARC 4
    LNmean_Oarrestdays = 10.0436515056944000
    Oarrestdays_est = {"l":0, "m":90, "mu":LNmean_Oarrestdays}
    LNsigma_Oarrestdays = ln_est_sig_only(Oarrestdays_est["l"],Oarrestdays_est["m"],Oarrestdays_est["mu"])

    
    
    #ARC 5
    #Current Baseline 60922_030838
    #note to self try: 
    treatdays_est = {"l":0, "m":610, "q":0.063285789, "x":365.25}
    #treatdays_est = {"l":0, "m":630, "q":0.060289179, "x":365.25} #<- q is average yearly treats / average yealry prevalence est 
    LNmean_treatdays, LNsigma_treatdays =  ln_est(treatdays_est["l"],treatdays_est["m"],treatdays_est["q"],treatdays_est["x"])
    
    
    #ARCS p_D, p_A, p_T
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

    #ARC 6
    #Current Baseline 60922_030838
    iadays_est = {"l":0, "m":1, "q":0.494, "x":120}
    LNmean_iadays, LNsig_iadays =  ln_est(iadays_est["l"],iadays_est["m"],iadays_est["q"],iadays_est["x"])
    #Current Excel Spreedsheet Nubmers
    # iadays_est = {"l":0, "m":1, "q":0.688, "x":120}
    # LNmean_iadays, LNsig_iadays =  ln_est(iadays_est["l"],iadays_est["m"],iadays_est["q"],iadays_est["x"])


    #ARC 7 - now calcualted based on entering age. adjust enter age probabilities here - as of 2/10/25 second major revision
    dup_prev_age_mean, dup_prev_age_sig = ln_est(12,32,.933,65)
    dup_init_age_mean, dup_init_age_sig = ln_est(12,21.5,.403,26)
    '''
    #Old
    dup_prev_age_mean, dup_prev_age_sig = ln_est(12,30,.75,42.4)
    dup_init_age_mean, dup_init_age_sig = ln_est(12,25,.75,37.9)
    '''
    #Arc 8 - NON opioid related crimes - goes from active and inactive to CJS. If from active use then eligible for CM
    #Baseline 5.4.24
    # nonOarrestdays_est = {"l":0, "m":9, "q":0.48254, "x":2374.125} #made treatments too high and O-crime a little low?
    # LNmean_nonOarrestdays, LNsigma_nonOarrestdays =  ln_est(nonOarrestdays_est["l"],nonOarrestdays_est["m"],nonOarrestdays_est["q"],nonOarrestdays_est["x"])
    #current Baseline 5.9.24
    LNmean_nonOarrestdays = 8.453750915
    nonOarrestdays_est = {"l":0, "m":9, "mu":LNmean_Oarrestdays}
    LNsigma_nonOarrestdays  = ln_est_sig_only(nonOarrestdays_est["l"],nonOarrestdays_est["m"],nonOarrestdays_est["mu"])



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
    params["n_runs"] = n_runs
    params["num_years"] = num_years
    params["days_per_year"] = days_per_year
    params["days_per_month"] = days_per_month
    params["start_year"] = start_year
    params["dup_init_age_mean"] = dup_init_age_mean
    params["dup_init_age_sig"] = dup_init_age_sig
    params["dup_prev_age_mean"] = dup_prev_age_mean
    params["dup_prev_age_sig"] = dup_prev_age_sig

    
    params["lam_user_arrival"] = lam_user_arrival
    params["LNmean_deathdays"] = LNmean_deathdays
    params["LNsigma_deathdays"] = LNsigma_deathdays
    params["ODdeathdays_est"] =ODdeathdays_est
    params["LNmean_hospdays"] = LNmean_hospdays
    params["LNsigma_hospdays"] = LNsigma_hospdays
    params["hospdays_est"] = hospdays_est
    params["LNmean_Oarrestdays"] = LNmean_Oarrestdays
    params["LNsigma_Oarrestdays"] = LNsigma_Oarrestdays
    params["Oarrestdays_est"] = Oarrestdays_est
    params["LNmean_nonOarrestdays"] = LNmean_nonOarrestdays
    params["LNsigma_nonOarrestdays"] = LNsigma_nonOarrestdays
    params["nonOarrestdays_est"] = nonOarrestdays_est
    params["LNmean_treatdays"] = LNmean_treatdays
    params["LNsigma_treatdays"] = LNsigma_treatdays
    params["treatdays_est"] = treatdays_est
    params["hospital_encounter_thres"] = hospital_encounter_thres
    params["LNmean_iadays"] = LNmean_iadays
    params["LNsig_iadays"] = LNsig_iadays
    params["iadays_est"] = iadays_est

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
    params["hospservice_est"] = hospservice_est 
    params["crimeservice_est"] = crimeservice_est 
    params["treatservice_est"] = treatservice_est 
    params["crimerel_est"] = crimerel_est 
    params["treatrel_est"] = treatrel_est
    params["hosprel_est"] = hosprel_est
    params["iarel_est"] = iarel_est 
    params["str_MatrixVal"] = str_MatrixVal
    params["HE_thres_baseline"] = hospital_encounter_thres_base
    params["ED2R_start_time"] = (ED2R_start_year - start_year) * 365.25
    params["MARI_thres"] = MARI_thres
    params["MARI_start_time"] = (MARI_start_year - start_year) * 365.25
    params["str_MARIVal"] = str_MARIVal 
    params["CM_thres"] = CM_thres
    params["CM_start_time"] = (CM_start_year - start_year) * 365.25
    params["str_CMVal"] = str_CMVal

    return params
def generate_starting_population(gen_dict, env, initial_done,params ):
    arrival_gen = gen_dict["arrival_gen"]
    crime_gen= gen_dict["Ocrime_gen"]
    hosp_gen= gen_dict["hosp_gen"]
    treat_gen= gen_dict["treat_gen"]
    start_gen = gen_dict["start_gen"]
    Person_Dict = {}
    Person = {} #Simpy dict of Users
    Persons = {} #dict and users
    ''' Average number of people in use'''
    starting_probs= []
    #print("function: ",arrival_gen.triangular(27298.81,34224.21,43260.59))
    starting_probs.append(arrival_gen.triangular(27298.81,34224.21,43260.59)) #number of indibiuals in starting population
    starting_probs.append(crime_gen.triangular(15,25,50) / starting_probs[0])
    starting_probs.append(hosp_gen.triangular(5,11,15) / starting_probs[0])
    starting_probs.append(treat_gen.triangular(300,450,500) / starting_probs[0])
    starting_probs.append((arrival_gen.triangular(starting_probs[0]/5, 2*(starting_probs[0]/5),4*(starting_probs[0]/5)))/starting_probs[0])
    for i in range(0, math.floor(starting_probs[0])):
        start_RN = start_gen.random()
        Person_Dict[i] = {}
        Person[i] = User(env,i, initial_done, Person_Dict, gen_dict, params)
        Person_Dict[i]["arrivalT"] = 0
        #test B lowered percent starting in arrest adn decrease inactive state
        if start_RN <    starting_probs[1]: #arrest
            Person_Dict[i]["Next_Interrupt"] = "crime"
            Person_Dict[i]["List_Crime_Type"]= ["Ocrime"]
        elif start_RN <    (starting_probs[2] + starting_probs[1]): #hospital
            Person_Dict[i]["Next_Interrupt"] = "hosp"
        elif start_RN <    (starting_probs[3]+ starting_probs[2] + starting_probs[1]): #treatment
            Person_Dict[i]["Next_Interrupt"] = "treat"
        elif start_RN <   (starting_probs[4]+ starting_probs[3] + starting_probs[2] + starting_probs[1]): # active
            pass #as this is how one typically enters the simulation
        else: #inactive 
            Person_Dict[i]["Next_Interrupt"] = "inactive"
    #print("function: ",arrival_gen.triangular(27298.81,34224.21,43260.59))
    Persons["dict"] = Person_Dict
    Persons["indv"] = Person
    return [Persons,starting_probs]