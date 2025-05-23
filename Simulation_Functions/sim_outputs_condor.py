import sys
import os
from Simulation_Functions.math_functions import *
from Simulation_Functions.graph_functions import *
from datetime import datetime
import time as ti

def print_model_outputs(warmup, seeds, starttime, original_stdout, results,params,Temp_results_folder):
    n_runs = params["n_runs"]
    num_years = params["num_years"]
    days_per_year = params["days_per_year"]
    days_per_month = params["days_per_month"]
    start_year= params["start_year"]
    lam_user_arrival = params["lam_user_arrival"]
    LNmean_deathdays = params["LNmean_deathdays"]
    LNsigma_deathdays = params["LNsigma_deathdays"]
    LNmean_hospdays = params["LNmean_hospdays"]
    ODdeathdays_est = params["ODdeathdays_est"]
    LNsigma_hospdays = params["LNsigma_hospdays"]
    LNmean_Oarrestdays = params["LNmean_Oarrestdays"]
    Oarrestdays_est = params["Oarrestdays_est"]
    LNsigma_Oarrestdays = params["LNsigma_Oarrestdays"]
    LNmean_nonOarrestdays = params["LNmean_nonOarrestdays"]
    nonOarrestdays_est = params["nonOarrestdays_est"]
    LNsigma_nonOarrestdays = params["LNsigma_nonOarrestdays"]
    hospdays_est = params["hospdays_est"] 
    LNmean_treatdays = params["LNmean_treatdays"]
    LNsigma_treatdays = params["LNsigma_treatdays"]
    treatdays_est = params["treatdays_est"]
    hospital_encounter_thres = params["hospital_encounter_thres"]
    LNmean_iadays = params["LNmean_iadays"]
    LNsig_iadays = params["LNsig_iadays"]
    iadays_est = params["iadays_est"]
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
    hospservice_est = params["hospservice_est"] 
    crimeservice_est = params["crimeservice_est"] 
    treatservice_est = params["treatservice_est"] 
    crimerel_est = params["crimerel_est"] 
    treatrel_est = params["treatrel_est"]
    hosprel_est = params["hosprel_est"]
    iarel_est = params["iarel_est"] 
    str_MARIVal= params["str_MARIVal"] 
    str_CMVal= params["str_CMVal"]
    str_MatrixVal= params["str_MatrixVal"] 
    #################### set up enmpty lists ###############################
    #Total Numbers
    Total_OD_Deaths = {}
    Total_nOD_Deaths = {}
    Total_OCrimes = {}
    Total_OCrimes_uniq = {} 
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
    Total_AllCrimes = {}

    #dataframes
    df_times = pd.DataFrame()
    for index, s in enumerate(results):
        Total_newUsers[index] = s[0]
        Total_Relapse[index] = s[1]
        Total_ActiveUsers[index] = s[2]
        Total_Prev[index] = s[3]
        Total_OD_Deaths[index] = s[4]
        Total_nOD_Deaths[index] = s[5]
        Total_OCrimes[index] = s[6]
        Total_OCrimes_uniq[index] = s[7]
        Total_Treatments[index] = s[8]
        Total_Treatments_uniq[index] = s[9]
        Total_Hosp[index] = s[10]
        Total_Hosp_uniq[index] = s[11]
        Total_InactiveUsers[index] = s[12]
        s[13].insert(0, "Scenerio", index, True)
        df_times = pd.concat([df_times, s[13]], ignore_index=True, sort=False)
        Total_AllCrimes[index] = s[14]
    write_file2 = open(Temp_results_folder+"/Test_SimulationsStatsSUMMARY_"+str(n_runs)+"Runs.txt", "w+")

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
    print("ARC 4 Opioid-Related Arrest Next Event Time: mu = %f, sigma = %f " %(LNmean_Oarrestdays, LNsigma_Oarrestdays))
    print(Oarrestdays_est)
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
    #n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_newUsers,num_years), inputs.df_initiation,"Dane County Opioid Initiation estimate (number of people)","Arrival")
    print("Avg number of Opioid Initations: ", dict_mu)
    print("95\% PI number of Opioid Initations: ", dict_min, dict_max)
    dict_mu, dict_max, dict_min = dict_mean(Total_ActiveUsers, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_ActiveUsers,num_years), None, "Number of Active Users at the end of the year","Active")
    print("Avg number of Individuals with Active Use: ", dict_mu)
    print("95\% PI number of Individuals with Active Use: ", dict_min, dict_max)
    dict_mu, dict_max, dict_min = dict_mean(Total_InactiveUsers, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_InactiveUsers,num_years), None, "Number of Active Users at the end of the year","Inactive")
    print("Avg number of Individuals with Inactive Use: ", dict_mu)
    print("95\% PI number of Individuals with Inactive Use: ", dict_min, dict_max)
    
    print("----------------------- Opioid Deaths ------------------------")
    dict_mu, dict_max, dict_min=dict_mean(Total_OD_Deaths, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_OD_Deaths,num_years), inputs.df_DCdeaths,'Deaths',"OR_Death")
    print("Avg number of Opioid Deaths: ", dict_mu)
    print("95\% PI number of Opioid Deaths: ", dict_min, dict_max)
    # print(Total_OD_Deaths)
    print("----------------------- non-Opioid Deaths ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_nOD_Deaths, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_nOD_Deaths,num_years), None,'Non-Opioid-Related Deaths',"nOR_Death")
    print("Avg number of non-Opioid Deaths: ", dict_mu )
    print("95\% PI number of non-Opioid Deaths: ", dict_min, dict_max)
    # print(Total_nOD_Deaths)
    print("----------------------- Opioid-related Crimes ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_OCrimes, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_OCrimes,num_years), inputs.df_Yarrests,'Opioid-Related Arrests',"O_Arrests")
    print("Avg number of Opioid-related Arrests: ", dict_mu)
    print("95\% PI number of Opioid Related Arrests: ", dict_min, dict_max)
    # print(Total_Crimes)
    dict_mu, dict_max, dict_min = dict_set_mean(Total_OCrimes_uniq, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_set_sd(Total_OCrimes_uniq,num_years), None,'Unique Individuals Arrested for an Opioid-Related Crime',"Indv_OArrested")
    print("Avg number of Individuals Arrested for an Opioid-related crime: ",  dict_mu )
    print("95\% PI number of Individuals Arrested for an Opioid-related crime: ", dict_min, dict_max)
    # print(Total_Crimes_uniq)
    print("----------------------- All Crimes ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_AllCrimes, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_AllCrimes,num_years), inputs.df_Yarrests,'Arrests (All)',"Arrests(All)")
    print("Avg number of Arrests (All): ", dict_mu)
    print("95\% PI number of Arrests (All): ", dict_min, dict_max)
    # print(Total_Crimes)
    print("----------------------- Treatment ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Treatments, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_Treatments,num_years), None,'Treatment Starts',"Treatments")
    print("Avg number of Treatment Starting Episodes: ",  dict_mu )
    print("95\% PI number of Treatment Starting Episodes: ", dict_min, dict_max)
    # print(Total_Treatments)
    dict_mu, dict_max, dict_min = dict_set_mean(Total_Treatments_uniq,num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_set_sd(Total_Treatments_uniq,num_years), inputs.df_treat,'Total Individuals',"Indv_Treated")
    print("Avg number of Individuals sent to Treatment: ", dict_set_mean(Total_Treatments_uniq,num_years))
    print("95\% PI number of Individuals sent to Treatment: ", dict_min, dict_max)
    # print(Total_Treatments_uniq)
    print("----------------------- Hospital Encounters ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Hosp, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_Hosp,num_years), inputs.df_HE,"Number of Discharges","HospEncounters")
    print("Avg number of Hospital Encounters: ", dict_mu )
    print("95\% PI number of Hospital Encounters: ", dict_min, dict_max)
    # print(Total_Hosp)
    dict_mu, dict_max, dict_min = dict_set_mean(Total_Hosp_uniq, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_set_sd(Total_Hosp_uniq,num_years), None,"Number of Individuals Discharged","Indv_HospEncounters")
    print("Avg number of Individuals with Hospital Encounters: ",  dict_mu )
    print("95\% PI number of Individuals with Hospital Encounters: ", dict_min, dict_max)
    # print(Total_Hosp_uniq)
    print("----------------------- Prevalence ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Prev, num_years)
    #ci_graph(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_Prev,num_years),inputs.df_prev["Year"], inputs.df_prev["Dane County use estimate (number of people) LOWER CI"].astype(float),inputs.df_prev["Dane County use estimate (number of people)"].astype(float),"Prevalence")
    print("Avg Opioid Prevalence: ", dict_mu )
    print("95\% PI Opioid Prevalence: ", dict_min, dict_max)
    # print(Total_Prev)
    print("----------------------- Relapse ------------------------")
    dict_mu, dict_max, dict_min = dict_mean(Total_Relapse, num_years)
    #ci_graph_point(n_runs, start_year, num_years, warmup, dict_mu, dict_max, dict_min, dict_sd(Total_Relapse,num_years), None, "Number of Relapses","Relapse")
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
    process = int(sys.argv[3])
    dst = 'Results_'+str(process)+'Process_ED2RVal_'+ str(int(str_MatrixVal))+ '_MARIVal_' + str(int(str_MARIVal)) + '_CMVal_' + str(int(str_CMVal)) +'_Scen_'+ str(n_runs) + '_Years_'+str(num_years)+ '_Time_'+ dt_string
    os.rename(Temp_results_folder, dst)
    print(" renamed temporary results folder to: ", dst)