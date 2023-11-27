'''' 
This File Created the Scenario Comparisons
Need to run Comppile_RawResults.py before running htis file
'''
from asyncio.windows_events import NULL
from contextlib import nullcontext
from Graphing_Functions import *
from Table_Functions import Avg_PerPersonRatesTable
import pandas as pd
import os
import statistics as st
import scipy

####folders with results
Baseline_Output_Folder_OG = r'Results_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023844/summaries600'
Baseline_Folder = r'Results_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023844/summaries'
Baseline_file_list = os.listdir(Baseline_Folder)
color_list = ["gray"]

All_RESULTS_folder_list = [r'Results_ED2RVal_22270_MARIVal_0_CMVal_20_Scen_1000_Years_25_Time_033023_092508/summaries',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_40_Scen_1000_Years_25_Time_033023_024503/summaries',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_60_Scen_1000_Years_25_Time_033023_042120/summaries',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_80_Scen_1000_Years_25_Time_033023_031124/summaries',
r'Results_ED2RVal_22270_MARIVal_0_CMVal_100_Scen_1000_Years_25_Time_033023_021852/summaries',
r'Results_ED2RVal_22270_MARIVal_20_CMVal_0_Scen_1000_Years_25_Time_033023_022830/summaries',
r'Results_ED2RVal_22270_MARIVal_40_CMVal_0_Scen_1000_Years_25_Time_033023_040647/summaries',
r'Results_ED2RVal_22270_MARIVal_60_CMVal_0_Scen_1000_Years_25_Time_033023_014303/summaries',
r'Results_ED2RVal_22270_MARIVal_80_CMVal_0_Scen_1000_Years_25_Time_033023_023019/summaries',
r'Results_ED2RVal_22270_MARIVal_100_CMVal_0_Scen_1000_Years_25_Time_033023_025006/summaries',
r'Results_ED2RVal_30000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_131333/summaries',
r'Results_ED2RVal_40000_MARIVal_20_CMVal_20_Scen_1000_Years_25_Time_033023_015145/summaries',
r'Results_ED2RVal_45000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023133/summaries',
r'Results_ED2RVal_60000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_043938/summaries',
r'Results_ED2RVal_60000_MARIVal_40_CMVal_40_Scen_1000_Years_25_Time_033023_021800/summaries',
r'Results_ED2RVal_75000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_050610/summaries',
r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_1000_Years_25_Time_033023_030711/summaries',
r'Results_ED2RVal_90000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_031549/summaries']
ScenOrder_files_to_table = {0:0, 
                            1:11, 2:12, 3:13, 4:14, 5:15, 
                            6:1, 7:2, 8:3, 9:4, 10:5,
                            11:6, 12:16, 13:7, 14:8, 15:17, 
                            16:9, 17:18, 18:10}
Output_Scenario = ['Base', 
                   'CM20','CM40','CM60','CM80','CM100',
                   'AD20','AD40','AD60','AD80','AD100',
                   'OD30','AD20_OD40_CM20', 'OD45','OD60', 'AD40_OD60_CM40', 
                   'OD75', 'AD60_OD80_CM60', 'OD90']
#number of scenarios to include
n = 600
##### parameters
months_in_year = 12 
num_years = 25
warmup = 5
actual_warmup = 5
start_year = 2013 - actual_warmup
n_runs = n
year_list = [2022,2027,2032]
# Params for CI Graphs
# plt.rcParams["axes.titlesize"] = 20
# plt.rcParams["legend.fontsize"] = 16
# plt.rcParams["axes.labelsize"] = 20
# plt.rcParams['xtick.labelsize'] = 18
# plt.rcParams['ytick.labelsize'] = 18
# Params for per person Graphs
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
baseline_histcreated = False
number_base = 0
joint_idx = 0
joint_idx_set = [8,3,13,16,17,18,19]
Output_Scenario_TableOrder = ['Base', 'AD20','AD40','AD60','AD80','AD100','OD30', 'OD45',
                     'OD60',  'OD75', 'OD90', 'CM20','CM40','CM60','CM80','CM100','AD20_OD40_CM20','AD40_OD60_CM40','AD60_OD80_CM60']
policy_dict = {
    'Base' : '0, 22, 0', 
    'AD20' : '20, 22, 0',
    'AD40' : '40, 22, 0',
    'AD60' : '60, 22, 0',
    'AD80' : '80, 22, 0',
    'AD100' : '100, 22, 0',
    'OD30' : '0, 30, 0', 
    'OD45' : '0, 45, 0',
    'OD60' : '0, 60, 0',  
    'OD75' : '0, 75, 0',
    'OD90' : '0, 90, 0',
    'CM20' : '0, 22, 20',
    'CM40' : '0, 22, 40',
    'CM60' : '0, 22, 60',
    'CM80' : '0, 22, 80',
    'CM100' : '0, 22, 100',
    'AD20_OD40_CM20' : '20, 40, 20',
    'AD40_OD60_CM40' : '40, 60, 40',
   'AD60_OD80_CM60': '60, 80, 60'
}
scen_list = []
output_list = []
table_list = []
year_mu_list = [[] for _ in range(len(year_list))]
year_err_list = [[] for _ in range(len(year_list))]
year_max_list = [[] for _ in range(len(year_list))]
year_min_list = [[] for _ in range(len(year_list))]
year_diffp_list = [[] for _ in range(len(year_list))]
year_diffsig_list = [[] for _ in range(len(year_list))]

new_year_list = []
for year in range(start_year, start_year+num_years): 
    new_year_list.append(int(year))
new_month_list = []
for month in range(0, num_years*months_in_year): 
    new_month_list.append(int(month))
try: 
    os.makedirs(Baseline_Output_Folder_OG + "\Comparison_MARI")
    os.makedirs(Baseline_Output_Folder_OG + "\Comparison_ED2R")
    os.makedirs(Baseline_Output_Folder_OG + "\Comparison_CM")
    os.makedirs(Baseline_Output_Folder_OG + "\Comparison_Joint")
except:
    pass
for policy in ["Joint"]:#:["AD", "OD", "CM", "Joint"]:
    '''MARI Labels'''
    if policy == "AD":
        Baseline_Output_Folder = Baseline_Output_Folder_OG + '/Comparison_MARI'
        comparison_folder_list = [r'Results_ED2RVal_22270_MARIVal_20_CMVal_0_Scen_1000_Years_25_Time_033023_022830/summaries',
        r'Results_ED2RVal_22270_MARIVal_40_CMVal_0_Scen_1000_Years_25_Time_033023_040647/summaries',
        r'Results_ED2RVal_22270_MARIVal_60_CMVal_0_Scen_1000_Years_25_Time_033023_014303/summaries',
        r'Results_ED2RVal_22270_MARIVal_80_CMVal_0_Scen_1000_Years_25_Time_033023_023019/summaries',
        r'Results_ED2RVal_22270_MARIVal_100_CMVal_0_Scen_1000_Years_25_Time_033023_025006/summaries'] 
        ls_style=[(5, (10, 3)), '--', "-.",  (0, (3, 10, 1, 10, 1, 10)), ':']
        color_style= ['gray', 'red', 'green', 'blue', 'cyan']
        leg_labels = ['Baseline', 'AD 20%', 'AD 40%', 'AD 60%', 'AD 80%', 'AD 100%']
        #ls_style=[(5, (10, 3)), '--', "-."]
        #color_style= ['gray', "blue", "red"]
        #leg_labels = ['Baseline',  'AD 60%', 'AD 100%']

    '''ED 2 REcovery Labels'''
    #Option 1: The following 5 are just straight forward probabilities an individual can be sent to treatment 
    # [ 0.3, 0.45, 0.6, 0.75, 0.9]
    if policy == "OD":
        Baseline_Output_Folder = Baseline_Output_Folder_OG + '/Comparison_ED2R'
        comparison_folder_list = [r'Results_ED2RVal_30000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_131333/summaries',
        r'Results_ED2RVal_45000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023133/summaries',
        r'Results_ED2RVal_60000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_043938/summaries',
        r'Results_ED2RVal_75000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_050610/summaries',
        r'Results_ED2RVal_90000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_031549/summaries']
        color_style= ['gray', 'red', 'green', 'blue', 'cyan']
        ls_style=[(5, (10, 3)), '--', "-.",  ':', (0, (1, 1.5))]
        leg_labels = ['Baseline', 'OD 30%', 'OD 45%', 'OD 60%', 'OD 75%', 'OD 90%']
    

    #option 2: the following 5 are each 1/5 of the remaining population between the baseline (i.e., 0.2227) and max (0.9582) values can be sent to treatment
    # [0.3698, 0.5169, 0.664, 0.8111, 0.9582]
    '''
    comparison_folder_list = []
    ls_style=[(5, (10, 3)), '--', "-.",  (0, (3, 10, 1, 10, 1, 10)), ':']
    color_style= ['gray', 'gray', 'gray', 'gray', 'gray',]
    leg_labels = ['Baseline', 'OD 37%', 'OD 52%',  'OD 66%', 'OD 81%', 'OD 95%']
    '''

    #Option 3: The following 6 are the (25,40,55,70,85,100) percentages of the Individuals eligible to be sent to treatment
    # [0.23145,0.37032, 0.50919, 0.64806, 0.78693, 0.9582]
    # comparison_folder_list = [] 
    # ls_style=[(5, (10, 3)), '--', "-.",  (0, (3, 10, 1, 10, 1, 10)), ':', (0, (1, 10))]
    # leg_labels = ['Baseline', 'OD 25%', 'OD 40%', 'OD 55%', 'OD 70%', 'OD 85%', 'OD 100%']

    '''CM Labels'''
    if policy == "CM":
        Baseline_Output_Folder = Baseline_Output_Folder_OG + '/Comparison_CM'
        comparison_folder_list = [r'Results_ED2RVal_22270_MARIVal_0_CMVal_20_Scen_1000_Years_25_Time_033023_092508/summaries',
        r'Results_ED2RVal_22270_MARIVal_0_CMVal_40_Scen_1000_Years_25_Time_033023_024503/summaries',
        r'Results_ED2RVal_22270_MARIVal_0_CMVal_60_Scen_1000_Years_25_Time_033023_042120/summaries',
        r'Results_ED2RVal_22270_MARIVal_0_CMVal_80_Scen_1000_Years_25_Time_033023_031124/summaries',
        r'Results_ED2RVal_22270_MARIVal_0_CMVal_100_Scen_1000_Years_25_Time_033023_021852/summaries'] 
        ls_style=[(5, (10, 3)), '--', "-.", ':',(0, (1, 10))]
        color_style= ['gray', 'red', 'green', 'blue', 'cyan']
        leg_labels = ['Baseline', 'CM 20%', 'CM 40%', 'CM 60%', 'CM 80%', 'CM100%']
    # '''

    '''Joint Strategy Labels'''
    if policy == "Joint":
        Baseline_Output_Folder = Baseline_Output_Folder_OG + '/Comparison_Joint'
        comparison_folder_list = [r'Results_ED2RVal_60000_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_043938/summaries',
        r'Results_ED2RVal_22270_MARIVal_60_CMVal_0_Scen_1000_Years_25_Time_033023_014303/summaries',
        r'Results_ED2RVal_22270_MARIVal_0_CMVal_60_Scen_1000_Years_25_Time_033023_042120/summaries',
        r'Results_ED2RVal_40000_MARIVal_20_CMVal_20_Scen_1000_Years_25_Time_033023_015145/summaries',
        r'Results_ED2RVal_60000_MARIVal_40_CMVal_40_Scen_1000_Years_25_Time_033023_021800/summaries',
        r'Results_ED2RVal_80000_MARIVal_60_CMVal_60_Scen_1000_Years_25_Time_033023_030711/summaries']
        color_style= [ 'green', 'cyan', 'gray','darkviolet' , 'red', 'blue' ]
        ls_style=[(5, (10, 3)), '--', (0, (3, 1, 1, 1)),"-.",  ':',(1 ,(1, 2,3,2))]
        leg_labels = ['Baseline', 'OD: 60% Only', 'AD 60% Only', 'CM 60% Only', 'OD: 40%, AD: 20%, CM: 20%','OD: 60%, AD: 40%, CM: 40%', 'OD: 80%, AD: 60%, CM: 60%']

    # color_style= ['red', 'blue']
    # ls_style=[(5, (10, 3)), (0, (3, 1, 1, 1)),':']
    
    # color_style= ['grey', 'green', 'orange','red', 'blue',]
    # ls_style=[(5, (10, 3)), '--',  (0, (3, 10, 1, 10, 1, 10)),  (0, (3, 1, 1, 1)),':']\\
    # leg_labels = ['Baseline', 'OD: 60%, AD: 40%, CM: 40%','OD: 80%, AD: 60%, CM: 60%']
    # '''

    ##### joint confidence interval
    '''
    https://www.statology.org/bonferroni-correction/
    Bonferroni-correction, individual alpha = 0.0025613787765302876
    '''
    alpha_joint = 0.05
    alpha_indv = 1 - (1- alpha_joint)**(1/(num_years-warmup))
    # print(alpha_indv)
    z_score = st.NormalDist().inv_cdf(1 - alpha_indv)
    check = False
    for idx, filename in enumerate(Baseline_file_list):
        secondary_csv = ''
        third_csv = ""
        if "y_Arrests" in filename:
            csv_filename = '/Base_Yearly_Arrests.csv'
            out_pic_name = "/Crimes_CI.png"
            name = "Opioid-Related Arrests per Year"
            xlabel = "Opioid-Related Arrests"
            check = False
        elif "y_Arrivals" in filename:
            csv_filename = '/Base_Yearly_Arrivals.csv'
            out_pic_name = "/Arrivals_CI.png"
            name = "Opioid Initiations per Year"
            xlabel = "Opioid Initiations"
            check = False
        elif "y_Active" in filename:
            csv_filename = '/Base_Yearly_Active_YearEnd.csv'
            out_pic_name = "/Active_CI.png"
            name = "Individuals in the Active State at the End of each Year"
            xlabel = "Individuals"
            check = False
        elif "y_Hosp" in filename:
            csv_filename = '/Base_Yearly_Hosp.csv'
            out_pic_name = "/Hosp_CI.png"
            name = "Hospital Encounters per Year"
            xlabel = "Hospital Encounters"
            check = False
        elif "y_NonODeaths" in filename:
            csv_filename = '/Base_Yearly_NonODeaths.csv'
            out_pic_name = "/NonODeaths_CI.png"
            name = "Non-Opioid Related Deaths per Year"
            xlabel = "Individuals"
            check = False
        elif "y_ODeaths" in filename:
            csv_filename = '/Base_Yearly_ODeaths.csv'
            out_pic_name = "/ODeaths_CI.png"
            name = "Opioid-Related Deaths per Year"
            xlabel = "Individuals"
            check = False
        elif "y_Relapses" in filename:
            csv_filename = '/Base_Yearly_Relapses.csv'
            out_pic_name = "/Relapses_CI.png"
            name = "Relapses per Year"
            xlabel = "Individuals"
            check = False
        elif "y_Treats" in filename:
            csv_filename = '/Base_Yearly_Treats.csv'
            out_pic_name = "/Treats_CI.png"
            name = "Treatment Episodes per Year"
            xlabel = "Treatment Episodes"
            check = False
        elif "y_Inactive" in filename:
            csv_filename = '/Base_Yearly_Inactive_YearEnd.csv'
            out_pic_name = "/Inactive_CI.png"
            name = "Individuals in the Inactive State at the End of each Year"
            xlabel = "Individuals"
            check = False
        elif "m_Crimes" in filename:
            csv_filename = '/Cum_Arrests.csv'
            out_pic_name = "/Cum_Crimes_CI.png"
            name = "Cumulative Number of Arrests"
            xlabel = "Cumulative Arrests"
            check = False
        elif "m_Hosp" in filename:
            csv_filename = '/Cum_Hosp.csv'
            out_pic_name = "/Cum_Hosp_CI.png"
            name = "Cumulative Number of Hospital Encounters"
            xlabel = "Cumulative Hospital Encounters"
            check = False
        elif "m_Treats" in filename:
            csv_filename = '/Cum_Treats.csv'
            out_pic_name = "/Cum_Treats_CI.png"
            name = "Cumulative Number of Treatment Episodes"
            xlabel = "Cumulative Treatment Episodes"
            check = False
        elif "m_ODeaths" in filename:
            csv_filename = '/Cum_ODeaths.csv'
            out_pic_name = "/Cum_ODeaths_CI.png"
            name = "Cumulative Number of Opioid-Related Deaths"
            xlabel = "Cumulative Opioid-Related Deaths"
            check = False
        elif "m_Relapses" in filename:
            csv_filename = '/Cum_Relapses.csv'
            out_pic_name = "/Cum_Relapses_CI.png"
            name = "Cumulative Number of Relapses"
            xlabel = "Cumulative Relapses"
            check = False
        elif "Arrest_ut" in filename:
            csv_filename = '/Base_MonthEnd_Arrest_ut.csv'
            out_pic_name = "/Ut_CJS.png"
            name = "Required Opioid-Related \n CJS Capacity over time"
            check = True
            ylabel = "Number of Individuals in the CJS \n due to an Opioid-Related Arrest"
        elif "Hosp_ut" in filename:
            csv_filename = '/Base_MonthEnd_Hosp_ut.csv'
            out_pic_name = "/Ut_HOSP.png"
            name = "Required Opioid Encounter \n Hospital Capacity over time"
            ylabel = "Number of Individuals in the Hospital \ndue to an Opioid Encounter"
            check = True
        elif "Treat_ut" in filename:
            csv_filename = '/Base_MonthEnd_Treat_ut.csv'
            out_pic_name = "/Ut_Treat.png"
            name = "Required Capacity of OUD \n Treatment over time"
            ylabel = "Number of Individuals \n in OUD Treatment"
            check = True
        elif "y_Indv_Arrest" in filename:
            csv_filename = '/Base_Yearly_Indv_Arrests.csv'
            out_pic_name = "/Per_person_Arrest.png"
            name = "Opioid-related Re-Arrest rate per Year"
            check = False
            secondary_csv = '/Base_Yearly_Arrests.csv'
            ylabel = "Opioid-related Re-Arrest Rate (%)"
        elif "y_Indv_Hosp" in filename:
            csv_filename = '/Base_Yearly_Indv_Hosp.csv'
            out_pic_name = "/Per_person_Hosp.png"
            name = "Re-Hospitalization Rate per Year"
            secondary_csv = '/Base_Yearly_Hosp.csv'
            check = False
            ylabel = "Re-Hospitalization Rate (%)"
        elif "y_Indv_Treats" in filename:
            csv_filename = '/Base_Yearly_Indv_Treats.csv'
            out_pic_name = "/Per_person_Treat.png"
            name = "OUD Treatment Re-start Rate per Year"
            secondary_csv = '/Base_Yearly_Treats.csv'
            check = False
            ylabel = "OUD Treatment Re-start Rate (%)"
        elif "y_Prevalence" in filename:
            csv_filename = '/Base_Yearly_Prevalence.csv'
            out_pic_name = "/Per_person_Prev.png"
            name = "Opioid use Re-start Rate per Year"
            secondary_csv = '/Base_Yearly_Relapses.csv'
            third_csv = '/Base_Yearly_Arrivals.csv'
            check = False
            ylabel = "Opioid uses Re-start Rate (%)"
        else:
            continue
        df_baseline = pd.read_csv(r''+Baseline_Folder+ csv_filename )
        df_baseline = df_baseline[0:n]
        df_baseline = df_baseline.T
        save_as = str(Baseline_Output_Folder) + out_pic_name
        if check == False:
            if secondary_csv == "":
                #continue
                ''' Plotting the Yearly total CIs'''
                base_mu, base_max, base_min = dict_mean(df_baseline[1:], num_years)
                base_sd = dict_sd(df_baseline[1:],num_years)        
                ''' Plot Baseline Scenario '''
                plt.figure(figsize=(10, 6),dpi=600)
                errSIM = base_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
                new_year_list = []
                for year in range(start_year, start_year+num_years): 
                    new_year_list.append(int(year))
                plt.errorbar(new_year_list[warmup:],base_mu[warmup:],errSIM[warmup:], elinewidth = 1, capsize=10, color ="black")
            
                ''' Plot each comparison '''
                for idx,Comparison_Folder in enumerate(comparison_folder_list):
                    if idx == 3:
                        pass
                    else:
                        df_comparison = pd.read_csv(r''+Comparison_Folder+ csv_filename)
                        df_comparison = df_comparison[0:n]
                        df_comparison = df_comparison.T
                        comp_mu, comp_max, comp_min = dict_mean(df_comparison[1:], num_years)
                        comp_sd = dict_sd(df_comparison[1:],num_years)
                        errCOMP = comp_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
                        plt.errorbar(new_year_list[warmup:],comp_mu[warmup:],errCOMP[warmup:], elinewidth = 1, capsize=10, color = color_style[idx],ls=ls_style[idx], alpha=.9)
                        # ci_graph_point(base_mu, base_sd, comp_mu, comp_sd, name ,start_year, num_years, warmup, n_runs,save_as,z_score)
                
                ''' Final plot changes '''
                plt.xlabel('Year')
                plt.xticks(np.arange(2010, 2035, step=5),rotation=45)
                # plt.ylabel("The Number of " + name)
                plt.ylabel("Number of "+xlabel)
                #plt.title("Simulated Joint 95% Confidence Intervals of \n the Number of " + name )
                #plt.legend(leg_labels) 
                plt.autoscale() 
                plt.tight_layout()
                plt.savefig(save_as)
                leg = plt.legend(leg_labels)
                ''' Legend File'''
                # then create a new image 
                # adjust the figure size as necessary 
                figsize = (4.5, 3)
                fig_leg = plt.figure(figsize=figsize)
                ax_leg = fig_leg.add_subplot( 111)
                # add the legend from the previous axes 
                #ax_leg.legend(leg.get_legend(), loc= 'center')
                ax_leg.legend(*ax.get_legend_handles_labels(), loc= 'center')
                # hide the axes frame and the x/y labels 
                ax_leg.axis('off')
                fig_leg.tight_layout()
                fig_leg.savefig(save_as +'_legend.png')
                plt.close()
        
            else:
                continue
                if third_csv != "":
                    #continue
                    new_year_list = []
                    for year in range(start_year, start_year+num_years): 
                        new_year_list.append(int(year))
                    df_baseline_p2 = pd.read_csv(r''+Baseline_Folder +secondary_csv)
                    df_baseline_p2 = df_baseline_p2[0:n]
                    df_baseline_p2 = df_baseline_p2.T
                    df_baseline_p3 = pd.read_csv(r''+Baseline_Folder +third_csv)
                    df_baseline_p3 = df_baseline_p3[0:n]
                    df_baseline_p3 = df_baseline_p3.T
                    df_baseline = (((df_baseline_p2 + df_baseline_p3) / df_baseline) -1)*100
                    base_mu, base_max, base_min = dict_mean(df_baseline[1:], num_years)
                    base_sd = dict_sd(df_baseline[1:],num_years)
                    errbase = base_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
                else:
                    t_crit = scipy.stats.t.ppf(q=1-(.05/2),df=599)
                    ''' Plot Average Number per Person '''
                    new_year_list = []
                    for year in range(start_year, start_year+num_years): 
                        new_year_list.append(int(year))
                    df_baseline_p2 = pd.read_csv(r''+Baseline_Folder +secondary_csv)
                    df_baseline_p2 = df_baseline_p2[0:n]
                    df_baseline_p2 = df_baseline_p2.T
                    df_baseline = ((df_baseline_p2/ df_baseline)-1)*100
                    # make figues
                    base_mu, base_max, base_min = dict_mean(df_baseline[1:], num_years)
                    base_sd = dict_sd(df_baseline[1:],num_years)
                    errbase = (base_sd/ np.sqrt(n_runs))*z_score #normal mean confidence intervals
                    fig, ax = plt.subplots(dpi=1200)
                    ax.errorbar(new_year_list[warmup:],base_mu[warmup:],errbase[warmup:], elinewidth = 1, capsize=10, color = 'black', alpha=.9)
                    # add to table lists
                    # scen_list.append(Output_Scenario_TableOrder[0])
                    # output_list.append(csv_filename)
                    for i, year in enumerate(year_list):
                        table_list.append([Output_Scenario[0], csv_filename, year, base_mu[year - start_year], errbase[year - start_year]/z_score, base_max[year - start_year], base_min[year - start_year], "" ,""])
                        # year_mu_list[i].append(base_mu[year - start_year])
                        # year_err_list[i].append(errbase[year - start_year]/z_score)
                        # year_max_list[i].append(base_max[year - start_year])
                        # year_min_list[i].append(base_min[year - start_year])
                        # year_diffp_list[i].append("")
                        # year_diffsig_list[i].append("")
                    for idx,Comparison_Folder in enumerate(All_RESULTS_folder_list):
                        df_comparison_p1 = pd.read_csv(r''+Comparison_Folder+ csv_filename)
                        df_comparison_p1 = df_comparison_p1[0:n]
                        df_comparison_p1 = df_comparison_p1.T
                        df_comparison_p2 = pd.read_csv(r''+Comparison_Folder+ secondary_csv)
                        df_comparison_p2 = df_comparison_p2[0:n]
                        df_comparison_p2 = df_comparison_p2.T
                        df_comparison = ((df_comparison_p2 / df_comparison_p1)-1)*100
                        comp_mu, comp_max, comp_min = dict_mean(df_comparison[1:], num_years)
                        comp_sd = dict_sd(df_comparison[1:],num_years)
                        errCOMP = (comp_sd/ np.sqrt(n_runs))*z_score #normal mean confidence intervals
                        df_diff = dict_diff(df_comparison[1:],df_baseline[1:],num_years)
                        # add to table lists
                        # scen_list.append(Output_Scenario_TableOrder[ScenOrder_files_to_table[idx]])
                        # output_list.append(csv_filename)
                        for i, year in enumerate(year_list):
                            table_list.append([Output_Scenario[idx+1], csv_filename, year, comp_mu[year - start_year], errCOMP[year - start_year]/z_score, comp_max[year - start_year], comp_min[year - start_year], df_diff.iloc[4,year - start_year] ,df_diff.iloc[5,year - start_year]])
                            # year_mu_list[i].append(comp_mu[year - start_year])
                            # year_err_list[i].append(errCOMP[year - start_year]/ z_score)
                            # year_max_list[i].append(comp_min[year - start_year])
                            # year_min_list[i].append(comp_max[year - start_year])
                            # year_diffp_list[i].append(df_diff.iloc[4,year - start_year])
                            # year_diffsig_list[i].append(df_diff.iloc[5,year - start_year])
                        #add to graph
                        if Comparison_Folder in comparison_folder_list:
                            scen_idx =comparison_folder_list.index(Comparison_Folder)
                            ax.errorbar(new_year_list[warmup:],comp_mu[warmup:],errCOMP[warmup:], elinewidth = 1, capsize=10, color = color_style[scen_idx],ls=ls_style[scen_idx], alpha=.9)                       
                    ''' Final plot changes '''
                    #continue
                    ax.set_xlabel('Year')
                    ax.set_xticks(np.arange(2010, 2035, step=5),rotation=45)
                    #ax.set_yticks(range(0,10,2))
                    ax.set_ylabel(ylabel)
                    #ax.set_title("Simulated Joint 95% Confidence Intervals of \n" + name )
                    #legend = ax.legend(leg_labels) 
                    #legend.get_frame().set_alpha(0)
                    fig.tight_layout()
                    fig.savefig(save_as)
                    plt.close(fig)  
                    ''' Legend File'''
                    # then create a new image 
                    # adjust the figure size as necessary 
                    figsize = (4.5, 3)
                    fig_leg = plt.figure(figsize=figsize)
                    ax_leg = fig_leg.add_subplot( 111)
                    # add the legend from the previous axes 
                    ax_leg.legend(*ax.get_legend_handles_labels(), loc= 'center')
                    # hide the axes frame and the x/y labels 
                    ax_leg.axis('off')
                    fig_leg.tight_layout()
                    fig_leg.savefig(save_as +'_legend.png')  
                    plt.close(fig)
                    ''' Final Table changes'''
                    
        else:
            ''' Plot each Required Capacities '''
            continue
            if baseline_histcreated == False:
                # continue
                base_mu, base_max, base_min = dict_mean(df_baseline[1:], num_years*months_in_year)
                base_sd = dict_sd(df_baseline[1:],num_years*12)
                ''' Plot Baseline Required Capacities '''
                plt.figure(figsize=(10, 6),dpi=600)
                bars1= plt.bar(new_month_list[warmup:],base_mu[warmup:], color ="blue", alpha=.8, label='Number of Individuals')
                line1 =plt.plot(new_month_list[warmup:],base_max[warmup:], color ="black", alpha=.8, label= "'95% Predition Interval" )
                line2= plt.plot(new_month_list[warmup:],base_min[warmup:], color ="black", alpha=.8)
                # handles = line1.get_lines() + bars1.containers
                #plt.legend(loc='upper left')
                plt.xlabel('Time')
                plt.xticks(np.arange(0, 300, step=3*12), labels = np.arange(2008,2033,step=3), rotation=45)
                plt.ylabel(ylabel)
                #plt.title(name)
                plt.tight_layout()
                plt.savefig(save_as)                    
                plt.close()
                if number_base == 3:
                    baseline_histcreated == True
                else:
                    number_base += 1
            #monthly mu, sd, err, min, max
            base_mu, base_max, base_min = dict_mean(df_baseline[1:], num_years*12)
            base_sd = dict_sd(df_baseline[1:],num_years*12) 
            errbase = base_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
            #yearly mu, sd, err at month 6
            base_mu_new = [base_mu[i] for i in range(5,len(base_mu), months_in_year)]
            base_sd_new =  [base_sd[i] for i in range(5,len(base_mu), months_in_year)]
            errbase_new = [errbase[i] for i in range(5,len(base_mu), months_in_year)]
            #yearly max, min, and prediction interval errbase
            base_max_new = [max(base_max[i:i+months_in_year]) for i in range(5,len(base_max), months_in_year)]
            basemin_new = [min(base_min[i:i+months_in_year]) for i in range(5,len(base_min), months_in_year)]
            errBASE = []
            errBASE = [(x-y)/2 for x, y in zip(base_max_new, basemin_new)]
            errMidBase = [x+y for x, y in zip(errBASE, basemin_new)]       
            ''' Error bar plot capacilty '''
            fig, ax = plt.subplots(dpi=600)
            #ax.errorbar(new_month_list[warmup:],base_mu[warmup:],errbase[warmup:], elinewidth = 1, capsize=10, color = 'black', alpha=.9)
            ax.errorbar(new_year_list[warmup:],base_mu_new[warmup:],errbase_new[warmup:], elinewidth = 1, capsize=10, color = 'black', alpha=.8)
            for idx,Comparison_Folder in enumerate(comparison_folder_list):
                    df_comparison = pd.read_csv(r''+Comparison_Folder+ csv_filename)
                    df_comparison = df_comparison[0:n]
                    df_comparison = df_comparison.T
                    comp_mu, comp_max, comp_min = dict_mean(df_comparison[1:], num_years*12)
                    comp_sd = dict_sd(df_comparison[1:],num_years*months_in_year)
                    errCOMP = comp_sd/ np.sqrt(n_runs)*z_score #normal mean confidence intervals
                    #yearly mu, sd, err at month 6
                    comp_mu_new = [comp_mu[i] for i in range(5,len(comp_mu), months_in_year)]
                    comp_sd_new =  [comp_sd[i] for i in range(5,len(comp_sd), months_in_year)]
                    errcomp_new = [errCOMP[i] for i in range(5,len(errCOMP), months_in_year)]
                    ax.errorbar(new_year_list[warmup:],comp_mu_new[warmup:],errcomp_new[warmup:], elinewidth = 1, capsize=10, color = color_style[idx],ls=ls_style[idx], alpha=.8)      
                    #ax.errorbar(new_month_list[warmup:],comp_mu[warmup:],errCOMP[warmup:], elinewidth = 1, capsize=10, color = color_style[idx],ls=ls_style[idx], alpha=.9)         
            #legend = ax.legend(leg_labels) 
            plt.xlabel('Year', size = 14)
            plt.xticks(np.arange(2010, 2035, step=5),rotation=45, size = 12)
            plt.yticks( size = 12)
            plt.ylabel(ylabel, size = 14)
            # plt.title("Mean "+name)
            plt.tight_layout()
            plt.savefig(save_as+ "_1.png",bbox_inches = "tight")
            plt.close()
             
            ''' min/ max capacity '''
            fig, ax = plt.subplots(dpi=600) 
            # ax.errorbar(new_year_list[warmup:],errMidBase[warmup:] ,errBASE[warmup:], elinewidth = 1, capsize=10, color ="black", alpha=.9)
            ax.plot(new_year_list[warmup:],base_max_new[warmup:], color ="black", alpha=.8 , label = leg_labels[0])
            for idx,Comparison_Folder in enumerate(comparison_folder_list):
                df_comparison = pd.read_csv(r''+Comparison_Folder+ csv_filename)
                df_comparison = df_comparison[0:n]
                df_comparison = df_comparison.T
                comp_mu, comp_max, comp_min = dict_mean(df_comparison[1:], num_years*months_in_year)
                comp_max_new = [max(comp_max[i:i+months_in_year]) for i in range(0,len(comp_max), months_in_year)]
                comp_min_new = [min(comp_min[i:i+months_in_year]) for i in range(0,len(comp_min), months_in_year)]
                errCOMP = []
                errCOMP = [(x-y)/2 for x, y in zip(comp_max_new, comp_min_new)]
                errMid = [x+y for x, y in zip(errCOMP, comp_min_new)]
                ax.plot(new_year_list[warmup:],comp_max_new[warmup:], color = color_style[idx],ls=ls_style[idx], alpha=.8, label = leg_labels[idx+1], linewidth= 2)
            #plt.legend(loc='upper left')
            plt.xlabel('Year', size = 14)
            plt.xticks(np.arange(2010, 2035, step=5),rotation=45, size = 12)
            plt.yticks( size = 12)
            plt.ylabel(ylabel, size = 14)
            # plt.title("Peak " + name)
            plt.tight_layout()
            plt.savefig(save_as + "_2.png",bbox_inches = "tight" )
            ''' Legend File'''
            # then create a new image 
            # adjust the figure size as necessary 
            figsize = (4.5, 3)
            fig_leg = plt.figure(figsize=figsize)
            ax_leg = fig_leg.add_subplot( 111)
            # add the legend from the previous axes 
            ax_leg.legend(*ax.get_legend_handles_labels(), loc= 'center')
            # hide the axes frame and the x/y labels 
            ax_leg.axis('off')
            fig_leg.tight_layout()
            fig_leg.savefig(save_as +'_legend.png')
            plt.close()       

'''Average Person Rates Table 

table_df = pd.DataFrame(table_list)


table = Avg_PerPersonRatesTable(table_df, policy_dict, year_list, Output_Scenario_TableOrder)
with open('Avg_personResultsTable.txt', 'w') as f:
    f.write(table)         
#'''