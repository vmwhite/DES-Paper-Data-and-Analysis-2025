'''
This file was converted from Results_Tables.RMD
to read in values and create final result latex table.
Before you run this you will need to:
    run Compile Raw Results.py
    change the folder list

'''
import pandas as pd
import numpy as np
from scipy.stats import t

import math
import scipy
def sig(df, idx):
    t_crit =scipy.stats.t.ppf(q=1-(.05/2),df=599)
    if abs(df.iloc[idx]) < t_crit or math.isnan(df.iloc[idx])== True :
        return 'No'
    else:
        return 'Yes'

# Defined Functions ###############################################################################################################
def Add_to_Sum_Table(df, df_old, Scen, Out):
    df2 = df.iloc[:, 61:302].melt(var_name='Month').assign(Scen=Scen).assign(Output=Out).groupby(['Scen', 'Output', 'Month']).agg(
        mean=('value', np.nanmean),
        #se=('value', lambda x: np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x)))),
        # se=('value', lambda x: np.nanstd(x) / np.sqrt(n)),
        sd=('value', lambda x: np.nanstd(x)), #std instead
        lower=('value', lambda x: np.nanmean(x) + t.ppf(0.025, np.sum(~np.isnan(x)) - 1) * np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x)))),
        upper=('value', lambda x: np.nanmean(x) + t.ppf(0.975, np.sum(~np.isnan(x)) - 1) * np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x))))
    ).reset_index()

    df_appended = pd.concat([df_old, df2])

    return df_appended

def Add_to_Yearly_Sum_Table(df, df_appended, Scen, Out, Cost, n):
    df2 = df.iloc[:n, 10:27].melt(var_name='Year', value_name='Total').assign(Scen=Scen).assign(Output=Out).groupby(['Scen', 'Output', 'Year']).agg(
        mean=('Total',  lambda x: np.nanmean(x)),
        #sse=('Total', lambda x: np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x)))),
        # se=('value', lambda x: np.nanstd(x) / np.sqrt(n)),
        sd=('Total', lambda x: np.nanstd(x)), #std instead
        lower=('Total', lambda x: np.nanmean(x) + t.ppf(0.025, np.sum(~np.isnan(x)) - 1) * np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x)))),
        upper=('Total', lambda x: np.nanmean(x) + t.ppf(0.975, np.sum(~np.isnan(x)) - 1) * np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x)))),
        max = ('Total', lambda x: max(x)),
        min=('Total', lambda x: min(x))
    ).reset_index()

    df2['mean_cost'] = df2['mean'] * Cost
    df2['low_cost'] = df2['lower'] * Cost
    df2['high_cost'] = df2['upper'] * Cost
    df2['max_cost'] = df2['max'] * Cost
    df2['min_cost'] = df2['min'] * Cost
    df2['max_count'] = df2['max'] 
    df2['min_count'] = df2['min'] 

    df_appended = pd.concat([df_appended, df2])

    return df_appended
## Event Costs ###############################################################################################################################
DC_to_WI_ratio = 205000 / 36000
A_Cost = (14707 * 0.75 + 183186)/DC_to_WI_ratio
C_Cost = 14707 * 0.25 + 6961
Inflation_rate = 1.19
n = 600


df_Per_Person_Yearly_Costs = {
    'Opioid-Related Death': [11548000],
    'Criminal Justice System': [C_Cost],
    'Hospital Encounters': [14705],
    'Treatment': [1660],
    'Active Use': [A_Cost],
    'Inactive_State': [0],
}
## Things to Update Results ###############################################################################################################
output_folder = r'Results_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023844/summaries600'

folder_list = [r'Results_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_1000_Years_25_Time_033023_023844',
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

CJS_cost_multfolder_list = [1,1,1,1,1,1,.8,.6,.4,.2, 0, 1,.8,1, 1, .6, 1, .4, 1]

Output_files= ['Base_Yearly_ODeaths.csv','Base_Yearly_Arrests.csv', 'Base_Yearly_Hosp.csv' ,'Base_Yearly_Treats.csv','Base_Yearly_Active_YearEnd.csv' ]
Output_title = ['Opioid-Related Death', 'Criminal Justice System', 'Hospital Encounters', 'Treatment', 'Active Use']
Output_Scenario = ['Base', 
                   'CM20','CM40','CM60','CM80','CM100',
                   'AD20','AD40','AD60','AD80','AD100',
                   'OD30','AD20_OD40_CM20', 'OD45','OD60', 'AD40_OD60_CM40', 
                   'OD75', 'AD60_OD80_CM60', 'OD90']
Output_Scenario_TableOrder = ['Base', 
                              'AD20','AD40','AD60','AD80','AD100',
                              'OD30', 'OD45','OD60',  'OD75', 'OD90', 
                              'CM20','CM40','CM60','CM80','CM100',
                              'AD20_OD40_CM20','AD40_OD60_CM40','AD60_OD80_CM60']
ScenOrder_files_to_table = {0:0, 
                            1:11, 2:12, 3:13, 4:14, 5:15, 
                            6:1, 7:2, 8:3, 9:4, 10:5,
                            11:6, 12:16, 13:7, 14:8, 15:17, 
                            16:9, 17:18, 18:10}
ScenOrder_table_to_files = {v: k for k, v in ScenOrder_files_to_table.items()}

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

# Calculating Results ################################################################################################################################################
# Remove the existing df.Yearly if it exists
df_Yearly = None

# Create an empty DataFrame named df_Yearly
df_Yearly = pd.DataFrame()
df_diff = pd.DataFrame()
df_diff_cost = pd.DataFrame()
df_mean_cost = pd.DataFrame()
for f_idx, folder in enumerate(folder_list):
    # Call the Add_to_Yearly_Sum_Table function for each data frame and append the results to df_Yearly
    for idx, file in enumerate(Output_files):
        file_location = folder +'/summaries/'+file
        table_values_raw = pd.read_csv(file_location)
        if Output_title[idx] == "Criminal Justice System":
            multiplier = CJS_cost_multfolder_list[f_idx]
        else:
            multiplier = 1
        ### calculate differences for final year,  2032
        df_diff[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_Mean_2032'] = table_values_raw.iloc[:n,25] 
        #df_diff_cost[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_CostMean_2032'] = table_values_raw.iloc[:n,25] * df_Per_Person_Yearly_Costs[Output_title[idx]]
        df_mean_cost[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_CostMean_2032'] = table_values_raw.iloc[:n,25] * df_Per_Person_Yearly_Costs[Output_title[idx]]*multiplier
        df_diff[Output_Scenario[f_idx]+'_' +Output_title[idx]+'_Diff_2032'] = table_values_raw.iloc[:n,25] - df_diff['Base_'+ Output_title[idx]+'_Mean_2032']
        #df_diff_cost[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_CostDiff_2032'] = (table_values_raw.iloc[:n,25] *  df_Per_Person_Yearly_Costs[Output_title[idx]]) - df_diff_cost['Base_'+ Output_title[idx]+'_CostMean_2032']
        df_Yearly = Add_to_Yearly_Sum_Table(table_values_raw , df_Yearly, Output_Scenario[f_idx], Output_title[idx],
                                        df_Per_Person_Yearly_Costs[Output_title[idx]], n)
        
        #for cost validaion in 2017
        df_diff[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_Mean_2017'] = table_values_raw.iloc[:n,10]
        #df_diff_cost[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_CostMean_2017'] = table_values_raw.iloc[:n,10] * df_Per_Person_Yearly_Costs[Output_title[idx]]
        df_mean_cost[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_CostMean_2017'] = table_values_raw.iloc[:n,10] * df_Per_Person_Yearly_Costs[Output_title[idx]]*multiplier
        df_diff[Output_Scenario[f_idx]+'_' +Output_title[idx]+'_Diff_2017'] = table_values_raw.iloc[:n,10] - df_diff['Base_'+ Output_title[idx]+'_Mean_2017']
        #df_diff_cost[Output_Scenario[f_idx]+'_'+Output_title[idx]+ '_CostDiff_2017'] = (table_values_raw.iloc[:n,10] *  df_Per_Person_Yearly_Costs[Output_title[idx]]) - df_diff_cost['Base_'+ Output_title[idx]+'_CostMean_2017']
        df_Yearly = Add_to_Yearly_Sum_Table(table_values_raw , df_Yearly, Output_Scenario[f_idx], Output_title[idx],
                                        df_Per_Person_Yearly_Costs[Output_title[idx]], n)
df_Yearly.to_csv(output_folder+"\All_yearly_values.csv", index=False)
#total cost calculations for each scenario # per scenario
df_all = df_diff
df_total_costs = pd.DataFrame()
for t_idx, line in enumerate(Output_Scenario_TableOrder):
    f_idx = ScenOrder_table_to_files[t_idx]
    df_total_costs["Total_2017_"+Output_Scenario[f_idx]+"_Mean_Cost"]= df_all.apply(lambda row: df_Per_Person_Yearly_Costs['Opioid-Related Death'][0]*row[2+(f_idx*(4+16))] 
                                                                               + df_Per_Person_Yearly_Costs['Criminal Justice System'][0]*row[6+(f_idx*(4+16))] *CJS_cost_multfolder_list[f_idx]
                                                                               + df_Per_Person_Yearly_Costs['Hospital Encounters'][0]*row[10+(f_idx*(4+16))]
                                                                                 + df_Per_Person_Yearly_Costs['Treatment'][0]*row[14+(f_idx*(4+16))] 
                                                                                 + df_Per_Person_Yearly_Costs['Active Use'][0]*row[18+(f_idx*(4+16))], axis=1 )
    df_total_costs["Total_2017_"+Output_Scenario[f_idx]+"_Diff_Cost"]= df_all.apply(lambda row: df_Per_Person_Yearly_Costs['Opioid-Related Death'][0]*row[3+(f_idx*(4+16))] 
                                                                               + df_Per_Person_Yearly_Costs['Criminal Justice System'][0]*(row[7+(f_idx*(4+16))] + row[6+(f_idx*(4+16))] - (CJS_cost_multfolder_list[f_idx]*row[6+(f_idx*(4+16))]))
                                                                               + df_Per_Person_Yearly_Costs['Hospital Encounters'][0]*row[11+(f_idx*(4+16))]
                                                                                 + df_Per_Person_Yearly_Costs['Treatment'][0]*row[15+(f_idx*(4+16))] 
                                                                                 + df_Per_Person_Yearly_Costs['Active Use'][0]*row[19+(f_idx*(4+16))], axis=1 )
    df_total_costs["Total_2032_"+Output_Scenario[f_idx]+"_Mean_Cost"]= df_all.apply(lambda row: df_Per_Person_Yearly_Costs['Opioid-Related Death'][0]*row[0+(f_idx*(4+16))] 
                                                                               + df_Per_Person_Yearly_Costs['Criminal Justice System'][0]*row[4+(f_idx*(4+16))] *CJS_cost_multfolder_list[f_idx]
                                                                                 + df_Per_Person_Yearly_Costs['Hospital Encounters'][0]*row[8+(f_idx*(4+16))] 
                                                                                 + df_Per_Person_Yearly_Costs['Treatment'][0]*row[12+(f_idx*(4+16))] 
                                                                                 + df_Per_Person_Yearly_Costs['Active Use'][0]*row[16+(f_idx*(4+16))], axis=1 )
    df_total_costs["Total_2032_"+Output_Scenario[f_idx]+"_Diff_Cost"]= df_all.apply(lambda row: df_Per_Person_Yearly_Costs['Opioid-Related Death'][0]*row[1+(f_idx*(4+16))] 
                                                                               + df_Per_Person_Yearly_Costs['Criminal Justice System'][0]*(row[5+(f_idx*(4+16))]-row[4+(f_idx*(4+16))]  +(row[4+(f_idx*(4+16))] *CJS_cost_multfolder_list[f_idx]))
                                                                                 + df_Per_Person_Yearly_Costs['Hospital Encounters'][0]*row[9+(f_idx*(4+16))] 
                                                                                 + df_Per_Person_Yearly_Costs['Treatment'][0]*row[13+(f_idx*(4+16))] 
                                                                                 + df_Per_Person_Yearly_Costs['Active Use'][0]*row[17+(f_idx*(4+16))], axis=1 )

df_cost_new =df_total_costs.agg(['mean', 'std', 'min', 'max'], axis=0)
df_cost_new.loc['cost_se'] = df_cost_new.apply((lambda row: (row[1] / np.sqrt(n))), axis=0)
df_cost_new.loc['cost_t_score'] = df_cost_new.apply((lambda row: row[0]/row[4]), axis=0)
df_cost_new.loc['cost_p_value'] = df_cost_new.apply((lambda row:  '$<$0.001' if  scipy.stats.t.sf(abs(row[5]),n-1)*2 < 0.001 and math.isnan(row[5]) == False else str(round(scipy.stats.t.sf(abs(row[5]),n-1)*2,3))), axis=0) 
df_cost_new.loc['SignificantCost?'] = df_cost_new.apply((lambda row: 'Yes' if  scipy.stats.t.sf(abs(row[5]),n-1)*2 < 0.05 and math.isnan(row[5]) == False else ( 'Yes1' if scipy.stats.t.sf(abs(row[5]),n-1)*2 < 0.1 and math.isnan(row[5]) == False else 'No')),axis=0) 
df_cost_new = df_cost_new = df_cost_new.T

#Obtain mean difference of total counts # per event type
df_diff_new = df_diff.agg(['mean', 'std'], axis=0)
df_diff_new.loc["mean_se"] = df_diff_new.apply((lambda row: (row[1] / np.sqrt(n))), axis=0)
df_diff_new.loc["mean_t_score"] = df_diff_new.apply((lambda row: (row[0] /row[2])), axis=0)
df_diff_new.loc["mean_p-value"] = df_diff_new.apply((lambda row:  '$<$0.001' if  scipy.stats.t.sf(abs(row[3]),n-1)*2 < 0.001 and math.isnan(row[3]) == False else str(round(scipy.stats.t.sf(abs(row[3]),n-1)*2,3))), axis=0) 
df_diff_new.loc['SignificantMean?']  = df_diff_new.apply((lambda row: 'Yes' if  scipy.stats.t.sf(abs(row[3]),n-1)*2 < 0.05 and math.isnan(row[3]) == False else ( 'Yes1' if scipy.stats.t.sf(abs(row[3]),n-1)*2 < 0.1 and math.isnan(row[3]) == False else 'No')),axis=0) 
df_diff_new = df_diff_new.T


df_2032_costs =  df_cost_new.T[[col for col in df_cost_new.T if '2032' in col]]
df_2017_costs =  df_cost_new.T[[col for col in df_cost_new.T if '2017' in col]]

df_2017_costs.to_csv(output_folder+"\costs_2017.csv", index=False, sep=";")
df_2032_costs.to_csv(output_folder+"\costs_2032.csv", index=False, sep=";")
df_total_costs.to_csv(output_folder+"\costs_total.txt", index=False, sep=";")
df_all.to_csv(output_folder+"\costs_all.txt", index=False, sep=";")

### Gernating Main_Results_Table in Latex ###############################################################################################################################
from Table_Functions import Combined_Results_Table
# using functions to create a new columns for SUMMED COST TABLE
table = Combined_Results_Table(df_cost_new, Output_Scenario_TableOrder,policy_dict, df_diff_new, Output_title,ScenOrder_table_to_files)
with open('CombinedResultsTable.txt', 'w') as f:
    f.write(table)