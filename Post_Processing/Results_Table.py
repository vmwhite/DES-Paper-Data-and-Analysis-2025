'''
This file reads in values from .csv files and create final result latex table.
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
n = 2
df_Per_Person_Yearly_Costs = {
    'Opioid-Related Death': [11548462],
    'Opioid-Related Arrests, non-Diverted': [55726],
    'Hospital Encounters': [20077],
    'Treatment': [1812],
    'Active Use': [34106],
    'Inactive_State': [0],
}
## Things to Update Results ###############################################################################################################
output_folder = r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606'

folder_list = [r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606',
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

Output_files= ['Base_Yearly_ODeaths.csv','Base_Yearly_OCrimes.csv', 'Base_Yearly_Hosp.csv' ,'Base_Yearly_Treats.csv', 'Base_Yearly_Active_YearEnd.csv' ,'Cum_ODeaths.csv','Cum_OCrimes.csv', 'Cum_Hosp.csv' ,'Cum_Treats.csv', 'Cum_Active_YearEnd.csv' ]
Output_title = ['Opioid-Related Death', 'Opioid-Related Arrests', 'Hospital Encounters', 'Treatment','Active Use','Cumulative Opioid-Related Death', 'Cumulative Opioid-Related Arrests', 'Cumulative Hospital Encounters', 'Cumulative Treatment','Cumulative Active Use'] # 'Prevalence']

Output_Scenario = ['Base', 
                   'OD30','OD45','OD60','OD75', 'OD90',
                   'AD20','AD40','AD60','AD80','AD100',
                   'CM20','CM40','CM60','CM80','CM100',
                   'AD20_OD40_CM20',  'AD40_OD60_CM40', 
                    'AD60_OD80_CM60']

Output_Scenario_TableOrder = ['Base', 
                              'AD20','AD40','AD60','AD80','AD100',
                              'OD30', 'OD45','OD60',  'OD75', 'OD90', 
                              'CM20','CM40','CM60','CM80','CM100',
                              'AD20_OD40_CM20','AD40_OD60_CM40','AD60_OD80_CM60']
ScenOrder_files_to_table = {0:0, 
                            1:6, 2:7, 3:8, 4:9, 5:10, 
                            6:1, 7:2, 8:3, 9:4, 10:5,
                            11:11, 12:12, 13:13, 14:14, 15:15, 
                            16:16, 17:17, 18:18}
ScenOrder_table_to_files = {v: k for k, v in ScenOrder_files_to_table.items()}

year_to_index = {
    2032 :25,    2031: 24,    2030:23,
    2029: 22,    2028:21,     2027:20,    2026:19,    2025:18,    2024:17,  2023:16,    2022:15,    2021:14,    2020:13,
    2019:12,    2018:11,    2017:10,    2016:9,     2015:8 
}

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

# Function to find the last value in a tuple based on other items
def find_last_value(data, item1, item2, item3, item4, item5):
    for tup in data:
        if tup[0] == item1 and tup[1] == item2 and tup[2] ==item3 and tup[3] ==item4 and tup[4] ==item5:
            return tup[-1]  # Return the last item in the tuple

years = [2017, 2023, 2027, 2032]
n= 600
# Calculating Results ###############################################################################################################################################
#'''
# Create an empty DataFrame named df_Yearly
data =[]
print("... gathering data ...")
# Iterate over folders and files
for f_idx, folder in enumerate(folder_list):
    for idx, file in enumerate(Output_files):
        file_location = f"{folder}/summaries{n}/{file}"
        table_values_raw = pd.read_csv(file_location)
        file2_location = f"{folder}/summaries{n}/Cum_OArrests.csv"
        table_values_Arrests_raw = pd.read_csv(file2_location)
        file3_location = f"{folder}/summaries{n}/Base_Yearly_OArrests.csv"
        table_values_ArrestsYear_raw = pd.read_csv(file3_location)
        scenario = Output_Scenario[f_idx]
        for year in years:
            for run in range(0,n):
                # Calculate differences for each year
                value = table_values_raw.iloc[run, year_to_index[year]]
                if 'Cum_OCrime' in file:
                    value = table_values_Arrests_raw.iloc[run, year_to_index[year]]
                    cost = value * df_Per_Person_Yearly_Costs["Opioid-Related Arrests, non-Diverted"][0]
                elif 'OCrime' in file:
                    value = table_values_ArrestsYear_raw.iloc[run, year_to_index[year]]
                    cost = value * df_Per_Person_Yearly_Costs["Opioid-Related Arrests, non-Diverted"][0]
                else:
                    if 'ODeath' in file:
                        output_title = 'Opioid-Related Death'
                    if 'Hosp' in file:
                        output_title = 'Hospital Encounters'
                    if 'Treat' in file:
                        output_title = 'Treatment'
                    if 'Active' in file:
                        output_title ='Active Use' 
                    cost = value * df_Per_Person_Yearly_Costs[output_title][0]
                data.append([scenario, run, file, 'Value', year, value])
                data.append([scenario, run, file, 'CostValue', year, cost])
                if scenario != "Base":
                    Base_val = find_last_value(data, "Base", run, file, "Value", year)
                    diff_value = Base_val - value
                    data.append([scenario, run, file,'Diff', year, diff_value])
                    Base_cost = find_last_value(data, "Base", run, file, "CostValue", year)
                    diff_cost = Base_cost - cost
                    data.append([scenario, run, file, 'DiffCost', year, diff_cost])

# Construct MultiIndex for columns
columns = ['Scenario', 'Run', 'file', 'Type', 'Year', 'Value']
# Create DataFrame
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_folder+"\All_raw_data.csv", index=False)
print(f'Printed raw data to {output_folder} \All_raw_data.csv')
#conduct paired t-test on main outcome values:
df_runs = pd.DataFrame(columns=['Scenario', 'Year', 'Type', 'file', 'mean', 'std', 'se', 'min', 'max', 't_stat', 'p_val'])
row = 0
for scenario in df['Scenario'].unique():
    for year in df['Year'].unique():
        for output in df['file'].unique():
            for type in df['Type'].unique():
                    group1= df[(df['Scenario'] == scenario) & (df['Year'] == year) & (df['Type'] == type)& (df['file'] == output)]['Value']
                    group_base = df[(df['Scenario'] == "Base") & (df['Year'] == year) & (df['Type'] == type)& (df['file'] == output)]['Value']
                    df_runs.loc[row, 'Scenario'] = scenario
                    df_runs.loc[row, 'Year'] = year
                    df_runs.loc[row, 'file'] = output
                    df_runs.loc[row, 'Type'] = type
                    df_runs.loc[row, 'mean'] = np.mean(group1)
                    df_runs.loc[row, 'std'] = np.std(group1)
                    df_runs.loc[row, 'se'] = np.std(group1) / np.sqrt(n)
                    df_runs.loc[row, 'max'] = np.max(group1)
                    df_runs.loc[row, 'min'] = np.min(group1)
                    if scenario == 'Base':
                        row += 1
                        continue
                    # Perform paired t-test
                    try:
                        t_statistic, p_value = scipy.stats.ttest_rel(group1, group_base)
                    except:
                         group_base = [0]*n
                         t_statistic, p_value = scipy.stats.ttest_rel(group1, group_base)
                    df_runs.loc[row, 't_stat'] = t_statistic
                    df_runs.loc[row, 'p_val'] = p_value
                    row += 1

df_runs['p_value'] = df_runs.apply((lambda row:  '$<$0.001' if row['p_val'] < 0.001 and math.isnan(row['p_val']) == False else str(round(row['p_val'],3))), axis=1) 
df_runs['Significant_p_val?'] = df_runs.apply((lambda row: 'Yes' if  row['p_val'] < 0.05 and math.isnan(row['p_val']) == False else ( 'Yes1' if row['p_val'] < 0.1 and math.isnan(row['p_val']) == False else 'No')),axis=1) 
df_runs.to_csv(output_folder+"\Allscen_run_summary.csv", index=False)
print(f'Printed combined run summary to {output_folder} \Allscen_run_summary.csv')

filter_condition = (df['Type'] == 'CostValue') | (df['Type'] == 'CostDiff')
filtered_df = df[filter_condition]

# Group by columns and sum the 'Value' column
df_costs =  filtered_df.groupby(['Scenario', 'Run', 'Type', 'Year']).agg({'Value':'sum'}).reset_index()
df_costs["se"] = df_costs.apply(lambda row: (row[3] / np.sqrt(n)) if pd.notnull(row[3]) else np.nan, axis=1)
for scenario in df['Scenario'].unique():
    for year in df['Year'].unique():
        for output in df['file'].unique():
            for type in df['Type'].unique():
                    group1= df_costs[(df_costs['Scenario'] == scenario) & (df_costs['Year'] == year) & (df_costs['Type'] == type)]['Value']
                    group_base = df_costs[(df_costs['Scenario'] == "Base") & (df_costs['Year'] == year) & (df_costs['Type'] == type)]['Value']
                    if scenario == 'Base':
                        df_costs.loc[(df_costs['Scenario'] == scenario) & (df_costs['Year'] == year) & (df_costs['Type'] == type), 't_stat'] = np.nan
                        df_costs.loc[(df_costs['Scenario'] == scenario) & (df_costs['Year'] == year) & (df_costs['Type'] == type), 'p_val'] = np.nan
                        continue
                    # Perform paired t-test
                    try:
                        t_statistic, p_value = scipy.stats.ttest_rel(group1, group_base)
                    except:
                         group_base = [0]*n
                         t_statistic, p_value = scipy.stats.ttest_rel(group1, group_base)
                    df_costs.loc[(df_costs['Scenario'] == scenario) & (df_costs['Year'] == year) & (df_costs['Type'] == type), 't_stat'] = t_statistic
                    df_costs.loc[(df_costs['Scenario'] == scenario) & (df_costs['Year'] == year) & (df_costs['Type'] == type), 'p_val'] = p_value
df_costs =  df_costs.groupby(['Scenario','Type', 'Year','t_stat', 'p_val'], dropna=False).agg({'Value':('mean','count','std', 'min', 'max' )}).reset_index()
# Flatten the column names
df_costs.columns = ['Scenario', 'Type', 'Year', 't_stat', 'p_val', 'Mean_Value', 'Count', 'Std', 'Min', 'Max']
df_costs['cost_p_value'] = df_costs.apply((lambda row:  '$<$0.001' if row['p_val'] < 0.001 and math.isnan(row['p_val']) == False else str(round(row['p_val'],3))), axis=1) 
df_costs['SignificantCost?'] = df_costs.apply((lambda row: 'Yes' if  row['p_val'] < 0.05 and math.isnan(row['p_val']) == False else ( 'Yes1' if row['p_val'] < 0.1 and math.isnan(row['p_val']) == False else 'No')),axis=1) 
df_costs.to_csv(output_folder+"\Allscen_total_cost_summary.csv", index=False)
print(f'Printed cost summary to {output_folder} \Allscen_total_cost_summary.csv')
#'''
### Gernating Main_Results_Table in Latex ###############################################################################################################################
from Table_Functions import Combined_Results_Table
df_costs = pd.read_csv(output_folder+"\Allscen_total_cost_summary.csv")
df_runs =pd.read_csv(output_folder+"\Allscen_run_summary.csv")
# using functions to create a new columns for SUMMED COST TABLE
year = 2032 # ten year outcomes
table = Combined_Results_Table(df_costs, Output_Scenario_TableOrder,policy_dict, df_runs, Output_title,ScenOrder_table_to_files,year)
with open('CombinedResultsTable.txt', 'w') as f:
    f.write(table)
#'''