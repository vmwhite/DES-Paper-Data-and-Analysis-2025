import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import csv
from Graphing_Functions import *
from Table_Functions import Policy_RegressionTable

output_folder = r''
#Scarerio list AD, OD, CM as variables
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

#Output_files= ['Base_Yearly_ODeaths.csv','Base_Yearly_OArrests.csv', 'Base_Yearly_Hosp.csv' ,'Base_Yearly_Treats.csv','Base_Yearly_Active_YearEnd.csv' ]
Output_files= ['Cum_ODeaths.csv','Cum_OCrimes.csv', 'Cum_Hosp.csv' ,'Cum_Treats.csv', 'Cum_Active_YearEnd.csv' ]
Output_title = ['Opioid-related death', 'Opioid-related arrest', 'Hospital encounters', 'Treatment', 'Active use']

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
                            16:6, 17:17, 18:18}
ScenOrder_table_to_files = {v: k for k, v in ScenOrder_files_to_table.items()}

year_to_index = {
    2032 :25,    2031: 24,    2030:23,
    2029: 22,    2028:21,     2027:20,    2026:19,    2025:18,    2024:17,  2023:16,    2022:15,    2021:14,    2020:13,
    2019:12,    2018:11,    2017:10,    2016:9,     2015:8 
}
# Create the reversed dictionary
index_to_year = {index: year for year, index in year_to_index.items()}

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

#number of scenarios to include
n = 600
##### parameters
months_in_year = 12 
num_years = 25
warmup = 5
actual_warmup = 5
start_year = 2013 - actual_warmup
n_runs = n
year_list = [2023,2027,2032]
years =  [year_to_index[year] for year in year_list]

#Outputs as 5-year and 10 year variables
#HE, active use, arrests, treament capcilty needs., deaths
data = {'AD' : [],
    'CM': [],
    'OD': []}
#Add each scenario to dataframe
for idx_scen, scenario in enumerate(Output_Scenario):
    # Extract the value associated with the key 'AD20'
    data['AD'].append(policy_dict[scenario].split(',')[0].strip())
    data['OD'].append(policy_dict[scenario].split(',')[1].strip())
    data['CM'].append(policy_dict[scenario].split(',')[2].strip())
    #years considered
    for y in years:
        for idx_out, output in enumerate(Output_title):
            out_data_name = f"{y}_year_{str(output)}"
            df_comparison = pd.read_csv(r''+folder_list[idx_scen]+'/summaries600/'+Output_files[idx_out])
            df_comparison = df_comparison[0:n]
            df_comparison = df_comparison.T
            comp_mu, comp_max, comp_min = dict_mean(df_comparison[1:], num_years)
            try:
                data[out_data_name].append(comp_mu[y-1])
            except:
                data[out_data_name]=[]
                data[out_data_name].append(comp_mu[y-1])

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Assuming df is your DataFrame containing the data
df['AD'] = df['AD'].astype(float)
df['OD'] = df['OD'].astype(float)
df['CM'] = df['CM'].astype(float)

# Calculate correlation coefficients
correlation_matrix = df.corr()

# Print correlation matrix
print("Correlation Matrix:", y, " Year")
print(correlation_matrix)
# Save correlation matrix to a CSV file
#correlation_matrix.to_csv('correlation_matrix.csv')


# List to store regression results
regression_results = []
independent_variables = 'AD, OD, CM'  # Define outside the loop

for year in years:
    for idx_out, output in enumerate(Output_title):
        out_data_name = f"{year}_year_{str(output)}"
        # Define independent variables (X) and dependent variable (y)
        if year == 0:
            X =df[['OD']]
        elif year == 5:
            X = df[[ 'OD','AD']]
        else:
            X = df[[ 'OD', 'AD', 'CM']]
        y = df[out_data_name]

        # Add a constant term to the independent variables (X) for intercept
        X = sm.add_constant(X)

        # Fit the regression model
        model = sm.OLS(y, X).fit()

        # Store regression results in a dictionary
        result_dict = {
            'Independent Variables': independent_variables,
            'Output Variable': out_data_name,
            'R-squared': model.rsquared,
            'Coefficients': model.params,
            'P-values': model.pvalues
        }
        regression_results.append(result_dict)

# Convert regression results to DataFrame
results_df = pd.DataFrame(regression_results)

# Save results to CSV
results_df.to_csv('regression_results.csv', index=False)

#Regression Table
reg_table = Policy_RegressionTable(results_df,policy_dict,Output_title, index_to_year, year_list,Output_Scenario_TableOrder)

with open('RegressionTable.txt', 'w') as f:
    f.write(reg_table)         
print("done.")
