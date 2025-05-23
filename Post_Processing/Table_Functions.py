import pandas as pd        
import math
from Graphing_Functions import *
import numpy as np
######################################## fxns ################################################
def sig2_df(df):
    if( 0 in range(math.floor(df['Add_low_costs']), math.ceil(df['Add_high_costs']))):
        return 'No'
    else:
        return 'Yes'
def Sum_calcs(df_2032_costs2):
    # Perform the calculations and formatting
    df_2032_costs2['mean_Cost_low'] = df_2032_costs2.apply(lambda row:  f"\${round(row[2] / 1000000, 0):,}", axis = 1)
    df_2032_costs2['mean_Cost_high'] =  df_2032_costs2.apply(lambda row:  f"\${round(row[3] / 1000000, 0):,}", axis = 1)


    # Perform the calculations and formatting
    df_2032_costs2['mean_Saving_high'] = df_2032_costs2.apply(lambda row:  f"\${round((row[5] / 1000000)*-1, 0):,}", axis = 1)
    df_2032_costs2['mean_Saving_low'] =  df_2032_costs2.apply(lambda row: f"\${round((row[6] / 1000000)*-1, 0):,}", axis = 1)

    # Concatenate the formatted values
    df_2032_costs2['mean_Cost_interval'] = df_2032_costs2.apply(lambda row: f"({row[7]}, {row[8]})", axis = 1)
    df_2032_costs2['mean_Saving_interval'] =  df_2032_costs2.apply(lambda row: f"({row[10]}, {row[9]})", axis = 1)

    df = df_2032_costs2.drop(df_2032_costs2.iloc[:,1:11], axis=1)
    return df

def add_top_column(df, idx,  Output_title):
        df = df.T[[col for col in df.T if Output_title[idx] in col]]
        return df

def latex_row(df_name, row):
    backslash_char = "\\"
    if df_name == "cost" :
        if row['p_val'] < 0.001:
                value = backslash_char +'bf{' + f"{round((row['Mean_Value'])/1000000, 0):,}" + '$' + backslash_char+'pm$ ' + str(round((row['Std']/np.sqrt(600))/1000000, 2)) +'$^{**}$ ('  + str(row['cost_p_value']) +')}'
        elif row['SignificantCost?'] == 'Yes':
            value = backslash_char +'bf{' + f"{round((row['Mean_Value'])/1000000, 0):,}" + '$' + backslash_char+'pm$ ' + str(round((row['Std']/np.sqrt(600))/1000000, 2)) +'$^*$ ('  + str(row['cost_p_value']) +')}'
        elif (row['p_val']) == np.nan:
            value = f"{round((row['Mean_Value'])/1000000, 0):,}"+ ' $' + backslash_char+'pm$ ' + str((round(row['Std']/np.sqrt(600)/1000000, 2)))
        else:
            value = f"{round((row['Mean_Value'])/1000000, 0):,}" + ' $' + backslash_char+'pm$ ' + str((round(row['Std']/np.sqrt(600)/1000000, 2))) +'('  + str(row['cost_p_value']) +')'
    elif df_name == "diff":
        if row['p_val'] < 0.001:
                value = '$^{**}$ ('  + str(row['p_value']) +')}'
        elif row['Significant_p_val?'] == 'Yes':
            value = '$^*$ ('  + str(row['p_value']) +')}'
        elif (row['p_val']) == np.nan:
            value = ""
        else:
            value = '('  + str(row['p_value']) +')'

    elif df_name == "val": 
        if row["Scenario"] == "Base":
            value = str(round(row['mean'], 2)) + ' $' + backslash_char+'pm$ ' + str(round(row['se'], 2))
        elif row['p_val'] < 0.001:
                value = backslash_char +'bf{' + str(round((row['mean']), 2)) + '$' + backslash_char+'pm$ ' + str(round(row['se'], 2)) 
        elif row['Significant_p_val?'] == 'Yes':
            value = backslash_char +'bf{' + str(round((row['mean']), 2)) + '$' + backslash_char+'pm$ ' + str(round(row['se'], 2))
        elif (row['p_val']) == np.nan:
            value = str(round(row['mean'], 2)) + ' $' + backslash_char+'pm$ ' + str(round(row['se'], 2))
        else:
            value = str(round(row['mean'], 2)) + ' $' + backslash_char+'pm$ ' + str((round(row['se'], 2)))
    return value

######################################## latex tables ################################################################################################################
############ All results table ###########################################################################
def Combined_Results_Table(df_costs, Output_Scenario_TableOrder,policy_dict, df, Output_title,ScenOrder_table_to_files,year):
    backslash_char = "\\"
    df_costs = df_costs[df_costs["Year"]==year]
    df_costs['Latex'] = df_costs.apply(lambda row: latex_row("cost",row), axis=1)
    df_diff = df[(df["Year"]==year) & (df['Type']=='Diff')]
    df_diff['Latex'] = df_diff.apply(lambda row: latex_row("diff",row), axis=1)
    df_val = df[(df["Year"]==year) & (df['Type']=='Value')]
    df_val['Latex'] = df_val.apply(lambda row: latex_row("val",row), axis=1)

    ############################ Create Latex Table ############################################
    table = f"{{\spacingset{{1.5}}"
    table +=f"\n\\begin{{sidewaystable}}"
    table +=f"\n\caption{{Simulated mean $\pm$ standard error (p-value) of cumulative total events, cost, and savings for scenarios after {year-2022} years }}"
    table +=f"\n\label{{tab:AD_All_Results}}"
    table +=f"\n\\resizebox{{\\textwidth}}{{!}}{{"
    table +=f"\n\\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|}}"
    table +=f"\n\hline"
    table +=f"\n\multicolumn{{2}}{{|c|}}{{Scenario}} & \multicolumn{{1}}{{c|}}{{Opioid-Related Deaths}} & \multicolumn{{1}}{{c|}}{{Opioid-Related Non-Diverted Arrests}} & \multicolumn{{1}}{{c|}}{{Opioid-Related Hospital Encounters}} & \multicolumn{{1}}{{c|}}{{OUD Treatment Starts}} & \multicolumn{{1}}{{c|}}{{Active Use Starts}} & \multirow{{1}}{{*}}{{Mean Total Cost}} & \multirow{{1}}{{*}}{{Mean Cost  Difference}}\\\\ \hline"
    table +=f"\n\multicolumn{{2}}{{|c|}}{{AD (\%), OD (\%), CM (\%)}} & \multicolumn{{1}}{{c|}}{{mean $\pm$ se (p-value)}}  & \multicolumn{{1}}{{c|}}{{mean $\pm$ se (p-value)}}  & \multicolumn{{1}}{{c|}}{{mean $\pm$ se (p-value)}}  & \multicolumn{{1}}{{c|}}{{mean $\pm$ se (p-value)}}  & \multicolumn{{1}}{{c|}}{{mean $\pm$ se (p-value)}} & \\$ in Millions (p-value) &  \\$ in Millions  \\\\ \hline \hline"

    i = 2
    for idx, scen in enumerate(Output_Scenario_TableOrder):
        idx_f = ScenOrder_table_to_files[idx]
        if scen == 'Base':
            table += f"\n\multirow{{{i}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{8mm}}{{\spacingset{{1}} \centering Base \\\\ Model}}}}}}"
        elif scen == 'AD20':
            table += f"\hline \hline"
            table += f"\n\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{15mm}}{{\spacingset{{1}}\centering Arrest \\\\ Diversion}}}}}}"
        elif scen =='OD30':
            table += f"\hline \hline"
            table += f"\n\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{15mm}}{{\spacingset{{1}}\centering Overdose \\\\ Diversion}}}}}}"
        elif scen == 'CM20':
            table += f"\hline \hline"
            table += f"\n\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{20mm}}{{\spacingset{{1}}\centering Case \\\\ Management}}}}}}"
        elif scen == 'AD20_OD40_CM20':
            table += f"\hline \hline"
            table += f"\n\multirow{{3}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{10mm}}{{\spacingset{{1}}\centering Policy \\\\ Mix}}}}}}"
        
        table += f" & \multirow{{{i}}}{{*}}{{{policy_dict[scen]}}} & \multirow{{{i}}}{{*}}{{{df_val[(df_val['Scenario']==scen) &(df_val['file']=='Cum_ODeaths.csv')].iloc[0, -1] + df_diff[(df_diff['Scenario']==scen) &(df_diff['file']=='Cum_ODeaths.csv')].iloc[0, -1]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_val[(df_val['Scenario']==scen) &(df_val['file']=='Cum_OCrimes.csv')].iloc[0, -1]+df_diff[(df_diff['Scenario']==scen) &(df_diff['file']=='Cum_OCrimes.csv')].iloc[0, -1]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_val[(df_val['Scenario']==scen) &(df_val['file']=='Cum_Hosp.csv')].iloc[0, -1]+df_diff[(df_diff['Scenario']==scen) &(df_diff['file']=='Cum_Hosp.csv')].iloc[0, -1]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_val[(df_val['Scenario']==scen) &(df_val['file']=='Cum_Treats.csv')].iloc[0, -1]+df_diff[(df_diff['Scenario']==scen) &(df_diff['file']=='Cum_Treats.csv')].iloc[0, -1]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_val[(df_val['Scenario']==scen) &(df_val['file']=='Cum_Active_YearEnd.csv')].iloc[0, -1]+df_diff[(df_diff['Scenario']==scen) &(df_diff['file']=='Cum_Active_YearEnd.csv')].iloc[0, -1]}}}"
        #for calling from df_diff_cost_new
        table += f" & \multirow{{{i}}}{{*}}{{{df_costs[(df_costs['Scenario']==scen)].iloc[0, -1]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{"
        if df_costs[(df_costs['Scenario']==scen)]["p_val"].iloc[0] <0.05:
            table +=f"{backslash_char}bf{{"
        table += f"{round((df_costs[(df_costs['Scenario']==scen)]['Mean_Value'].iloc[0] - df_costs[(df_costs['Scenario']=='Base')]['Mean_Value'].iloc[0])/1000000,2)}}}"
        if df_costs[(df_costs['Scenario']==scen)]["p_val"].iloc[0] <0.05:
            table +=f"}}"
        if scen =="Base":    
            table +=f"\n\\\\ &&&&&&&&\\\\"
            i -= 1
        else: 
            table +=f"\n\\\\"
        
        
    table += f" \hline"
    table +=f"\n\multicolumn{{9}}{{l}}{{*{{Statistically significant difference from the Base Model at a level of 0.05 using 2 -sided paired t-test}}}}\\\\"
    table +=f"\n\multicolumn{{9}}{{l}}{{*{{Note: Overlapping confidence intervals that are marked statistically significant through a paired t-test can be attributed to Type one error \citep{{knol_misuse_2011}} }}}}\\\\"
    table +=f"\n\end{{tabular}} "
    table +=f"\n}}"
    table +=f"\n\end{{sidewaystable}}"
    table +=f"\n}}"
    print(table)
    
    return table
##############  Avg_PerPerson Table  ################################################################################
def Avg_PerPersonRatesTable(df_table, policy_dict, year_list, Output_Scenario_TableOrder):
    #df_table columns: scen, csv_filename, year, comp_mu[year - start_year], errCOMP[year - start_year]/z_score, comp_max[year - start_year], comp_min[year - start_year], df_diff.iloc[4,year - start_year] ,df_diff.iloc[5,year - start_year]])
    df_table.to_csv("Avg_personResultsTable.csv",sep=',', index=False, encoding='utf-8')
    backslash_char = "\\"
    table = f"{{\spacingset{{1.5}}"
    table += f"\n\\begin{{sidewaystable}}[htbp]"
    table += f"\n\centering"
    table +=f"\n\caption{{Simulated mean $\pm$ standard error (p-value) of Year 2032 opioid-related re-arrest, hospitalization, and treatment re-start rates for all scenarios}}"

    table +=f"\n\label{{tab:perPerson_Results}}"
    table +=f"\n\\resizebox{{\\textwidth}}{{!}}{{"
    table +=f"\n\\begin{{tabular}}{{|c|"
    for c in range(3*len(year_list)):
        table +=f"c|c|"
    table +=f"}}"
    table +=f"\n\hline"
    table +=f"\n\multicolumn{{2}}{{|c|}}{{Scenario}} & \multicolumn{{{len(year_list)}}}{{c|}}{{Re-Hospitalisation rate}} & \multicolumn{{{len(year_list)}}}{{c|}}{{Re-Arrest rate}} & \multicolumn{{{len(year_list)}}}{{c|}}{{OUD Treatment Re-Start rate}} \\\\ \hline"
    table +=f"\n\multicolumn{{2}}{{|c|}}{{\multirow{{2}}{{*}}{{AD (\%), OD (\%), CM (\%)}}}} & \multicolumn{{{len(year_list)}}}{{c|}}{{mean (\%) $\pm$ se (p-value)}} & \multicolumn{{{len(year_list)}}}{{c|}}{{mean (\%) $\pm$ se (p-value)}} & \multicolumn{{{len(year_list)}}}{{c|}}{{mean (\%) $\pm$ se (p-value)}}  \\\\   \cline{{3-11}}"
    table +=f"\n\multicolumn{{2}}{{|c|}}{{}} "
    for i in range(len(year_list)):
        for year in year_list:
            table += f" & {year}"
    table += f"\\\\ \hline \hline"
    ### Add Base Row ###
    table += f"\n\multirow{{{2}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{8mm}}{{\spacingset{{1}} \centering Base \\\\ Model}}}}}}"
    ### ADD sCENARIO Rows ###
    for idx, scen in enumerate(Output_Scenario_TableOrder):
        if scen == 'AD20':
            table += f"\hline \hline"
            table += f"\n\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{15mm}}{{\spacingset{{1}}\centering Arrest \\\\ Diversion}}}}}}"   
        elif scen =='OD30':
            table += f"\hline \hline"
            table += f"\n\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{15mm}}{{\spacingset{{1}}\centering Overdose \\\\ Diversion}}}}}}"
        elif scen == 'CM20':
            table += f"\hline \hline"
            table += f"\n\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{20mm}}{{\spacingset{{1}}\centering Case \\\\ Management}}}}}}"
        elif scen == 'AD20_OD40_CM20':
            table += f"\hline \hline"
            table += f"\n\multirow{{3}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\parbox[t]{{10mm}}{{\spacingset{{1}}\centering Policy \\\\ Mix}}}}}}"
        table += f" & \multirow{{{1}}}{{*}}{{{policy_dict[scen]}}}"
        for idx in range(len(df_table)):
            for year in year_list:
                if df_table.iloc[idx][2] == year and df_table.iloc[idx][0] == scen:
                    table += f" &"
                if df_table.iloc[idx][8] == 'yes' and df_table.iloc[idx][2] == year and df_table.iloc[idx][0] == scen:
                    table += f"\\bf{{"
                if df_table.iloc[idx][2] == year and df_table.iloc[idx][0] == scen:
                    table += f" {round(df_table.iloc[idx][3],2)} $\pm$ {round(df_table.iloc[idx][4],3)}"
                if df_table.iloc[idx][8] == 'yes' and df_table.iloc[idx][2] == year and df_table.iloc[idx][0] == scen:
                    if df_table.iloc[idx][7] ==  '$<0.001$':
                        table += f"$^{{**}}$}}"  
                    else:
                        table += f"$^{{*}}$}}"  
        table += f"\\\\\n"
    table += f"\n \hline "
    table +=f"\n\multicolumn{{11}}{{l}}{{**{{Statistically significant difference, with p-value $<0.001$, from the Base Model at a level of 0.05 using 2-sided paired t-test}}}}\n"
    table += f"\end{{tabular}} \n }}  "
    table += f"\n \\end{{sidewaystable}} \n }}"

    return table

######### Policy regression / comparison table ###########################################################################################
def Policy_RegressionTable(df_table, policy_dict, output_list, index_to_year,year_list):
    backslash_char = "\\"
    table = f"{{\spacingset{{1.5}}"
    table += f"\n\\begin{{table}}[htbp]"
    table += f"\n\centering"
    table +=f"\n\caption{{OLS regression of policies vs. model outputs post policy implementation}}"
    table +=f"\n\label{{tab:policycorr}}"
    table +=f"\n\\resizebox{{\\textwidth}}{{!}}{{"
    table +=f"\n\\color{{blue}}\\begin{{tabular}}"
    table +=f"{{|c|c|c|c|c|c|}}"
    table +=f"\n\hline"
    table +=f"\n\multicolumn{{1}}{{|c|}}{{\multirow{{2}}{{*}}{{Year}}}} & \multicolumn{{1}}{{c|}}{{\multirow{{2}}{{*}}{{Model Output}}}} & \multicolumn{{{4}}}{{c|}}{{Regression Coefficients (p-value)}}\\\\  \cline{{3-6}}  "
    table +=f"\n & & \multicolumn{{1}}{{c|}}{{Intercept (p-value)}} & \multicolumn{{1}}{{c|}}{{AD (p-value)}} & \multicolumn{{1}}{{c|}}{{OD (p-value)}}  & \multicolumn{{1}}{{c|}}{{CM (p-value)}}  \\\\  "
    table += f"\hline \hline"
    for idx in range(0,len(df_table)):
        try:
            year_idx = int(df_table.iloc[idx][1][0:2])
        except:
            year_idx = int(df_table.iloc[idx][1][0:1])
        year = index_to_year[year_idx]
        output = df_table.iloc[idx][1][7:]
        table += f"\n {year} & {output} & "
        #intercept
        table += f"{round(df_table.iloc[idx][3][0],2)}"
        table += f"{' ($<$ 0.001)'if round(df_table.iloc[idx][4][0],2) < 0.001 else '(' + str(round(df_table.iloc[idx][4][0],2)) + ')'}"
        #OD
        table += f"&{round(df_table.iloc[idx][3][1],2)}"
        table += f"{' ($<$ 0.001)'if round(df_table.iloc[idx][4][1],2) < 0.001 else '(' + str(round(df_table.iloc[idx][4][1],2)) + ')'}"
        table += f"&{round(df_table.iloc[idx][3][2],2)}"
        table += f"{' ($<$ 0.001)'if round(df_table.iloc[idx][4][2],2) < 0.001 else '(' + str(round(df_table.iloc[idx][4][2],2)) + ')'}"
        table += f"&{round(df_table.iloc[idx][3][3],2)}"
        table += f"{' ($<$ 0.001)'if round(df_table.iloc[idx][4][3],2) < 0.001 else '(' + str(round(df_table.iloc[idx][4][3],2)) + ')'}"
        table += f" \\\\ \hline"
    table +=f"\n\multicolumn{{6}}{{l}}{{{{}}}}\n"
    table += f"\end{{tabular}} \n }}  "
    table += f"\n \\end{{table}} \n }}"

    return table

