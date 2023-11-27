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

def main_calcs(df_Yearly2, year):
    backslash_char = "\\"
    df_Yearly2['p_value_diff']  = None
    df_Yearly2['sig_diff?']  = None
    for idx in range(len(df_Yearly2)-1):
        df_Yearly2['p_value_diff'][idx] =  df_Yearly2['mean_p-value'][idx+1]
        df_Yearly2['sig_diff?'][idx] =  df_Yearly2['SignificantMean?'][idx+1]
    df_Yearly2 = df_Yearly2.T[[col for col in df_Yearly2.T if year in col]]
    df_Yearly2 = df_Yearly2.T
    df_Yearly2 = df_Yearly2.iloc[::2]
    df_Yearly2['Total'] = df_Yearly2.apply(lambda row: f"{(backslash_char +'bf{' + str(round(row[0], 2)) + ' $' + backslash_char+'pm$ ' + str(round(row[2], 2)) +'$^*$ (' + str(row[6]) +')}') if row[7] == 'Yes' else str(round(row[0], 2)) + ' $' + backslash_char+'pm$ ' + str(round(row[2], 2)) +' (' + str(row[6]) +')'}", axis = 1)
    return df_Yearly2
def add_top_column(df, idx,  Output_title):
        df = df.T[[col for col in df.T if Output_title[idx] in col]]
        return df
######################################## latex tables ################################################################################################################
def Combined_Results_Table(df_2032_costs2, Output_Scenario_TableOrder,policy_dict, df_Yearly2, Output_title,ScenOrder_table_to_files):
    df_sum = df_2032_costs2
    df_sum['mean_total_cost']  = None
    df_sum['std_total_cost']  = None
    for idx in range(len(df_sum)):
        df_sum['mean_total_cost'][idx] =  df_sum['mean'][idx-1]
        df_sum['std_total_cost'][idx] =  df_sum['cost_se'][idx-1]
    df_sum = df_sum.T[[col for col in df_sum.T if "2032" in col]]
    df_sum = df_sum[[col for col in df_sum if "Diff" in col]]
    df_sum = df_sum.T
    

    df = main_calcs(df_Yearly2, "2032")

    df_0 = add_top_column(df, 0, Output_title)
    df_1 = add_top_column(df,  1, Output_title)
    df_2 = add_top_column(df, 2, Output_title)
    df_3 = add_top_column(df,  3, Output_title)
    df_4 = add_top_column(df,  4, Output_title)
    backslash_char = "\\" 
    ############################ Create Latex Table ############################################
    table = f"{{\spacingset{{1.5}}"
    table +=f"\n\\begin{{sidewaystable}}"
    table +=f"\n\caption{{Simulated mean $\pm$ standard error (p-value) of total events, cost, and savings in the Year 2032 for scenarios}}"
    table +=f"\n\label{{tab:AD_All_Results}}"
    table +=f"\n\\resizebox{{\\textwidth}}{{!}}{{"
    table +=f"\n\\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|}}"
    table +=f"\n\hline"
    table +=f"\n\multicolumn{{2}}{{|c|}}{{Scenario}} & \multicolumn{{1}}{{c|}}{{Opioid-Related Deaths}} & \multicolumn{{1}}{{c|}}{{Opioid-Related Arrests}} & \multicolumn{{1}}{{c|}}{{Opioid-Related Hospital Encounters}} & \multicolumn{{1}}{{c|}}{{OUD Treatment Starts}} & \multicolumn{{1}}{{c|}}{{Active Use Starts}} & \multirow{{1}}{{*}}{{Mean Total Cost}} & \multirow{{1}}{{*}}{{Mean Cost  Difference}}\\\\ \hline"
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
        
        table += f" & \multirow{{{i}}}{{*}}{{{policy_dict[scen]}}} & \multirow{{{i}}}{{*}}{{{df_0.iloc[:,idx_f][8]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_1.iloc[:,idx_f][8]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_2.iloc[:,idx_f][8]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_3.iloc[:,idx_f][8]}}}"
        table += f" & \multirow{{{i}}}{{*}}{{{df_4.iloc[:,idx_f][8]}}}"
        #for calling from df_diff_cost_new
        table += f" & \multirow{{{i}}}{{*}}{{{(backslash_char + 'bf{' + str(round(float(df_sum.iloc[idx_f][8]/1000000), 2)) + ' $' + backslash_char+'pm$ ' +  str(round(float(df_sum.iloc[idx_f][9]/1000000), 2)) +'}') if  df_sum.iloc[idx_f][7]== 'Yes' else  str(round(float(df_sum.iloc[idx_f][8]/1000000), 2))  + ' $' + backslash_char+'pm$ ' +  str(round(float(df_sum.iloc[idx_f][9]/1000000), 2)) }"
        table += f"{'$^*$  ' if df_sum.iloc[idx_f][7] == 'Yes' else ''}"
        table += f" ({df_sum.iloc[idx_f][6]})}}"
        table += f" & \multirow{{{i}}}{{*}}{{{(backslash_char + 'bf{' + str(round(float(df_sum.iloc[idx_f][0]/1000000), 2)) +'}') if   df_sum.iloc[idx_f][7] == 'Yes' else str(round(float(df_sum.iloc[idx_f][0]/1000000), 2))  }}}"
        if scen =="Base":    
            table +=f"\n\\\\ &&&&&&&&\\\\"
            i -= 1
        else: 
            table +=f"\\\\"
        
        
    table += f" \hline"
    table +=f"\n\multicolumn{{9}}{{l}}{{*{{Statistically significant difference from the Base Model at a level of 0.05 using 2 -sided paired t-test}}}}\\\\"
    table +=f"\n\multicolumn{{9}}{{l}}{{*{{Note: Overlapping confidence intervals that are marked statistically significant through a paired t-test can be attributed to Type one error \citep{{knol_misuse_2011}} }}}}\\\\"
    table +=f"\n\end{{tabular}} "
    table +=f"\n}}"
    table +=f"\n\end{{sidewaystable}}"
    table +=f"\n}}"
    print(table)
    
    return table

def Avg_PerPersonRatesTable(df_table, policy_dict, year_list, Output_Scenario_TableOrder):
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
    table +=f"\n\multicolumn{{2}}{{|c|}}{{Scenario}} & \multicolumn{{{len(year_list)}}}{{c|}}{{Re-Arrest rate}} & \multicolumn{{{len(year_list)}}}{{c|}}{{Re-Hospitalisation rate}} & \multicolumn{{{len(year_list)}}}{{c|}}{{OUD Treatment Re-Start rate}} \\\\ \hline"
    table +=f"\n\multicolumn{{2}}{{|c|}}{{\multirow{{2}}{{*}}{{AD (\%), OD (\%), CM (\%)}}}} & \multicolumn{{{len(year_list)}}}{{c|}}{{mean (\%) $\pm$ se (p-value)}} & \multicolumn{{{len(year_list)}}}{{c|}}{{mean (\%) $\pm$ se (p-value)}} & \multicolumn{{{len(year_list)}}}{{c|}}{{mean (\%) $\pm$ se (p-value)}}  \\\\   \cline{{3-11}}"
    table +=f"\n & "
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
                    table += f" {round(df_table.iloc[idx][3],2)} $\pm$ {round(df_table.iloc[idx][4],3)} "
                if df_table.iloc[idx][8] == 'yes' and df_table.iloc[idx][2] == year and df_table.iloc[idx][0] == scen:
                    table += f"*}}"  
        table += f"\\\\"
    table += f"\n \hline "
    table +=f"\n\multicolumn{{11}}{{l}}{{*{{Statistically significant difference, with p-value $<0.001$, from the Base Model at a level of 0.05 using 2-sided paired t-test}}}}\n"
    table += f"\end{{tabular}} \n }}  "
    table += f"\n \\end{{sidewaystable}} \n }}"

    return table


