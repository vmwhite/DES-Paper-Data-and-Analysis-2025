############################ OLS Latex Table ############################################
def OLS_Results_Table(df, param_name, df_prcc, year_list):
    backslash_char = "\\" 
    ############################ Create Latex Table ############################################
    table = f"{{\spacingset{{1.5}}"
    table +=f"\n\\begin{{table}}"
    table +=f"\n\caption{{ OLS effect size of input parameters vs main outputs in 2032 for (60,80,60) scenario }}"
    table +=f"\n\label{{tab:OLS_Results}}"
    table +=f"\n\\resizebox{{\\textwidth}}{{!}}{{"
    table +=f"\n\\begin{{tabular}}{{|c||c|c|c||c|c|c||c|c|c||c|c|c||c|c|c|}}"
    table +=f"\n\hline"
    table +=f"\n\multicolumn{{1}}{{|c|}}{{\mulitrow{{3}}{{4cm}}{{Parameter Number}}}} & \multicolumn{{3}}{{|c|}}{{Opioid-Related Deaths}} & \multicolumn{{3}}{{|c|}}{{Opioid-Related Arrests}} & \multicolumn{{3}}{{|c|}}{{Opioid-Related Hospital Encounters}} & \multicolumn{{3}}{{|c|}}{{OUD Treatment Starts}} & \multicolumn{{3}}{{|c|}}{{Active Use Starts}}\\\\ \hline"
    table +=f"\n & \multicolumn{{3}}{{c|}}{{ effect size}}  & \multicolumn{{3}}{{c|}}{{ effect size}}  & \multicolumn{{3}}{{c|}}{{ effect size}}  & \multicolumn{{3}}{{c|}}{{ effect size}}  & \multicolumn{{3}}{{c|}}{{effect size}}  \\\\ \hline \hline"
    table +=f"\n & Year 2016 & Year 2018 & Year 3032 & Year 2016 & Year 2018 & Year 3032  & Year 2016 & Year 2018 & Year 3032   &Year 2016 & Year 2018 & Year 3032   & Year 2016 & Year 2018 & Year 3032   \\\\ \hline \hline"

    i=1
    for idx, param in enumerate(param_name):
        # add parameter string
        table += f" \n\n \multirow{{{i}}}{{*}}{{{ df_prcc[idx]['Param_num']+1}}}"
        #check if p_val of death is significant if so bf
        for y_idx, year in enumerate(year_list):
            table += f" & {(backslash_char + 'bf{' + str(df[4][idx][y_idx]['effect_size']) ) if df_prcc[idx]['ODeaths_pval_'+year] <0.05 else str(df[4][idx][y_idx]['effect_size'])}"
            table += f"{'$^*$}   ' if df_prcc[idx]['ODeaths_pval_'+year] <0.05 else ''}"
        #check if p_val of arrest is significant if so bf
        for y_idx, year in enumerate(year_list):
            table += f" & {(backslash_char + 'bf{' + str(df[1][idx][y_idx]['effect_size']) ) if df_prcc[idx]['OArrest_pval_'+year] <0.05 else str(df[1][idx][y_idx]['effect_size'])}"
            table += f"{'$^*$ }  ' if df_prcc[idx]['OArrest_pval_'+year] <0.05 else ''}"                
        #check if p_val of hosp is significant if so bf
        for y_idx, year in enumerate(year_list):
            table += f" & {(backslash_char + 'bf{' + str(df[2][idx][y_idx]['effect_size']) ) if df_prcc[idx]['Hosp_pval_'+year] <0.05 else str(df[2][idx][y_idx]['effect_size'])}"
            table += f"{'$^*$ }  ' if df_prcc[idx]['Hosp_pval_'+year] <0.05 else ''}"
        #check if p_val of treat is significant if so bf
        for y_idx, year in enumerate(year_list):
            table += f" & {(backslash_char + 'bf{' + str(df[7][idx][y_idx]['effect_size']) ) if df_prcc[idx]['Treats_pval_'+year] <0.05 else str(df[7][idx][y_idx]['effect_size'])}"
            table += f"{'$^*$  } ' if df_prcc[idx]['Treats_pval_'+year] <0.05 else ''}"
        #check if p_val of active is significant if so bf
        for y_idx, year in enumerate(year_list):
            table += f" & {(backslash_char + 'bf{' + str(df[0][idx][y_idx]['effect_size']) ) if df_prcc[idx]['Active_pval_'+year] <0.05 else str(df[0][idx][y_idx]['effect_size'])}"
            table += f"{'$^*$  } ' if df_prcc[idx]['Active_pval_'+year] <0.05 else ''}"
        #next row of table 
        table += f"\\\\ \hline"

    table +=f"\n\multicolumn{{16}}{{l}}{{*{{Statistically significant PRCC t-test with Bonforroni Corrected p-val at 0.05 }}}}\\\\"
    table +=f"\n\end{{tabular}} "
    table +=f"\n}}"
    table +=f"\n\end{{table}}"
    table +=f"\n}}"

    return table

############################ PRCC Latex Table ############################################
def PRCC_Results_Table(df, param_name, year_list):
    backslash_char = "\\" 
    ############################ Create Latex Table ############################################
    table = f"{{\spacingset{{1.5}}"
    table +=f"\n\\begin{{sidewaystable}}"
    table +=f"\n\caption{{ Partial rank correlation coefficient (p-value) of input parameters vs main outputs in 2032 for (60,80,60) scenario }}"
    table +=f"\n\label{{tab:PRCC_Results}}"
    table +=f"\n\\resizebox{{\\textwidth}}{{!}}{{"
    table +=f"\n\\begin{{tabular}}{{|c||c|c|c||c|c|c||c|c|c||c|c|c||c|c|c|}}"
    table +=f"\n\hline"
    table +=f"\n\multicolumn{{1}}{{|c|}}{{\mulitrow{{3}}{{4cm}}{{Parameter Number}}}} & \multicolumn{{3}}{{|c|}}{{Opioid-Related Deaths}} & \multicolumn{{3}}{{|c|}}{{Opioid-Related Arrests}} & \multicolumn{{3}}{{|c|}}{{Opioid-Related Hospital Encounters}} & \multicolumn{{3}}{{|c|}}{{OUD Treatment Starts}} & \multicolumn{{3}}{{|c|}}{{Active Use Starts}}\\\\ \hline"
    table +=f"\n & \multicolumn{{3}}{{|c|}}{{ coefficient (p-value)}}  & \multicolumn{{3}}{{|c|}}{{ coefficient (p-value)}}  & \multicolumn{{3}}{{|c|}}{{ coefficient (p-value)}}  & \multicolumn{{3}}{{|c|}}{{ coefficient (p-value)}}  & \multicolumn{{3}}{{|c|}}{{coefficient  (p-value)}}  \\\\ \hline \hline"
    table +=f"\n & Year 2016 & Year 2018 & Year 3032 & Year 2016 & Year 2018 & Year 3032  & Year 2016 & Year 2018 & Year 3032   &Year 2016 & Year 2018 & Year 3032   & Year 2016 & Year 2018 & Year 3032   \\\\ \hline \hline"

    i=1
    for idx, param in enumerate(param_name):
        for l_idx, list in enumerate(df):
            if df[l_idx]["Param_code"] == param:
                # add parameter string
                table += f" \n\n \multirow{{{i}}}{{*}}{{{df[l_idx]['Param_num']+1}}}"
                #check if p_val of death is significant if so bf
                for year in year_list:
                    table += f" & {(backslash_char + 'bf{' + str(df[l_idx]['ODeaths_coe_'+year]) ) if df[l_idx]['ODeaths_pval_'+year] <0.05 else str(df[l_idx]['ODeaths_coe_'+year])}"
                    table += f"{'$^*$  ' if df[l_idx]['ODeaths_pval_'+year] <0.05 else ''}"
                    table += f"{'(' + str(df[l_idx]['ODeaths_pval_'+year]) + ') }' if df[l_idx]['ODeaths_pval_'+year] <0.05 else '(' + str(df[l_idx]['ODeaths_pval_'+year]) + ')'}"
                #check if p_val of arrest is significant if so bf
                for year in year_list:
                    table += f" & {(backslash_char + 'bf{' + str(df[l_idx]['OArrest_coe_'+year]) ) if df[l_idx]['OArrest_pval_'+year] <0.05 else str(df[l_idx]['OArrest_coe_'+year])}"
                    table += f"{'$^*$  ' if df[l_idx]['OArrest_pval_'+year] <0.05 else ''}"
                    table += f"{'(' + str(df[l_idx]['OArrest_pval_'+year]) + ') }' if df[l_idx]['OArrest_pval_'+year] <0.05 else '(' + str(df[l_idx]['OArrest_pval_'+year]) + ')'}"
                #check if p_val of hosp is significant if so bf
                for year in year_list:
                    table += f" & {(backslash_char + 'bf{' + str(df[l_idx]['Hosp_coe_'+year]) ) if df[l_idx]['Hosp_pval_'+year] <0.05 else str(df[l_idx]['Hosp_coe_'+year])}"
                    table += f"{'$^*$  ' if df[l_idx]['Hosp_pval_'+year] <0.05 else ''}"
                    table += f"{'(' + str(df[l_idx]['Hosp_pval_'+year]) + ') }' if df[l_idx]['Hosp_pval_'+year] <0.05 else '(' + str(df[l_idx]['Hosp_pval_'+year]) + ')'}"
                #check if p_val of treats is significant if so bf
                for year in year_list:
                    table += f" & {(backslash_char + 'bf{' + str(df[l_idx]['Treats_coe_'+year]) ) if df[l_idx]['Treats_pval_'+year] <0.05 else str(df[l_idx]['Treats_coe_'+year])}"
                    table += f"{'$^*$  ' if df[l_idx]['Treats_pval_'+year] <0.05 else ''}"
                    table += f"{'(' + str(df[l_idx]['Treats_pval_'+year]) + ') }' if df[l_idx]['Treats_pval_'+year] <0.05 else '(' + str(df[l_idx]['Treats_pval_'+year]) + ')'}"
                #check if p_val of active is significant if so bf
                for year in year_list:
                    table += f" & {(backslash_char + 'bf{' + str(df[l_idx]['Active_coe_'+year]) ) if df[l_idx]['Active_pval_'+year] <0.05 else str(df[l_idx]['Active_coe_'+year])}"
                    table += f"{'$^*$  ' if df[l_idx]['Active_pval_'+year] <0.05 else ''}"
                    table += f"{'(' + str(df[l_idx]['Active_pval_'+year]) + ') }' if df[l_idx]['Active_pval_'+year] <0.05 else '(' + str(df[l_idx]['Active_pval_'+year]) + ')'}"
                #next row of table 
                table += f"\\\\ \hline"

    table +=f"\n\multicolumn{{16}}{{l}}{{*{{Statistically significant PRCC t-test with Bonforroni Corrected p-val at 0.05 }}}}\\\\"
    table +=f"\n\end{{tabular}} "
    table +=f"\n}}"
    table +=f"\n\end{{sidewaystable}}"
    table +=f"\n}}"

    return table

