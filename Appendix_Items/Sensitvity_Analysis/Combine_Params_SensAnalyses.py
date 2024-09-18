import re
import os
import math
import numpy as np
import ast
import csv
import re
cwd = os.getcwd() 
folder = r'Sensitivity_Runs\OutputFiles_ParamsUsed'
files = os.listdir(folder)
params = []
params.append(['Run',"dup_init_age_mean","dup_init_age_sig","dup_prev_age_mean","dup_prev_age_sig",
               "Starting_Pop_Size","Starting_Hosp_UT_Size","Starting_CJS_UT_Size","Starting_Treat_UT_Size","starting_inactive_UT_Size",
                "lam_user_arrival","LNmean_deathdays_pre2019", "LNsigma_deathdays_pre2019", "LNmean_deathdays_post2019", "LNsigma_deathdays_post2019",
                 "LNmean_hospdays", "LNsigma_hospdays", "LNmean_Oarrestdays", "LNsigma_Oarrestdays", 
                 "LNmean_treatdays", "LNsigma_treatdays", "LNmean_iadays", "LNsig_iadays", "LNmean_nonOarrestdays", "LNsigma_nonOarrestdays", 
                 "P(CJS | HE)", "P(Treatment | HE)", "1 - P(Death|HE)",
                 "LNmean_hospservice", "LNsig_hospservice", "LNmean_crimeservice", "LNsig_crimeservice", "LNmean_treatservice", "LNsig_treatservice", 
                 "LNmean_crimerel", "LNsig_crimerel", "LNmean_treatrel", "LNsig_treatrel", "LNmean_hosprel", "LNsig_hosprel", "LNmean_iarel", "LNsig_iarel"])
for idx, d in enumerate(files):
    if d.endswith('.out'):
        count = 0
        fin = open(folder+"/"+d, 'r')
        lines = fin.readlines()
        found_ts = False
        for idx2,line in enumerate(lines):
            if found_ts:
                #clean file to get string in format "[,,,]" to parse
                temp_list=""
                for n in range(idx2,idx2+11):
                    temp_list =temp_list+ lines[n]
                temp_list = temp_list.replace('\n',"")
                temp_list = temp_list.replace(' ',", ")
                #parse clean string to python list
                parsed_list = ast.literal_eval(temp_list)
                #addes specific run number to list
                run = re.findall(r'\d+', str(files[idx]))
                parsed_list.insert(0,int(run[1]))
                #appends to total list
                params.append(parsed_list)
                break
                                
            if "L2-star Discrepency" in line:
                count += 1
                found_ts = True
        fin.close()

print("-------- Params ---------")
print(params)

with open("Sensitivity_Runs\Sens_Params.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(params)

f.close()