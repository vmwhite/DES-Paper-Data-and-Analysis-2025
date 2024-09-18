# DES-Paper-Data-and-Analysis-2023

This repository is the supportive code and data for the paper:

[1] V. M. White and L. A. Albert. (2023). “Evaluating diversion and treatment policies for opioid use disorder,” Pre-print. Available at: https://arxiv.org/abs/2311.05076 

## The "Data_Sources" folder
This folder obtains the collected raw data used to estimate the input parameters for the Dane County case study. Note that most of the data was converted from the original .xlsx files to .csv for faster loading into Python. 
- The file **"LNdist_DoubleCheck.xlsx"** contains any calculations made from the original data sources and an overall summary
- The file **"NSDUH - Drug Initiation Data/InitationTables.xlsx"** shows the calculations to estimate Dane County Opioid Use Prevalence and Initiation.

## The "Simulation_Functions/DataCleanngDCSim_NEW.py" file
This is Python code used to clean the raw data in the "DataSources" folder.

## The "AllScenarioMatrix.csv" file
This file contains the list of scenarios tested. Where:
- The first column represents the percentage of the Arrest Diversion policy implementation 
- The second column represents the percentage of the Overdose Diversion policy implementation
- The third column represents the percentage of the Re-entry Case Management implementation

## The "Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py" file
This is the Python code for the simulation. When you run the file, you enter the following four arguments:
- The 'Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py' file
- The number of runs you wish to conduct
- The length in years you want the simulation to run (this does not include the hard-coded 5-year warm-up period)
- the line number in the "AllScenarioMatrix.csv" that corresponds to the exact scenario you wish to run

## The "Post_Processing" folder
This folder contains various Python files used to process the outputs of "Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py" to the formats presented in [1] Specifically: 
- The **"Compile_RawResults.py"** file is Python Code that converts the raw output of the "Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py" to .csv files.
- The **"Prediction_interval_Calibration_Figs.py"** file is Python code used to create the .png files used in Figure 2 for [1]
- The "Results_Table.py" file is the Python code to create Table 4 for [1] and outputs the latex table as "CombinedResults.txt"
- The **"Scenario_Comparison.py"** file to create graphs in .png and tables in .txt format used in Table 5, Figure 3, and Appendix C for [1]
- The **"Graphing_Functions.py"** file is supportive Python code called by "Scenario_Comparison.py" and contains matplotlib formatted graphs.
- The **"Table_Functions.py"** file is supportive Python code called by "Scenario_Comparison.py" and contains the code to reformat data frames to latex.

## The "Appendix_Items" folder
This folder contains various Python files used to conduct supplemental analysis reported in the Appendices in [1]. Specifically: 
### In "Policy_Comparison" folder:
- The "Correlation_btwn_Policies.py" file was used to create the policy correlation table and analysis in Section 6 of [1].
### In "Replication_Analysis" folder:
- The "EstimatingNumberOfScenatios.py" file was used to support Appendix B for [1].
### In "Sensitivity_Analysis" folder:
- The "GenerateSensitivityMatrix.py" file was used to sample Sobol Sequences for Appendix C in [1].
- The "Simulation_OUD_Treatment_Policy_Sensitivity.py" file calls "GenerateSensitivityMatrix.py" and conducts the sensitivity analysis used in Appendix C for [1].
- The "Combine_Sens_Outputs.py" file compiles the raw sensitivity analysis outputs from "Sensitivity_Analysis/Simulation_OUD_Treatment_Policy_Sensitivity.py" to .csv files.
- The "ReadTotalYearlyEvents.py" file compiles the raw base model outputs from "Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py" to .csv files.
- The "RegressionAnalysis.py" file conducts the normality, scatterplots, PRCC, and OLS analyses for Appendix C in [1].
- The "Sens_Tables.py" file is a supportive Python code called "Sensitivity_Analysis/RegressionAnalysis.py" and contains the code to reformat data frames to latex for tables A2 and A3.

