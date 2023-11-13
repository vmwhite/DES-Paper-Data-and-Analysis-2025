# DES-Paper-Data-and-Analysis-2023

This repository is the supportive code and data for the paper:

[1] V. M. White and L. A. Albert. (2023). “Evaluating diversion and treatment policies for opioid use disorder,” Pre-print. Available at: https://arxiv.org/abs/2311.05076 

## The "DataSources" folder
This folder obtains the collected raw data used to estimate the input parameters for the Dane County case study. Note that most of the data was converted from the original .xlsx files to .csv for faster loading into Python. 
- The file "LNdist_DoubleCheck.xlsx" contains any calculations made from the original data sources and an overall summary
- The file "NSDUH - Drug Initiation Data/InitationTables.xlsx" shows the calculations to estimate Dane County Opioid Use Prevalence and Initiation.

## The "AllScenarioMatrix.csv"
This file contains the list of scenarios tested. Where:
- The first column represents the percentage of the Arrest Diversion policy implementation 
- The second column represents the percentage of the Overdose Diversion policy implementation
- The third column represents the percentage of the Re-entry Case Management implementation

## "Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py"
This is the Python code for the simulation. When you run the file, you enter the following four arguments:
- The 'Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py' file
- The number of runs you wish to conduct
- The length in years you want the simulation to run (this does not include the hard-coded 5-year warm-up period)
- the line number in the "AllScenarioMatrix.csv" that corresponds to the exact scenario you wish to run

## "Compile_RawResults.py"
This Python Code converts the raw output of the "Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py" to .csv files.

## "DataCleanngDCSim_NEW.py"
This is Python code used to clean the raw data in the "DataSources" folder.
