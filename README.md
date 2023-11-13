# DES-Paper-Data-and-Analysis-2023

This repository is the supportive code and data for the paper: https://arxiv.org/abs/2311.05076 

[1] V. M. White and L. A. Albert. (2023). “Evaluating diversion and treatment policies for opioid use disorder,” Pre-print. 

## The "DataSources" folder
This folder obtains the collected raw data used to estimate the input parameters for the Dane County case study. 

## The "AllScenarioMatrix.csv"
This file contains the list of scenarios tested. Where 
- The first column represents the percentage of the Arrest Diversion policy implementation 
- The second column represents the percentage of the Overdose Diversion policy implementation
- The third column represents the percentage of the Re-entry Case Management implementation

## "Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py"
is Python code for the simulation. When you run the file, you enter the following four arguments:
- The 'Simulation_DC_OpioidPolicing_MARI_ED_and_CM.py' file
- The number of runs you wish to conduct
- The length in years you want the simulation to run (this does not include the hard-coded 5-year warm-up period)
- the line number in the "AllScenarioMatrix.csv" that corresponds to the exact scenario you wish to run

