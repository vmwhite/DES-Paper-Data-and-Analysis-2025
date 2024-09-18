########################################
# Data Cleaning and input for Dane County Opioid user Model
# calls classes
# last update 11/23/20
########################################

from concurrent.futures import thread
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import os
import sys

print ("The current working directory is", os.getcwd())
print("Begin writing DataSources/data_summary.txt")
os.makedirs('DataSources', exist_ok=True)
write_file = open("DataSources/data_summary.txt", "w")
original_stdout = sys.stdout
sys.stdout = write_file

# def line_graph(DFdata,DFyears,name): 
#     fig, ax = plt.subplots()
#     plt.plot(DFyears,DFdata, marker=".")
#     plt.xlabel('Year')
#     plt.ylabel('Number of '+ name)
#     # plt.xlim([0,num_years+1])
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     # plt.legend(['Expected', 'Actual'])
#     plt.savefig('DataSources/'+name+'_linegraph.png')
#     plt.close()

######################## DC Opioid Death Data ###############################
'''
The following applied for both DC and US Death Data.
Queried from: https://wonder.cdc.gov/mcd.html
CDC Wonder, Drug overdose deaths involving any opioid. Defined as ICD-10 code indicating:
    ICD-10 Code selected: UCD - Drug/Alcohol Induced Causes
        NCHS has defined selected causes of death groups for analysis of all ages mortality data: 
            Drug-Induced causes, Alcohol-Induced Causes, All Other Causes. 
            The group code values are not actual ICD codes published in the International Classification of Diseases, 
            but are "recodes" defined to support analysis by the Selected Causes of Death groups.
        These Codes include but are not limited too
            drug poisoning as an underlying cause of death: (X40-X44, X60-X64, X85, or Y10-Y14) 
    AND a contributing cause of death: T400 (opium), T401 (heroin), T402 (natural and semisynthetic opioid), 
        T403 (methadone), T404 (synthetic opioid other than methadone), or T406 (other and unspecified narcotic)
    Ages 12-100+ (to match initiation NSDUH data)
'''
print( "-------------------- Dane County Opioid Death Data Summary ---------------------------------")
## read in data
# df_DCdeaths = pd.read_csv(r'DataSources/CDC Wonder Queries - Death Data/Multiple Cause of Death, 1999-2020_DC_OpioidDeaths_byYear.txt', sep = "\t")
lst = [1999,2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
lst2 = [12, 14, 14, 15, 19, 24, 28, 28, 34, 34, 35, 46, 60, 55, 70, 62, 60, 90, 88, 85, 106, 123]
df_DCdeaths =   pd.DataFrame(list(zip(lst, lst2)),
               columns =[ 'Year','Deaths'])
print("*******Dane County Deaths Table. Source: CDC Wonder***********")
print(df_DCdeaths.to_string()) #final df for dane county death rates
##plot of data over years
#line_graph(df_DCdeaths['Deaths'], df_DCdeaths["Year"], "Opioid_Related_Deaths")

######################## DC non-opioid Death Data ###############################

print( "-------------------- Dane County All Death Data Summary ---------------------------------")
## read in data
# df_DCdeathsALL = pd.read_csv(r'DataSources/CDC Wonder Queries - Death Data/Multiple Cause of Death, 1999-2020_allDC_deahts.txt', sep = "\t")
# df_DCdeathsALL = df_DCdeathsALL.dropna(thresh=3)
# df_DCdeathsALL = df_DCdeathsALL.rename({'Deaths': 'DeathsALL'}, axis='columns')
print("*******Dane County All Deaths Table. Source: CDC Wonder***********")
# print(df_DCdeathsALL.to_string())#final df for dane county death rates
# ## transform data
# df_DCdeathsNonOpioid = pd.merge(df_DCdeathsALL, df_DCdeaths, on='Year')
# df_DCdeathsNonOpioid["nonOpioidDeaths"] = df_DCdeathsNonOpioid["DeathsALL"] - df_DCdeathsNonOpioid["Deaths"]
lst = [1999,2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
lst2 = [2512, 2462, 2607, 2617, 2617, 2582, 2592, 2560, 2719, 2681, 2675, 2784, 2852, 2944, 3048, 3211, 3083, 3095, 3252, 3325, 3295, 3828]
df_DCdeathsNonOpioid =   pd.DataFrame(list(zip(lst, lst2)),
               columns =[ 'Year','nonOpioidDeaths'])
print("*******Dane County Non-Opioid Deaths Table. Source: CDC Wonder***********")
print(df_DCdeathsNonOpioid.to_string()) #final df for dane county death rates
#CHECK: This is conistent with https://www.dhs.wisconsin.gov/publications/p01690.pdf 2012-2015 death rate per 100,000 is 12.4 (188) :) there's is on the low end because they use all ages, we use 12+. Therefore, the death rate is lower because of less younger people using. 
#plot of data over years
#line_graph(df_DCdeathsNonOpioid['nonOpioidDeaths'], df_DCdeaths["Year"], "NonOpioid_Related_Deaths")

######################## US Opioid Death Data ###############################
'''
For Details on source see above DC Death Data.
Needed for calculating Dane County "share" of opioid crisis for initiation rate, CURRRENTLY done in seperate excel document, InitiationTables.xlsx

# US Death Rate and Population Data
df_USdeaths = pd.read_excel(r'DataSources/CDC Wonder Queries - Death Data/Multiple Causes of Death, 1999-2019.xlsx', sheet_name="US", engine='openpyxl')
## transform data
#all US deaths
df_USdeaths = df_USdeaths[(df_USdeaths.Deaths !='Suppressed')] # remove supppressed rows
df_USdeaths["Deaths"] = pd.to_numeric(df_USdeaths['Deaths'], errors='coerce')

df_USdeathsALL = df_USdeaths #preserve for later use
df_USdeaths['DeathsALL'] = df_USdeaths['Deaths'].groupby(df_USdeaths['Year']).transform('sum') # sum each year for All Opioid Death types (T400, T401, T402, T403, T404, T406)
df_USdeaths = df_USdeaths[['Year', 'Population', 'DeathsALL']].drop_duplicates(subset=['Year']) #Remove duplicate years
#add in US heroin deaths
df_USdeathsH =df_USdeathsALL[(df_USdeathsALL['Multiple Cause of death Code'] =='T40.1')] #Filter for Heroin only
df_USdeathsH.rename(columns={'Deaths':'HDeaths'}, inplace=True) #rename deaths column
df_USdeathsH = df_USdeathsH[['Year', 'Population', 'HDeaths']]#Keep only interested columns
df_USdeaths = pd.merge(df_USdeaths,df_USdeathsH, on=['Year', 'Population'])
#add in US Perscription Opioid deaths
df_USdeathsP =df_USdeathsALL[df_USdeathsALL['Multiple Cause of death Code'].isin(['T40.2', 'T40.3', 'T40.4'])] #Filter for Perscription Misuse only, following methodolody of Wisc DHS WISH death calculations
df_USdeathsP = df_USdeathsP[(df_USdeathsP.Deaths !='Suppressed')] # remove supppressed rows
df_USdeathsP['PDeaths'] = df_USdeathsP['Deaths'].groupby(df_USdeathsP['Year']).transform('sum') # sum each year for All Prescription Opioid Death types (T402, T403, T404)
df_USdeathsP = df_USdeathsP[['Year', 'Population', 'PDeaths']].drop_duplicates(subset=['Year']) #Remove duplicate years
df_USdeathsP = df_USdeathsP[['Year', 'Population', 'PDeaths']]#Keep only interested columns
df_USdeaths = pd.merge(df_USdeaths,df_USdeathsP, on=['Year', 'Population'])
#print("US Deaths Table. Source CDC  Wonder", df_USdeaths)
'''
######################## Drug Initiation Data ########################
'''
Source: NSDUH
#Working code to calculate initaiton rate in python, currently calculations are done in the InitiationTables.xlsx file.
df_initiation = pd.read_excel(r'DataSources/NSDUH - Drug Initiation Data/InitiationTables.xlsx', sheet_name="NSDUH_Initiation")
df_initiation = pd.merge(df_initiation,df_USdeaths, on=['Year'])
print(df_initiation)
'''
print( "-------------------- Dane County approx Initiation Data Summary ---------------------------------")
#Using Option D, death based estimation
#df_initiation = pd.read_excel(r'DataSources/NSDUH - Drug Initiation Data/InitiationTables.xlsx', sheet_name="IncidenceDeaths", engine='openpyxl')
df_initiation = pd.read_csv(r'DataSources/NSDUH - Drug Initiation Data/InitiationTables-IncidenceDeaths.csv')
df_initiation = df_initiation[df_initiation['Year'] >= 2015] # full estimates are only for 2015 and later
df_initiation["Dane County Opioid Initiation estimate (number of people)"] = df_initiation["Dane County Heroin initation estimate (number of people)"] + df_initiation["Dane County  Pain Reliever initiation estimate (number of people)"]
df_initiation["Days per Year"] = [365,366,365,365,365]
df_initiation["Dane County Opioid Initiation estimate (number of people) per day"] = df_initiation["Dane County Opioid Initiation estimate (number of people)"] / df_initiation["Days per Year"]
init_rate = df_initiation["Dane County Opioid Initiation estimate (number of people) per day"].mean()
df_initiation['Year'] = df_initiation['Year'].astype(str)
print("*******Source: NSDUH***********")
print(df_initiation.to_string())
print("rate of first time drug use, i.e., Average days until next initiation post 2015: ", init_rate)
#plot of data over years
#line_graph(df_initiation["Dane County Opioid Initiation estimate (number of people)"], df_initiation["Year"], "Opioid_Initiation (number of people)")

######################## Drug Prevalence Data ########################
'''
Source: NSDUH
'''
print( "-------------------- Dane County approx Use Data Summary ---------------------------------")
#Using Option D, death based estimation
#df_prev = pd.read_excel(r'DataSources/NSDUH - Drug Initiation Data/InitiationTables.xlsx', sheet_name="PrevalenceDeaths", engine='openpyxl')
df_prev = pd.read_csv(r'DataSources/NSDUH - Drug Initiation Data/InitiationTables-Prevalence.csv')
df_prev = df_prev[df_prev['Year'] >= 2016] # full estimates are only for 2016 and later
df_prev["Year"] = df_prev["Year"].astype(str)
# print(search.index("2016"))
# df_prev["Dane County Opioid use estimate (number of people)"] = df_prev["Dane County Heroin use estimate (number of people)"].astype(float)+ df_prev["Dane County Pain Reliever use estimate (number of people)"].astype(float)
print("*******Source: NSDUH***********")
print(df_prev.to_string())
#plot of data over years
#line_graph(df_prev["Dane County use estimate (number of people)"], df_prev["Year"], "Opioid_Use (number of people)")

######################## DC Drug-Related Arrest Data ########################
'''
think about instead using FBI API: https://crime-data-explorer.fr.cloud.gov/pages/docApi
-can list by agency and by offensee type. from from further back, but drugs more generally

SRS Wisconsin Drug Offense Data:
https://www.doj.state.wi.us/dles/bjia/ucr-arrest-data
Adults and Juvenilles in Dane County for Drug Arrests 2015 -2019 data summaries
Drug types we are interested in:
    a. Opium or cocaine and their derivatives (morphine, heroin, codeine)
    c. Synthetic narcotics—manufactured narcotics which can cause true drug addiction (demerol, methadones)
    e. Unknown
Filtered Arrest Descriptions:
    'Drug - Unknown' 
    'Drug Possession - Opium/Cocaine' 
    'Drug Possession - Synthetic' 
    'Drug Sale - Opium/Cocaine' 
    'Drug Sale - Synthetic'

- Assuming age 12+ by using dane county population estimates for ages 12+
'''
print( "-------------------- Dane County Opioid Arrest Summary ---------------------------------")
# 2015 Drug Arrests Types - Adding and Filtering 
#df_arrests = pd.read_excel(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2015 (4_29_20).xlsx', engine='openpyxl')
df_arrests = pd.read_csv(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2015 4_29_20.csv')
df_arrests= df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.AdultOrJuvenile == "Total") ]
mask = df_arrests['ArrestDescription'].isin(["Drug Sale - Synthetic","Drug Sale - Opium/Cocaine","Drug - Unknown", "Drug Possession - Opium/Cocaine","Drug Possession - Synthetic"])
df_nonOpioidarrests = df_arrests[~mask]
df_arrests = df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.Category == "Drug Crimes") & (df_arrests.AdultOrJuvenile == "Total")]
df_arrestsOpioid = df_arrests[(df_arrests.ArrestDescription == "Drug - Unknown") | (df_arrests.ArrestDescription == "Drug Possession - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Possession - Synthetic") | (df_arrests.ArrestDescription == "Drug Sale - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Sale - Synthetic" ) ]

#2016 Drug Arrests Types - Adding and Filtering 
#df_arrests = pd.read_excel(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2016 (4_29_20).xlsx', engine='openpyxl')
df_arrests = pd.read_csv(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2016 4_29_20.csv')
df_arrests= df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.AdultOrJuvenile == "Total") ]
mask = df_arrests['ArrestDescription'].isin(["Drug Sale - Synthetic","Drug Sale - Opium/Cocaine","Drug - Unknown", "Drug Possession - Opium/Cocaine","Drug Possession - Synthetic"])
df_nonOpioidarrests2 = df_arrests[~mask]
df_nonOpioidarrests = pd.concat([df_nonOpioidarrests,df_nonOpioidarrests2])
df_arrests = df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.Category == "Drug Crimes") & (df_arrests.AdultOrJuvenile == "Total")]
df_arrests = df_arrests[(df_arrests.ArrestDescription == "Drug - Unknown") | (df_arrests.ArrestDescription == "Drug Possession - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Possession - Synthetic") | (df_arrests.ArrestDescription == "Drug Sale - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Sale - Synthetic" ) ]
df_arrestsOpioid = pd.concat([df_arrests,df_arrestsOpioid])

#2017 Drug Arrests Types - Adding and Filtering 
#df_arrests = pd.read_excel(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2017 (4_29_20).xlsx', engine='openpyxl')
df_arrests = pd.read_csv(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2017 4_29_20.csv')
df_arrests= df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.AdultOrJuvenile == "Total") ]
mask = df_arrests['ArrestDescription'].isin(["Drug Sale - Synthetic","Drug Sale - Opium/Cocaine","Drug - Unknown", "Drug Possession - Opium/Cocaine","Drug Possession - Synthetic"])
df_nonOpioidarrests2 = df_arrests[~mask]
df_nonOpioidarrests = pd.concat([df_nonOpioidarrests,df_nonOpioidarrests2])
df_arrests = df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.Category == "Drug Crimes") & (df_arrests.AdultOrJuvenile == "Total")]
df_arrests = df_arrests[(df_arrests.ArrestDescription == "Drug - Unknown") | (df_arrests.ArrestDescription == "Drug Possession - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Possession - Synthetic") | (df_arrests.ArrestDescription == "Drug Sale - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Sale - Synthetic" ) ]
df_arrestsOpioid = pd.concat([df_arrests,df_arrestsOpioid])

#2018 Drug Arrests Types - Adding and Filtering 
#df_arrests = pd.read_excel(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2018 (4_29_20).xlsx', engine='openpyxl')
df_arrests = pd.read_csv(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2018 4_29_20.csv')
df_arrests= df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.AdultOrJuvenile == "Total") ]
mask = df_arrests['ArrestDescription'].isin(["Drug Sale - Synthetic","Drug Sale - Opium/Cocaine","Drug - Unknown", "Drug Possession - Opium/Cocaine","Drug Possession - Synthetic"])
df_nonOpioidarrests2 = df_arrests[~mask]
df_nonOpioidarrests = pd.concat([df_nonOpioidarrests,df_nonOpioidarrests2])
df_arrests = df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.Category == "Drug Crimes") & (df_arrests.AdultOrJuvenile == "Total")]
df_arrests = df_arrests[(df_arrests.ArrestDescription == "Drug - Unknown") | (df_arrests.ArrestDescription == "Drug Possession - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Possession - Synthetic") | (df_arrests.ArrestDescription == "Drug Sale - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Sale - Synthetic" ) ]
df_arrestsOpioid = pd.concat([df_arrests,df_arrestsOpioid])

#2019 Drug Arrests Types - Adding and Filtering 
#df_arrests = pd.read_excel(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2019 (4_29_20).xlsx', engine='openpyxl')
df_arrests = pd.read_csv(r'DataSources/Wisc DOJ - Arrest_Crime Data/WIBERS data/Agency-level Arrests 2019 4_29_20.csv')
df_arrests= df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.AdultOrJuvenile == "Total") ]
mask = df_arrests['ArrestDescription'].isin(["Drug Sale - Synthetic","Drug Sale - Opium/Cocaine","Drug - Unknown", "Drug Possession - Opium/Cocaine","Drug Possession - Synthetic"])
df_nonOpioidarrests2 = df_arrests[~mask]
df_nonOpioidarrests = pd.concat([df_nonOpioidarrests,df_nonOpioidarrests2])
df_arrests = df_arrests[(df_arrests.County == "Dane" ) & (df_arrests.Category == "Drug Crimes") & (df_arrests.AdultOrJuvenile == "Total")]
#print(np.unique(df_arrests['ArrestDescription'])) #shows all 'ArrestDescription' fields. Note: I checked that all years 2015-2019 had the same initial fields
df_arrests = df_arrests[(df_arrests.ArrestDescription == "Drug - Unknown") | (df_arrests.ArrestDescription == "Drug Possession - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Possession - Synthetic") | (df_arrests.ArrestDescription == "Drug Sale - Opium/Cocaine") | (df_arrests.ArrestDescription == "Drug Sale - Synthetic" ) ]
#print(np.unique(df_arrests['ArrestDescription'])) #shows reamining 'ArrestDescription' fields, should match fields stated in the 'DC Drug-Related Arrest Data' Description. 
df_arrestsOpioid = pd.concat([df_arrests,df_arrestsOpioid])

#Final Full arrest Dataframe year, PD, Type
#print(df_arrestsOpioid)

#Drug Arrests Summarized by year
df_Yarrests = df_arrestsOpioid.groupby('ArrestYear')['ArrestCount'].sum()
df_Yarrests = pd.DataFrame({"index": df_Yarrests.index, "ArrestCount":df_Yarrests.values})
df_Yarrests = df_Yarrests.rename({df_Yarrests.columns[0]:'Year'}, axis =1)
df_Yarrests["Year"] =df_Yarrests["Year"].astype(str)
print("*******Source: SRS Wisconsin Drug Offense Data ***********")
print(df_Yarrests.to_string())

# non opioid Arrests Summarized by year
df_YarrestsNON = df_nonOpioidarrests.groupby('ArrestYear')['ArrestCount'].sum()
df_YarrestsNON = pd.DataFrame({"index": df_YarrestsNON.index, "ArrestCount":df_YarrestsNON.values})
df_YarrestsNON = df_YarrestsNON.rename({df_YarrestsNON.columns[0]:'Year'}, axis =1)
df_YarrestsNON["Year"] =df_YarrestsNON["Year"].astype(str)
print("*******Source: SRS Wisconsin non-opioid Offense Data ***********")
print(df_YarrestsNON.to_string())
#plot of data over years
#line_graph(df_Yarrests["ArrestCount"], df_Yarrests["Year"], "Opioid_Arrests (Number of Arrests)")

######################## DC Treatment Data ########################
'''
Taken from Wisconsin DHS: "Opioids: Treatment Data by County Dashboard"
https://www.dhs.wisconsin.gov/opioids/treatment-data-county.htm
Filtered for Dane County Only. Does not include private insurance claims since these are in #of episodes and not specific to Dane County.
Does include County-Authorized Treatment and  Medicaid Treatment and is #of individuals who sought treatment, these two groups may not be mutually exclusive.
'''
print( "-------------------- Dane County Opioid Treatment Summary ---------------------------------")
#df_treat = pd.read_excel(r'DataSources/Wisc DHS - Treatment Data/TreatmentDaneCounty.xlsx', sheet_name ="OpioidSummary", engine='openpyxl')
df_treat = pd.read_csv(r'DataSources/Wisc DHS - Treatment Data/TreatmentDaneCounty.csv')
#remove WI data keep Dane County data
df_treat= df_treat.iloc[:, :-1]
#add together County-Authorized Treatment individuals and  Medicaid Treatment individuals
df_treat = df_treat.groupby('Year')['Total Individuals - Dane'].sum()
df_treat= pd.DataFrame({"Year": df_treat.index, "Total Individuals":df_treat.values})
df_treat["Year"]=df_treat["Year"].astype(str)
print( "********** Source: Wisconsin DHS, Opioids: Treatment Data by County Dashboard **********")
print(df_treat.to_string())
#line_graph(df_treat["Total Individuals - Dane"], df_treat["Year"], "Opioid_Treatment (Number of Treated Individuals)") #plot of data over years

######################## DC Hospital Encounter Data ########################
'''
Taken from Wisconsin DHS: "WISH Query: Opioid-Related Hospital Encounters"
https://www.dhs.wisconsin.gov/wish/opioid/hospital-encounters.htm
Filtered for Dane County Only, Population adjusted rates for ages 15+
'''
print( "-------------------- Dane County Opioid Hospital Encounter Summary ---------------------------------")
#df_HE = pd.read_excel(r'DataSources/Wisc DHS - Opioid-Related Hospital Encounters/Opioid-Related Hospital Encounters (Dane County, 2005-2019).xlsx', sheet_name ="Ages 14+ All HE",header=9)
df_HE = pd.read_csv(r'DataSources/Wisc DHS - Opioid-Related Hospital Encounters/Opioid-Related Hospital Encounters Dane County 2005-2019 All_he_14plus.csv')
df_HE = df_HE.rename({df_HE.columns[0]:'Year'}, axis =1) #Rename first column
df_HE = df_HE[(df_HE.Year != "All")] #get rid of "all" row
df_HE["Number of Discharges"]=df_HE["Number of Discharges"].astype(float)
print( "********** Source: Wisconsin DHS, Opioids: HEment Data by County Dashboard **********")
print(df_HE.to_string())
#line_graph(df_HE["Number of Discharges"], df_HE["Year"], "Opioid_HE (Number of Hospital Encounters)") #plot of data over years

############## Summary plots ################################
#plot 1 all 
# plt.plot(df_DCdeaths["Year"], df_DCdeaths['Deaths'],  marker=".", color="r")
# plt.plot(df_initiation["Year"], df_initiation["Dane County Opioid Initiation estimate (number of people)"], marker=".", color="b")
# plt.plot(df_prev["Year"],df_prev["Dane County use estimate (number of people)"],  marker=".", color="m")
# plt.plot( df_Yarrests["Year"],df_Yarrests["ArrestCount"], marker=".", color="c")
# plt.plot(df_treat["Year"], df_treat["Total Individuals"], marker=".", color="g")
# plt.plot(df_HE["Year"], df_HE["Number of Discharges"], marker=".", color="k")

# plt.legend(['Deaths', 'Initiation', "Use" ,"Arrests" ,"Treatment", "Hospital Encounters"])
# plt.xlabel('Year')
# plt.ylabel('Number')
# plt.xlim([2004,2020])
# plt.savefig('DataSources/Opioid_ALL_linegraph.png')
# plt.close()

# #plot 2 without use
# plt.plot(df_DCdeaths["Year"], df_DCdeaths['Deaths'],  marker=".", color="r")
# plt.plot(df_initiation["Year"], df_initiation["Dane County Opioid Initiation estimate (number of people)"], marker=".", color="b")
# plt.plot( df_Yarrests["Year"],df_Yarrests["ArrestCount"], marker=".", color="c")
# plt.plot(df_treat["Year"], df_treat["Total Individuals"], marker=".", color="g")
# plt.plot(df_HE["Year"], df_HE["Number of Discharges"], marker=".", color="k")

# plt.legend(['Deaths', 'Initiation', "Arrests" ,"Treatment", "Hospital Encounters"])
# plt.xlabel('Year')
# plt.ylabel('Number')
# plt.xlim([2004,2020])
# plt.savefig('DataSources/Opioid_ALL2_linegraph.png')
# plt.close()

############## End of File ########################

sys.stdout = original_stdout
print("Done writing DataSources/data_summary.txt")

# '''Printing Values from figures 1 and 2'''
# print("DEATHS")
# print(df_DCdeaths['DeathsALL'])
# print("aRRIVALS")
# print(df_initiation["Dane County Opioid Initiation estimate (number of people)"])
# print("PREVALENCE")
# print(df_prev["Dane County Opioid use estimate (number of people)"])
# print("ARRESTS")
# print(df_Yarrests["ArrestCount"])
# print("INDIVIDUALS IN TREATMENT")
# print(df_treat["Total Individuals"])
# print("HOSPITAL ENCOUNTERS")
# print(df_HE["Number of Discharges"])



