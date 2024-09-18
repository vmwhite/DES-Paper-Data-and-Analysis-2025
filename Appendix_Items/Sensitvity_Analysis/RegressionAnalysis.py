#combine both csv files
#run linear regreassion to get standardize coeffiences
# calculate \u03C3-normalized derivartives.
#if they are similar the model is linear. 

#put results into table 
#plot more influential variables to outputs
'''
########## note to self. might have to open params.csv file to make sure it is spacing the file correctly and not offsetting columns and rows. if df is empty this is what happend. 
- need to manually chage first line of column names before merging in the Avg_Service_Runs.csv


Understanding OLS.Summary
Omnibus describes the normalcy of the distribution of our residuals using skew and kurtosis as measurements. A 0 would indicate perfect normalcy. 
Prob(Omnibus) is a statistical test measuring the probability the residuals are normally distributed. A 1 would indicate perfectly normal distribution. 
Jarque-Bera (JB) and Prob(JB) are alternate methods of measuring the same value as Omnibus and Prob(Omnibus) using skewness and kurtosis. We use these values to confirm each other. 
Skew is a measurement of symmetry in our data, with 0 being perfect symmetry. 
Kurtosis measures the peakiness of our data, or its concentration around 0 in a normal curve. Higher kurtosis implies fewer outliers.
Durbin-Watson is a measurement of homoscedasticity, or an even distribution of errors throughout our data. 
    Heteroscedasticity would imply an uneven distribution, for example as the data point grows higher the relative error grows higher. 
    Ideal homoscedasticity will lie between 1 and 2. 
Condition number is a measurement of the sensitivity of our model as compared to the size of changes in the data it is analyzing. 
    03BClticollinearity is strongly implied by a high condition number. 
    03BClticollinearity a term to describe two or more independent variables that are strongly related to each other and are falsely affecting our predicted variable by redundancy.
'''

import re
import os
import math
import numpy as np
import ast
import csv
import pandas as pd
import itertools
import openpyxl
 

import pingouin as pg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from statsmodels.stats.multitest import multipletests


import matplotlib.pyplot as plt
import statsmodels.api as sma
# import statsmodels.formula.api as sm
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.compat import lzip
from PIL import Image



from scipy.stats import loguniform

def mean_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean scores of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """
    
    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append(round(mean_scores[i], 5))

    return pd.Series(data=out_col, index=mean_scores.index)

def calculate_vif(x):
    thresh = 50
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no.")
        print(i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:")
        print(a)
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)

def map_ranges_to_unity(ranges):
    # Find the minimum and maximum values from all the ranges
    min_value = min(ranges, key=lambda x: x[0])[0]
    max_value = max(ranges, key=lambda x: x[1])[1]

    # Calculate the range span
    range_span = max_value - min_value

    # Map each range to a value between 0 and 1
    mapped_ranges = []
    for vals in ranges:
        start= vals[0]
        end = vals[1]
        mapped_start = (start - min_value) / range_span
        mapped_end = (end - min_value) / range_span
        mapped_ranges.append((mapped_start, mapped_end))

    return range_span, max_value, min_value, mapped_ranges


def find_mapped_position(value, max_val, min_val):
    position = (value - min_val) / (max_val - min_val)
    return position 




# Example usage
Sens_ranges = [
    (2.73,	3.02),
    (0.53,	0.59),
    (2.95,	3.26),
    (0.44,	0.48),
    (25934.05,	46291.35),
    (4.75,	15.75),
    (14.25,	52.50),
    (285.00,	525.00),
    (0.19,	0.84),
    (10,32,	11.41),
    (16.72,	18.47),
    (3.98,	4.40),
    (16.28,	17.99),
    (3.93,	4.35),
    (8.62,	9.53),
    (2.86,	3.16),
    (9.54,	10.55),
    (2.24,	2.47),
    (7.10,	7.85),
    (0.98,	1.08),
    (4.58,	5.06),
    (2.09,	2.31),
    (7.49,	8.27),
    (2.26,	2.50),
    (0.0095,	0.0105),
    (0.2116,	0.2338),
    (0.0207,	0.0229),
    (0.78,	0.86),
    (0.45,	0.50),
    (2.05,	2.27),
    (1.40,	1.54),
    (4.54,	5.02),
    (1.12,	1.23),
    (3.12,	3.45),
    (1.53,	1.69),
    (4.30,	4.75),
    (1.04,	1.15),
    (1.86,	2.05),
    (1.33,	1.47),
    (5.98,	6.61),
    (2.38,	2.63)
        ]

param_name = {'dup_init_age_mean': 'Warmup\n Initiation Age \u03BC',
'dup_init_age_sig':'Warmup \nInitiation Age \u03C3',
'dup_prev_age_mean':'Warmup \nPrevalence Age \u03BC',
'dup_prev_age_sig':'Warmup \nPrevalence Age \u03C3',
'Starting_Pop_Size': 'Warmup \nTotal Starting Population',
'Starting_CJS_UT_Size':'Warmup \nExpected CJS \nStarting Population',
'Starting_Hosp_UT_Size':'Warmup \nExpected HE \nStarting Population',
'Starting_Treat_UT_Size':'Warmup \nExpected Treat \nStarting Population',
'starting_inactive_UT_Size':'Warmup prob. \nIndividual Starts \nin Inactive State',
'lam_user_arrival':'Arc (1) \u03BB',
'LNmean_deathdays_pre2019':'Arc (2) \npre 2019 \u03BC',
'LNsigma_deathdays_pre2019':'Arc (2) \npre 2019 \u03C3',
'LNmean_deathdays_post2019':'Arc (2) \npost 2019 \u03BC',
'LNsigma_deathdays_post2019':'Arc (2) \npost 2019 \u03C3',
'LNmean_hospdays':'Arc (3) \u03BC',
'LNsigma_hospdays':'Arc (3) \u03C3',
'LNmean_Oarrestdays':'Arc (4) \u03BC',
'LNsigma_Oarrestdays':'Arc (4) \u03C3',
'LNmean_treatdays':'Arc (5) \u03BC',
'LNsigma_treatdays':'Arc (5) \u03C3',
'LNmean_iadays':'Arc (6) \u03BC',
'LNsig_iadays':'Arc (6) \u03C3',
'LNmean_nonOarrestdays':'Arc (8) \u03BC',
'LNsigma_nonOarrestdays':'Arc (8) \u03C3',
'P(CJS | HE)':'$p_A$',
'P(Treatment | HE)':'$p_T$',
'1 - P(Death|HE)':'$p_D$',
'LNmean_hospservice':'Arc (A) \u03BC',
'LNsig_hospservice':'Arc (A) \u03C3',
'LNmean_crimeservice':'Arc (B) \u03BC',
'LNsig_crimeservice':'Arc (B) \u03C3',
'LNmean_treatservice':'Arc (C) \u03BC',
'LNsig_treatservice':'Arc (C) \u03C3',
'LNmean_crimerel':'Arc (D) \u03BC',
'LNsig_crimerel':'Arc (D) \u03C3',
'LNmean_treatrel':'Arc (E) \u03BC',
'LNsig_treatrel':'Arc (E) \u03C3',
'LNmean_hosprel':'Arc (F) \u03BC',
'LNsig_hosprel':'Arc (F) \u03C3',
'LNmean_iarel':'Arc (G) \u03BC',
'LNsig_iarel':'Arc (G) \u03C3'
}
param_name_latex = {'dup_init_age_mean': 'Warmup\n Initiation Age $\mu$',
'dup_init_age_sig':'Warmup \nInitiation Age $\sigma$',
'dup_prev_age_mean':'Warmup \nPrevalence Age $\mu$',
'dup_prev_age_sig':'Warmup \nPrevalence Age $\sigma$',
'Starting_Pop_Size': 'Warmup \nTotal Starting Population',
'Starting_CJS_UT_Size':'Warmup \nExpected CJS \nStarting Population',
'Starting_Hosp_UT_Size':'Warmup \nExpected HE \nStarting Population',
'Starting_Treat_UT_Size':'Warmup \nExpected Treat \nStarting Population',
'starting_inactive_UT_Size':'Warmup prob. \nIndividual Starts \nin Inactive State',
'lam_user_arrival':'Arc (1) $\lambda$',
'LNmean_deathdays_pre2019':'Arc (2) \npre 2019 $\mu$',
'LNsigma_deathdays_pre2019':'Arc (2) \npre 2019 $\sigma$',
'LNmean_deathdays_post2019':'Arc (2) \npost 2019 $\mu$',
'LNsigma_deathdays_post2019':'Arc (2) \npost 2019 $\sigma$',
'LNmean_hospdays':'Arc (3) $\mu$',
'LNsigma_hospdays':'Arc (3) $\sigma$',
'LNmean_Oarrestdays':'Arc (4) $\mu$',
'LNsigma_Oarrestdays':'Arc (4) $\sigma$',
'LNmean_treatdays':'Arc (5) $\mu$',
'LNsigma_treatdays':'Arc (5) $\sigma$',
'LNmean_nonOarrestdays':'Arc (8) $\mu$',
'LNsigma_nonOarrestdays':'Arc (8) $\sigma$',
'1 - P(Death|HE)':'$p_D$',
'P(CJS | HE)':'$p_A$',
'P(Treatment | HE)':'$p_T$',
'LNmean_iadays':'Arc (6) $\mu$',
'LNsig_iadays':'Arc (6) $\sigma$',
'LNmean_hospservice':'Arc (A) $\mu$',
'LNsig_hospservice':'Arc (A) $\sigma$',
'LNmean_crimeservice':'Arc (B) $\mu$',
'LNsig_crimeservice':'Arc (B) $\sigma$',
'LNmean_treatservice':'Arc (C) $\mu$',
'LNsig_treatservice':'Arc (C) $\sigma$',
'LNmean_crimerel':'Arc (D) $\mu$',
'LNsig_crimerel':'Arc (D) $\sigma$',
'LNmean_treatrel':'Arc (E) $\mu$',
'LNsig_treatrel':'Arc (E) $\sigma$',
'LNmean_hosprel':'Arc (F) $\mu$',
'LNsig_hosprel':'Arc (F) $\sigma$',
'LNmean_iarel':'Arc (G) $\mu$',
'LNsig_iarel':'Arc (G) $\sigma$'
}
sens_range_span, max_val, min_val, mapped_sens_ranges = map_ranges_to_unity(Sens_ranges)


''' Import Sensitivity Paramerters and Baseline Output Folder'''

dirname = os.path.dirname(__file__)
df_Params = pd.read_csv (r'Sensitivity_Runs\Sens_Params.csv')
# print(df_Params)

df_ServiceTimes_Output = pd.read_csv(r'Sensitivity_Runs\ Avg_Service_Times.csv')
df_Sens_ServiceTimes =pd.merge(df_Params,df_ServiceTimes_Output,left_on='Run', right_on="Run") 

df_BaselineOutput = pd.read_csv(r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_ServiceTimes.csv')
# print(df_Output)

num_runs = df_Sens_ServiceTimes.shape[0]  # Gives number of rows
print(df_Sens_ServiceTimes) #confirmed that these are joining correctly

''' Creates Histograms of Uncertainty Analysis with Baseline Outputs'''      
#'''
#The following prints histograms of the sampled parameter distributions and Output Distributions
for idx, col in enumerate(df_Sens_ServiceTimes):
    if idx == 0:
        pass
    else:
        # print(col)
        fig, ax = plt.subplots()
        df_Sens_ServiceTimes.hist(column=col, figsize=(10,8), alpha=0.5, bins=15, ax=ax, label ='Uncertainty Analysis')
        if idx < 40: #labels parameter sampling histogram
            dist_dir= os.path.join(dirname, 'Distributions\\Params\\')
            try: 
                os.makedirs(dist_dir)
            except:
                pass
            plt_file_name = 'hist_col' + str(idx) + '.jpeg'
        else: #labels output service time histograms
            dist_dir= os.path.join(dirname, 'Distributions\\Outputs\\')
            try: 
                os.makedirs(dist_dir)
            except:
                pass
            if col == "Enter Age":
                df_BaselineOutput.hist(column=col, alpha=0.5, bins= 30, color='green', ax=ax, label = 'Baseline Model')
                plt.title("Histogram of Average " + col + " (N = " + str(num_runs) +")", fontsize =16 )
                plt.xlabel("Average " + col, fontsize =16)
                plt.legend(loc='best')
            elif col == "InactiveAll State": #need to calcualted InactiveAll column in main simulation
                df_BaselineOutput.hist(column=col, alpha=0.8, bins= 10, color='green', ax=ax, label = 'Baseline Model')
                plt.title("Histogram of Average " + col + " (N = " + str(num_runs) +")", fontsize =16 )
                plt.xlabel("Average " + col, fontsize =16)
            elif col == 'InactiveCrime State':
                plt.axvline(100, color="green", linestyle="solid", alpha=0.8)
                df_BaselineOutput.hist(column=col, alpha=0.8, bins= 10, color='green', ax=ax, label = 'Baseline Model')
                plt.title("Histogram of Average Number of Days \n in " + col + " (N = " + str(num_runs) +")", fontsize =16)
                plt.xlabel("Average Number of Days in " + col, fontsize =16)
                plt.legend(loc='best')
                
            else:
                df_BaselineOutput.hist(column=col, alpha=0.8, bins= 10, color='green', ax=ax, label = 'Baseline Model')
                plt.title("Histogram of Average Number of Days \n in " + col + " (N = " + str(num_runs) +")", fontsize =16)
                plt.xlabel("Average Number of Days in " + col, fontsize =16)
                plt.legend(loc='best')
            plt.ylabel("Frequency", fontsize =16)
            plt_file_name = 'hist_col' + str(idx) + '.jpeg'
        plt.savefig(dist_dir + plt_file_name)
        plt.close()
#'''
''' Running Linear Regression of Parameters on Service Time Outputs'''
#'''
df_x_params = df_Params.drop(columns=['Run'])
params_list = list(df_x_params.columns)
Results = ['Col', 'X_0', params_list]
for output_col in df_ServiceTimes_Output:
    if output_col == 'Run':
        pass
    else:
        x_train,x_test,y_train,y_test = train_test_split(df_x_params, df_ServiceTimes_Output[output_col],test_size = 0.2,random_state = 100) 
        lm1 = LinearRegression()
        lm1.fit(x_train,y_train)
        coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(lm1.coef_))], axis = 1)
        # r_sq = lm1.score(df_x_params, output_col)
        # print(f"coefficient of determination: {r_sq}")
        # print(f"intercept: {lm1.intercept_}")
        # print(f"slope: {lm1.coef_}")
        Results.append([output_col,lm1.intercept_,lm1.coef_])
        y_pred = lm1.predict(x_test)
        y_error = y_test - y_pred
        print(r2_score(y_test,y_pred))
        ### lm2 #####
        X_train = sma.add_constant(x_train) ## let's add an intercept (beta_0) to our model
        X_test = sma.add_constant(x_test) 
        lm2 = sm.OLS(y_train,X_train).fit()
        print(lm2.summary())
        y_pred2 = lm2.predict(X_test)
        #### Detecting and Removing 03BClticollinearity : https://www.listendata.com/2018/01/linear-regression-in-python.html
        ## Ridge regression
        results = {}
        numeric_features = list(set(x_train.columns))
        preprocessor= make_column_transformer( (StandardScaler(), numeric_features))

        feature_names = numeric_features
        models = {
            'dummy_reg': make_pipeline(preprocessor, DummyRegressor()),
            'base_linear': make_pipeline(preprocessor, LinearRegression()),
            'base_ridge': make_pipeline(preprocessor, Ridge(random_state=2018)),
            'base_lasso': make_pipeline(preprocessor, Lasso(random_state=2018))
        }
        results["DummyRegressor"] = mean_cross_val_scores(models['dummy_reg'], X_train, y_train,
                                                     return_train_score=True)
        results["BaseLinear"] = mean_cross_val_scores(models['base_linear'], X_train, y_train,
                                                     return_train_score=True)
        results["BaseRidge"] = mean_cross_val_scores(models['base_ridge'], X_train, y_train, 
                                                     return_train_score=True)
        results["BaseLasso"] = mean_cross_val_scores(models['base_lasso'], X_train, y_train,
                                                     return_train_score=True)
        
        param_grid = {
            "lasso__alpha": [pow(10, x) for x in range(-2, 2, 1)]
        }
        tuned_lasso = GridSearchCV(
            models['base_lasso'], param_grid, n_jobs=-1, return_train_score=True
        )
        tuned_lasso.fit(X_train, y_train);
        print(tuned_lasso.best_params_)
        models['tuned_lasso'] = make_pipeline(preprocessor,
                                      Lasso(alpha=tuned_lasso.best_params_['lasso__alpha'], random_state=2018))
        results["TunedLasso"] = mean_cross_val_scores(models['tuned_lasso'], X_train, y_train,
                                                     return_train_score=True)
        results_df = pd.DataFrame(results)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
        feature_coef_lasso = pd.DataFrame(
        data={
            "Lasso Coefficient": models['tuned_lasso'].named_steps["lasso"].coef_.flatten()
        },
        index=feature_names
        ).sort_values("Lasso Coefficient", ascending=False)

        feature_coef_lasso
        feature_coef_ridge = pd.DataFrame(
            data={
            "Ridge Coefficient": models['base_ridge'].named_steps["ridge"].coef_.flatten(),
            },
            index=feature_names
        ).sort_values("Ridge Coefficient", ascending=False)

        feature_coef_linear = pd.DataFrame(
            data={
                "Linear Coefficient": models['base_linear'].named_steps["linearregression"].coef_.flatten(),
            },
            index=feature_names
        ).sort_values("Linear Coefficient", ascending=False)
        pd.concat([feature_coef_linear, feature_coef_lasso, feature_coef_ridge], axis=1).sort_values("Lasso Coefficient", ascending=False)
        # Caluclating the test scores for all the models
        results = {'DummyRegressor': models['dummy_reg'].score(X_test, y_test),
                'BaseLinear': models['base_linear'].score(X_test, y_test),
                'BaseRidge': models['base_ridge'].score(X_test, y_test),
                'BaseLasso': models['base_lasso'].score(X_test, y_test),
                'TunedLasso': models['tuned_lasso'].score(X_test, y_test)
                }
        test_results = pd.DataFrame(data=results, index=['test_score'])
        index_as_list = results_df.index.tolist()
        idx = index_as_list.index('test_score')
        index_as_list[idx] = 'validaton_score'
        #Caluclating the test scores for all the models
        results = {'DummyRegressor': models['dummy_reg'].score(X_test, y_test),
                'BaseLinear': models['base_linear'].score(X_test, y_test),
                'BaseRidge': models['base_ridge'].score(X_test, y_test),
                'BaseLasso': models['base_lasso'].score(X_test, y_test),
                'TunedLasso': models['tuned_lasso'].score(X_test, y_test)
                }
        test_results = pd.DataFrame(data=results, index=['test_score'])
        # Appending the results in the earlier dataframe
        results_df = results_df.append(test_results)
        print(results_df)



        ## option : Lasso Regression
        # Create and fit the Lasso regression model
        lasso = Lasso(alpha=1)  # Set the regularization parameter alpha
        lasso.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = lasso.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)
        ## option 2: Variance inflation factor https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
        [variance_inflation_factor(x_train.values, j) for j in range(x_train.shape[1])]
        train_out = calculate_vif(x_train)
        column_names = train_out.columns.tolist()
        train_out = sma.add_constant(train_out) ## let's add an intercept (beta_0) to our model
        x_test = x_test.loc[:, column_names]
        X_test = sma.add_constant(x_test)
        lm2 = sm.OLS(y_train,train_out).fit()
        lm2.summary() 
        #### Checking normality of residuals/error terms We use Shapiro Wilk test from scipy library to check the normality of residuals.Null Hypothesis: The residuals are normally distributed.
        #### Checking for autocorrelation To ensure the absence of autocorrelation we use Ljungbox test. Null Hypothesis: Autocorrelation is absent.
        #### Checking heteroscedasticity Using Goldfeld Quandt we test for heteroscedasticity, Null Hypothesis: Error terms are homoscedastic
        name = ['F statistic', 'p-value']
        test = sms.het_goldfeldquandt(lm2.resid, lm2.model.exog)
        print(lzip(name, test))
# print(Results)
with open("Sensitivity_Runs\RegressionResults.csv", "w", newline="") as f: #for uncertainty and sensitivity analysis
        writer = csv.writer(f)
        writer.writerows(Results)
        f.close()
#'''
''' Standard Outputs'''
Sens_output_file_list = [r'Sensitivity_Runs\ Avg_Active_YearEnd.csv',
r'Sensitivity_Runs\ Avg_OArrests.csv',
r'Sensitivity_Runs\ Avg_Hosp.csv',
r'Sensitivity_Runs\ Avg_Inactive_YearEnd.csv',
r'Sensitivity_Runs\ Avg_Odeaths.csv',
r'Sensitivity_Runs\ Avg_Prevalence.csv',
r'Sensitivity_Runs\ Avg_Relapses.csv',
r'Sensitivity_Runs\ Avg_Treats.csv']


Base_output_file_list = [r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_Active_YearEnd.csv',
r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_OArrests.csv',
r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_Hosp.csv',
r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_Inactive_YearEnd.csv',
r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_ODeaths.csv',
r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_Prevalence.csv',
r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_Relapses.csv',
r'Results_0Process_ED2RVal_22270_MARIVal_0_CMVal_0_Scen_600_Years_25_Time_051724_203606\summaries600\Base_Yearly_Treats.csv']

output_str = ["Active", "OArrest", "Hosp", "Inactive", "ODeaths", "Prev", "Relapses","Treats"] 
years_col = ["Year_9","Year_11", "Year_25"]
years_str = ["2016","2018", "2032"]
main_output_count = len(Sens_ranges)
alpha = 0.5
bon_corr = 0.5/main_output_count
n = 600
dist_dir = r'Sensitivity_Runs\Scatter_Plots'

df_PRCC = pd.DataFrame()
num_tests = 41
df_OLS = []
qqplot_images = []

for f_idx, f in enumerate(Sens_output_file_list):
    #read in csv files for main output analysis
    df_Sens = pd.read_csv(Sens_output_file_list[f_idx]) #sensitivity  results. 
    df_Base = pd.read_csv(Base_output_file_list[f_idx]) # main analysis in paper results
    ### Create Q-Q plots
    sm.qqplot(df_Base[years_col[0]][0:n], line="45", fit=True)
    plt.title(f"Q-Q plot of year {years_str[0]} Number of {output_str[f_idx]}, n = {len(df_Base[years_col[0]][0:n])}")
    plt.savefig(f"{dist_dir}\{output_str[f_idx] + years_str[0]}")
    plt.close()
    # Open the and append image
    if output_str[f_idx]  in ["Active", "OArrest", "Hosp", "ODeaths", "Treats"]: 
        image = Image.open(f"{dist_dir}\{output_str[f_idx] + years_str[0]}.png")
        qqplot_images.append(image)
    sm.qqplot(df_Base[years_col[1]][0:n], line="45", fit=True)
    plt.title(f"Q-Q plot of year {years_str[1]} Number of {output_str[f_idx]}, n = {len(df_Base[years_col[1]][0:n])}")
    plt.savefig(f"{dist_dir}\{output_str[f_idx] + years_str[1]}")
    plt.close()
    # Open the and append image
    if output_str[f_idx]  in ["Active", "OArrest", "Hosp", "ODeaths", "Treats"]: 
        image = Image.open(f"{dist_dir}\{output_str[f_idx] + years_str[1]}.png")
        qqplot_images.append(image)
    sm.qqplot(df_Base[years_col[2]][0:n], line="45", fit=True)
    plt.title(f"Q-Q plot of year {years_str[2]} Number of {output_str[f_idx]}, n = {len(df_Base[years_col[2]][0:n])}")
    plt.savefig(f"{dist_dir}\{output_str[f_idx] + years_str[2]}")
    plt.close()
    # Open the and append image
    if output_str[f_idx]  in ["Active", "OArrest", "Hosp", "ODeaths", "Treats"]: 
        image = Image.open(f"{dist_dir}\{output_str[f_idx] + years_str[2]}.png")
        qqplot_images.append(image)

    # Observing Year 2017 and 2032 for each output 
    df_Year_8_main = df_Sens[['Run',years_col[0]]]
    df_Year_10_main = df_Sens[['Run',years_col[1]]]
    df_Year_24_main = df_Sens[['Run',years_col[2]]]
    #combine with list of x_variables 
    df_Year_8_main = pd.merge(df_Year_8_main,df_Params, on="Run")
    df_Year_10_main = pd.merge(df_Year_10_main,df_Params, on="Run")
    df_Year_24_main = pd.merge(df_Year_24_main,df_Params, on="Run")
    dfs_year = [df_Year_8_main,df_Year_10_main,df_Year_24_main]
    #''' create histograms
    for idx, col in enumerate(df_Sens):
        if col in years_col:
            ## need to create data frames of all parameters as X_i and y as output
            fig, ax = plt.subplots()
            # plt.axvline(x = df_Sens.loc[0][col],color = 'r', label = 'axvline - full height')
            df_Sens.hist(column=col, figsize=(10,8), alpha=0.5, bins=15, ax=ax, label ='Uncertainty Analysis')
            df_Base.hist(column=col, alpha=0.8, bins= 10, color='green', ax=ax, label = 'Baseline Model')
            plt.title("Histogram of Average Number of \n in " + col + " (N = " + str(num_runs) +")", fontsize =16)
            plt.xlabel("Average Number of Days in " + col, fontsize =16)
            plt.legend(loc='best')
            plt.ylabel("Frequency", fontsize =16)
            plt_file_name = 'hist_' + output_str[f_idx]+"_" + str(col) + '.jpeg'
            plt.savefig(dist_dir + plt_file_name)
            plt.close()
        else:
            pass
    #'''
    
    df_OLS.append([])
    
    for y_idx, year in enumerate(years_col):
        df = dfs_year[y_idx]
        df_cols = list(df.columns)
        df_cols = df_cols[2:]
        #''' create scatterplots
        # Create a list to hold the scatterplot images for the current year
        scatterplot_images = []
        plt.rcParams.update({'font.size': 30})
        # make a scatterplot of each input vs a given output
        for idx, col in enumerate(df):
            if idx in [0,1]:
                out_col = col
                out_str =  years_str[y_idx] +  " " + output_str[f_idx]
                out_idx = idx
                continue
            # Create scatter plot
            plt.scatter(df[col],df[out_col])

            # Set labels and title
            #plt.ylabel(out_str)

            plt.xlabel('Parameter: ' + param_name[col])
            #plt.title('Scatter Plot of ' + col + " vs. " + out_str)

            # Replace characters
            col = col.replace("|", "_")
            col = col.replace("(", "")
            col = col.replace(")", "")

            plt_file_name = '\scat_' + out_str +"_" + col + '.jpeg'
             # Check if the path exists
            try:
                scat_dir = os.path.join(dirname, 'Scatter_Plots\\', year)
                # Create the directory if it doesn't exist
                os.makedirs(scat_dir)
            except:
                pass
            plt.tight_layout()
            plt.savefig(scat_dir + plt_file_name)
            plt.close()
            
            # Open the scatterplot image
            image = Image.open(scat_dir + plt_file_name)
            
            # Append the scatterplot image to the list
            scatterplot_images.append(image)

            # Rest of the code...
        
        # Specify the number of columns and rows in the grid
        num_columns = 4
        num_rows = 10

        combined_image = Image.new('RGB', (2100,2850),color='white')
        # Calculate the size of each grid cell
        cell_width = combined_image.width // num_columns
        cell_height = combined_image.height // num_rows

        # Create a new blank image for the combined grid
        grid_width = num_columns * cell_width
        grid_height = num_rows * cell_height
        combined_grid = Image.new('RGB', (grid_width, grid_height),color='white')
        # Iterate over the scatterplot images and paste them onto the blank image
        for img_idx,image in enumerate(scatterplot_images):
            # Resize the scatterplot image to fit the grid cell size
            scatterplot_image_resized = image.resize((cell_width, cell_height))

            # Calculate the column and row indices for the current image
            col_idx = img_idx % num_columns
            row_idx = img_idx // num_columns

            # Calculate the paste position for the current image
            paste_x = col_idx * cell_width
            paste_y = row_idx * cell_height
            image.resize((cell_width, cell_height))
            # Paste the resized scatterplot image onto the grid
            combined_grid.paste(scatterplot_image_resized, (paste_x, paste_y))

            combined_file_name = f"combined_image_{f_idx}.jpeg"
            combined_grid.save(scat_dir + combined_file_name)
        #'''

        #''' Steps. caclualte partial rank correlation coefficients for allvariable adn corresponding p-values 
        ### using Pingouin
        ### https://pingouin-stats.org/build/html/generated/pingouin.partial_corr.html#pingouin.partial_corr
        
        for idx2, col in enumerate(df_cols):
            df_temp = pg.partial_corr(data=df, x=df_cols[idx2], y=years_col[y_idx], alternative="two-sided", method="spearman")
            df_temp.insert(0,'Input', col )
            df_temp.insert(0,'Output', years_col[y_idx] + " " + output_str[f_idx])
            df_PRCC = pd.concat([df_PRCC, df_temp])

            # Apply find_mapped_position() function
            mapped_position = df[col].apply(find_mapped_position, args=(max_val, min_val))
            
            # Apply lognormal transformation
            transformed_input = np.log(mapped_position)
            
            #''' Calcualte OLS regression
            X = sm.add_constant(transformed_input)
            #X = sm.add_constant(df[col].apply(find_mapped_position, args=(max_val, min_val)))
            model = sm.OLS(np.log(df[years_col[y_idx]]), X)
            results = model.fit()

            # Extract the effect size (slope coefficient)
            if y_idx == 0:
                df_OLS[f_idx].append([])
            df_OLS[f_idx][idx2].append({})
            df_OLS[f_idx][idx2][y_idx]["Output"] = output_str[f_idx]
            df_OLS[f_idx][idx2][y_idx]["Year"] = years_str[y_idx]  
            df_OLS[f_idx][idx2][y_idx]["Param"]= col
            df_OLS[f_idx][idx2][y_idx]["effect_size"]= round(results.params[col],3)
            # Extract the p-value for the coefficient estimate
            df_OLS[f_idx][idx2][y_idx]["p_value"] = round(results.pvalues[col],3)
            '''
            
            
        #'''

## combined qq plot
# Specify the number of columns and rows in the grid
num_columns = 3
num_rows = 5

combined_image = Image.new('RGB', (2100,2850),color='white')
# Calculate the size of each grid cell
cell_width = combined_image.width // num_columns
cell_height = combined_image.height // num_rows

# Create a new blank image for the combined grid
grid_width = num_columns * cell_width
grid_height = num_rows * cell_height
combined_grid = Image.new('RGB', (grid_width, grid_height),color='white')
# Iterate over the scatterplot images and paste them onto the blank image
for img_idx,image in enumerate(qqplot_images):
    # Resize the scatterplot image to fit the grid cell size
    scatterplot_image_resized = image.resize((cell_width, cell_height))

    # Calculate the column and row indices for the current image
    col_idx = img_idx % num_columns
    row_idx = img_idx // num_columns

    # Calculate the paste position for the current image
    paste_x = col_idx * cell_width
    paste_y = row_idx * cell_height
    image.resize((cell_width, cell_height))
    # Paste the resized scatterplot image onto the grid
    combined_grid.paste(scatterplot_image_resized, (paste_x, paste_y))

    combined_file_name = f"combined_qqplot.jpeg"
    combined_grid.save(f"{dist_dir}\ {combined_file_name}")

#'''
df_PRCC['p-val'] = multipletests(df_PRCC['p-val'], method='bonferroni')[1]
#df_PRCC['p-val'] = df_PRCC['p-val'].apply(lambda x: x*num_tests)
df_PRCC.to_excel("Sensitivity_Runs\PRCC_output.xlsx")  
DF_OLS = pd.DataFrame(df_OLS)
DF_OLS.to_excel("Sensitivity_Runs\OLS_output.xlsx")  




#''' Create Output Table of PRCC for paper
PRCC_table_list = []
for idx,item in enumerate(param_name):
    PRCC_table_list.append({})
    PRCC_table_list[idx]["Param_num"] = idx
    PRCC_table_list[idx]["Param_code"] = item
    PRCC_table_list[idx]["Param"] = param_name[item]
for idx,row in df_PRCC.iterrows():
    for year in years_col:
        if year in row[0]:
            for output in output_str:
                if output in row[0]:
                    for idx,item in enumerate(param_name):
                        if PRCC_table_list[idx]["Param_code"] == row[1]:
                            PRCC_table_list[idx][output+"_coe_"+year] = round(row[3],2)
                            PRCC_table_list[idx][output+"_pval_"+year] = round(row[5],3)


from Sens_Tables import PRCC_Results_Table
table = PRCC_Results_Table(PRCC_table_list,param_name_latex, years_col)
with open('Sensitivity_Runs\PRCC_Results.txt', 'w') as f:
    f.write(table)

from Sens_Tables import OLS_Results_Table
table = OLS_Results_Table(df_OLS,param_name_latex,PRCC_table_list, years_col)
with open('Sensitivity_Runs\OLS_Results.txt', 'w') as f:
    f.write(table)


print("done?")


# maps for linear regression
#value = 25
#position = find_mapped_position(value, Sens_ranges, mapped_sens_ranges, sens_range_span,max_val, min_val,)

''' Creating Tables of CI intervals of Model Outputs'''
#The following Outputs a CSV file of the CI intervals of Model Outputs
df1 = pd.DataFrame()
# for idx, col in enumerate(df_Output):
#     df1[col] = df_Output[col].describe()
#     df1 = df1[col].append()
# print(df1)

