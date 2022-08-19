import sys
import math
import numpy as np
from operator import itemgetter
import time
import copy
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale

#############################################################################
#
# Global parameters
#
#####################

target_idx=0                                        #Index of Target variable
cross_val=1                                         #Control Switch for CV
norm_target=0                                       #Normalize target switch
norm_features=1                                     #Normalize target switch
feat_select=0                                       #Control Switch for Feature Selection
fs_type=4                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=0                                         #Control switch for low variance filter on features
feat_start=1                                        #Start column of features
k_cnt=5                                             #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3

#Set global model parameters
rand_st=1                                           #Set Random State variable for randomizing splits on runs


#############################################################################
#
# Load Data
#
#############################################################################
root_dir = '/workspace/GradSchool/2020-2/Adv_ML/Final/data/processed_data'

train_stat_file = root_dir + '/train_stat.csv'
train_result_file = root_dir + '/EPL_Results_from_2014_to_2019.csv'
test_stat_file = root_dir + '/test_stat.csv'
test_result_file = root_dir + '/EPL_Result_2019-20.csv'

train_stat = pd.read_csv(train_stat_file)
train_result = pd.read_csv(train_result_file)
test_stat = pd.read_csv(test_stat_file)
test_result = pd.read_csv(test_result_file)

stat_col = list(train_stat.columns)
result_col = list(train_result.columns)

#############################################################################
#
# Preprocess data
#
#############################################################################


col_list = ['Appearances', 'Clean sheets', 'Goals conceded', 'Tackles', 'Last man tackles', 'Blocked shots',
            'Interceptions', 'Clearances', 'Headed Clearance','Clearances off line', 'Recoveries', 'Duels won',
            'Duels lost', 'Successful 50/50s', 'Aerial battles won', 'Aerial battles lost', 'Own goals',
            'Errors leading to goal', 'Assists', 'Passes per match', 'Big chances created', 'Crosses', 'Through balls',
            'Accurate long balls', 'Yellow cards', 'Red cards', 'Fouls', 'Offsides', 'Hit woodwork', 'Goals per match',
            'Penalties scored', 'Freekicks scored', 'Shots', 'Shots on target', 'Big chances missed', 'Saves',
            'Penalties saved', 'Punches', 'High Claims', 'Catches', 'Sweeper clearances', 'Throw outs', 'Goal Kicks',
            'minutes_played']

if norm_features == 1:
    for col in col_list:
        train_stat[col] = scale(train_stat[col])
        test_stat[col] = scale(test_stat[col])

#############################################################################
#
# Feature Selection
#
#############################################################################


if lv_filter == 1:
    print('--LOW VARIANCE FILTER ON--', '\n')

    # LV Threshold
    sel = VarianceThreshold(threshold=0.5)  # Removes any feature with less than 20% variance
    fit_mod = sel.fit(data_np)
    fitted = sel.transform(data_np)
    sel_idx = fit_mod.get_support()

    # Get lists of selected and non-selected features (names and indexes)
    temp = []
    temp_idx = []
    temp_del = []
    for i in range(len(data_np[0])):
        if sel_idx[i] == 1:  # Selected Features get added to temp header
            temp.append(header[i + feat_start])
            temp_idx.append(i)
        else:  # Indexes of non-selected features get added to delete array
            temp_del.append(i)

    print('Selected', temp)
    print('Features (total, selected):', len(data_np[0]), len(temp))
    print('\n')

    # Filter selected columns from original dataset
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)  # Deletes non-selected features by index

# Feature Selection
if feat_select == 1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''

    print('--FEATURE SELECTION ON--', '\n')

    ##1) Run Feature Selection #######
    if fs_type == 1:
        # Stepwise Recursive Backwards Feature removal
        if binning == 1:
            clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy',
                                         random_state=rand_st)
            sel = RFE(clf, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')
        if binning == 0:
            rgr = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=3, criterion='mse',
                                        random_state=rand_st)
            sel = RFE(rgr, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')

        fit_mod = sel.fit(data_np, target_np)
        print(sel.ranking_)
        sel_idx = fit_mod.get_support()

    if fs_type == 2:
        # Wrapper Select via model
        if binning == 1:
            clf = '''Unused in this homework'''
            sel = SelectFromModel(clf, prefit=False, threshold='mean',
                                  max_features=None)  # to select only based on max_features, set to integer value and set threshold=-np.inf
            print('Wrapper Select: ')
        if binning == 0:
            rgr = SVR(kernel='linear', gamma=0.1, C=1.0)
            sel = SelectFromModel(rgr, prefit=False, threshold='mean', max_features=None)
            print('Wrapper Select: ')

        fit_mod = sel.fit(data_np, target_np)
        sel_idx = fit_mod.get_support()

    if fs_type == 3:
        if binning == 1:  ######Only work if the Target is binned###########
            # Univariate Feature Selection - Chi-squared
            sel = SelectKBest(chi2, k=k_cnt)
            fit_mod = sel.fit(data_np,
                              target_np)  # will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
            print('Univariate Feature Selection - Chi2: ')
            sel_idx = fit_mod.get_support()

        if binning == 0:  ######Only work if the Target is continuous###########
            # Univariate Feature Selection - Mutual Info Regression
            sel = SelectKBest(mutual_info_regression, k=k_cnt)
            fit_mod = sel.fit(data_np, target_np)
            print('Univariate Feature Selection - Mutual Info: ')
            sel_idx = fit_mod.get_support()

        # Print ranked variables out sorted
        temp = []
        scores = fit_mod.scores_
        for i in range(feat_start, len(header)):
            temp.append([header[i], float(scores[i - feat_start])])

        print('Ranked Features')
        temp_sort = sorted(temp, key=itemgetter(1), reverse=True)
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')

    if fs_type == 4:
        # Full-blown Wrapper Select (from any kind of ML model)
        if binning == 1:  ######Only work if the Target is binned###########
            start_ts = time.time()
            sel_idx = []  # Empty array to hold optimal selected feature set
            best_score = 0  # For classification compare Accuracy or AUC, higher is better, so start with 0
            feat_cnt = len(data_np[0])
            # Create Wrapper model
            clf = '''Unused in this homework'''  # This could be any kind of classifier model

        if binning == 0:  ######Only work if the Target is continuous###########
            start_ts = time.time()
            sel_idx = []  # Empty array to hold optimal selected feature set
            best_score = sys.float_info.max  # For regression compare RMSE, lower is better, so start with max sys float value
            feat_cnt = len(data_np[0])
            # Create Wrapper model
            rgr = SVR(kernel='linear', gamma=0.1, C=1.0)  # This could be any kind of regressor model

        # Loop thru feature sets
        roll_idx = 0
        combo_ctr = 0
        feat_arr = [0 for col in range(feat_cnt)]  # Initialize feature array
        for idx in range(feat_cnt):
            roll_idx = idx
            feat_space_search(feat_arr, idx)  # Recurse
            feat_arr = [0 for col in range(feat_cnt)]  # Reset feature array after each iteration

        print('# of Feature Combos Tested:', combo_ctr)
        print(best_score, sel_idx, len(data_np[0]))
        print("Wrapper Feat Sel Runtime:", time.time() - start_ts)

    ##2) Get lists of selected and non-selected features (names and indexes) #######
    temp = []
    temp_idx = []
    temp_del = []
    for i in range(len(data_np[0])):
        if sel_idx[i] == 1:  # Selected Features get added to temp header
            temp.append(header[i + feat_start])
            temp_idx.append(i)
        else:  # Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected', temp)
    print('Features (total/selected):', len(data_np[0]), len(temp))
    print('\n')

    ##3) Filter selected columns from original dataset #########
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)  # Deletes non-selected features by index)


#############################################################################
#
# calculate function
#
#############################################################################


def sum_stat(data, column_list, num, is_team=False):
    value = []

    if is_team:
        for i in column_list:
            value.append(data[i].sum())
    else:
        for i in column_list:
            value.append(data[i].sum() / len(data) * 3)

    return value


def cal_lineup_value(data, season, team_name, formation=[4, 3, 3]):
    # data: players stat data list, season: str, team_name:str, formation: list
    team_mem = data[data['teams_played_for'] == team_name]
    team_mem = team_mem[team_mem['Season'] == season]

    col = ['Appearances', 'Clean sheets', 'Goals conceded', 'Tackles', 'Last man tackles', 'Blocked shots',
           'Interceptions', 'Clearances', 'Headed Clearance', 'Clearances off line', 'Recoveries', 'Duels won',
           'Duels lost', 'Successful 50/50s', 'Aerial battles won', 'Aerial battles lost', 'Own goals',
           'Errors leading to goal', 'Assists', 'Passes per match', 'Big chances created', 'Crosses', 'Through balls',
           'Accurate long balls', 'Yellow cards', 'Red cards', 'Fouls', 'Offsides', 'Hit woodwork', 'Goals per match',
           'Penalties scored', 'Freekicks scored', 'Shots', 'Shots on target', 'Big chances missed', 'Saves',
           'Penalties saved', 'Punches', 'High Claims', 'Catches', 'Sweeper clearances', 'Throw outs', 'Goal Kicks',
           'minutes_played']

    gk_num = 1
    df_num = formation[0]
    mf_num = formation[1]
    fw_num = formation[2]

    team_value = pd.DataFrame(columns=col)
    temp_team_value = pd.DataFrame(columns=col)

    gk = team_mem[team_mem['Position'] == 'Goalkeeper']
    df = team_mem[team_mem['Position'] == 'Defender']
    mf = team_mem[team_mem['Position'] == 'Midfielder']
    fw = team_mem[team_mem['Position'] == 'Forward']

    # calculate each position's value
    team_gk_value = sum_stat(gk, col, gk_num)
    team_df_value = sum_stat(df, col, df_num)
    team_mf_value = sum_stat(mf, col, mf_num)
    team_fw_value = sum_stat(fw, col, fw_num)

    # calculate 11 player's value
    temp_team_value.loc[0] = team_gk_value
    temp_team_value.loc[1] = team_df_value
    temp_team_value.loc[2] = team_mf_value
    temp_team_value.loc[3] = team_fw_value

    team_value.loc[0] = sum_stat(temp_team_value, col, 0, is_team=True)

    return team_value


#############################################################################
#
# Train SciKit Models
#
#############################################################################

print('--ML Model Output--', '\n')