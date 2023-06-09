"""
                              @author: Johanna Gesperger & Philipp Matten (2023)
                              @email: philipp.matten@gmail.com 

This code runs a Random Rorest Classifier for tumor (TU), infiltration zone (INF), brain parenchyma (BP) classification 
          -> each for the features extracted from the OCT data
          
"""

# imports
import os
import json
import joblib
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix 

from metadataparser import MetaDataParser

cwd, _ = os.path.split(__file__)

MetaDataParser = MetaDataParser(cwd) # contains methods that return meta data for pre-processing and training
_trainings = MetaDataParser._get_experiment_meta_data()


##################################################################################
############### TRAINING FUNCTION FOR RFC AND HYPERPARAM TUNING ##################
##################################################################################          
def tune_hyperparams_and_return_RFC(param_grid, X_train, X_test, y_train, y_test):
     scaler = MinMaxScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     ####### Hyperparameter-Tuning #######
     rf = RandomForestClassifier()
     grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=250, cv=15, verbose=1, random_state=42, n_jobs=-1)
     grid_search.fit(X_train, y_train)
     print(f"[RESULT:] of Decision Tree classifier on testset = {grid_search.score(X_test, y_test):.3f}")
     print(f"[INFO:] Best set of params for current experiment were: {grid_search.best_params_}")
     ####### RETRAIN RFC ON OPTIMAL PARAMS FOUND DURING HYPERPARAM-TUNING AND RETURN RFC OBJECT ######
     rfc = RandomForestClassifier(bootstrap=grid_search.best_params_["bootstrap"], 
                                  n_estimators=grid_search.best_params_["n_estimators"],
                                  max_features=grid_search.best_params_["max_features"],
                                  min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                                  min_samples_split=grid_search.best_params_["min_samples_split"],
                                  max_depth=grid_search.best_params_["max_depth"]
                                  ).fit(X_train, y_train)

     # print training resuls
     return rfc, grid_search


#############################################################
####### SET OF HYPERPARAMETERS FOR TRAINING RFC MODEL #######
#############################################################
param_grid = {
"bootstrap": [True, False],
"max_depth": [10, 20, 30, 40, 60, 80, 100, 150, 200, None],
"max_features": ["auto", "sqrt"],
"min_samples_leaf": [1, 2, 3, 4, 5],
"min_samples_split": [2, 3, 5],
"n_estimators": [100, 200, 400, 600, 1000, 1500, 2000]
}

# #DUMMY LIST
# param_grid = {
# "bootstrap": [True],
# "max_depth": [10, 20],
# "max_features": ["auto", "sqrt"],
# "min_samples_leaf": [1,2],
# "min_samples_split": [2, 3],
# "n_estimators": [200, 300]
# } 


###############################################################################
########################## START PROCESSING LOOP ##############################
###############################################################################

for i, val in enumerate(_trainings):
     
     print(f"\n[INFO:] Processing experiment No.{i}")
     # make log file for current training iteration
     now = datetime.now()
     dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
     log_file_path = os.path.join(cwd, 'LogFiles', dt_string + _trainings[i][3].split('.joblib')[0] + '.txt')
     ###### (ii) LOAD DATA ####### 
     #1) Import feature data from CSV file
     data = pd.read_table(_trainings[i][0])
     #2) Load feature name list from JSON file
     with open(_trainings[i][2], 'r') as f:
          features = json.load(f)
     feature_names = features['feature_group'] # 'feature_group' = name of list in JSON file
     
     test_data, train_data = split_data_train_test(data, _trainings[i][1], percentage=0)
     test_data = test_data.sample(frac = 1)
     train_data = train_data.sample(frac = 1)
     with open(log_file_path, 'a') as f:
          unique_train_patIDs = train_data['PatID'].unique()
          unique_test_patIDs = test_data['PatID'].unique()
          f.write(f"Patient IDs in the TRAINING data set are:\n{unique_train_patIDs}\n")
          f.write(f"Patient IDs in the TEST data set are:\n{unique_test_patIDs}\n")
     
     ###### SETUP TRAINING #######
     X_train = train_data[feature_names]
     X_test = test_data[feature_names]
     #classify accoring to key in >>>_trainings[i][1]<<< 
     y_train = train_data[_trainings[i][1]]
     y_test = test_data[_trainings[i][1]]
     with open(log_file_path, 'a') as f:
          f.write(f"[INFO:] Shapes for whole dataset training {X_train.shape, X_test.shape, y_train.shape, y_test.shape}\n")
          
     ### Train classifier ###
     rfc, grid = tune_hyperparams_and_return_RFC(param_grid, X_train, X_test, y_train, y_test)
     log_grid_search_predictions(grid, log_file_path)

     # Save model trained on all parameters
     print(f"[INFO:] Saving RFC trained on all features from No.{i} to file {_trainings[i][3]}...")
     path_all_params_model = os.path.join(cwd, 'Models', dt_string + "_" + _trainings[i][3])
     joblib.dump(rfc, path_all_params_model)
     # Grab feature importantance of RFC, trained on all features
     feature_importances = rfc.feature_importances_
     # find which features are in the 90th and 50th percentile and map to corresponding names
     idx_feature_importances_90th = np.where(feature_importances > np.percentile(feature_importances, 90))[0]
     idx_feature_importances_50th = np.where(feature_importances > np.percentile(feature_importances, 50))[0]
     feature_names_90th_percentile = list(map(lambda i: feature_names[i], idx_feature_importances_90th))
     feature_names_50th_percentile = list(map(lambda i: feature_names[i], idx_feature_importances_50th))
     feature_importances_90th_percentile = list(map(lambda i: feature_importances[i], idx_feature_importances_90th))
     feature_importances_50th_percentile = list(map(lambda i: feature_importances[i], idx_feature_importances_50th))

     ###### SAVE FEATURE IMPORTANCE/SIGNIFICANCE TO JSON FILE #######
     # JSON contains 3 dictionaries with the feature names and their values -> in order: all TFs, 90th percentile TFs, 50th persentile TFs
     with open(os.path.join(cwd, 'FeatureImportances', dt_string + "_" + _trainings[i][3].split('.joblib')[0] + '_FeatureSignificances.json'), 'w+') as f:
          json.dump(dict(zip(feature_names, feature_importances)), f, indent="")
          json.dump(dict(zip(feature_names_90th_percentile, feature_importances_90th_percentile)), f, indent="")
          json.dump(dict(zip(feature_names_50th_percentile, feature_importances_50th_percentile)), f, indent="")
     
     ######################### RE-RUN TRAINING ON REDUCED NUMBER OF FEATURES ########################################
     # for feature group 4 - i.e. only flourescence as a feature - finding important features is pointless 
     # -> for the other three groups, we re-train the classifier on the identified 90th and 50th percentile feartures
     ################################################################################################################
     if os.path.basename(_trainings[i][2]) != "feature_group_4.json":
          
          print(f"[INFO:] Re-running classification for experiment No.{i} top 50% (50th percentile) of important features...")
          X_train = train_data[feature_names_50th_percentile]
          X_test = test_data[feature_names_50th_percentile]
          y_train = train_data[_trainings[i][1]]
          y_test = test_data[_trainings[i][1]]
          print(f"Shapes for 50th perc. dataset training {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
          rfc50, grid50 = tune_hyperparams_and_return_RFC(param_grid, X_train, X_test, y_train, y_test)
          # Save model trained on all parameters
          log_grid_search_predictions(grid50, log_file_path)
          path_50th_perc_model = os.path.join(cwd, 'Models', dt_string + "_" + _trainings[i][3].split('.joblib')[0] + "_50th_percentile.joblib")
          joblib.dump(rfc50, path_50th_perc_model)
          
          print(f"[INFO:] Re-running classification for experiment No.{i} top 10% (90th percentile) of important features...")
          X_train = train_data[feature_names_90th_percentile]
          X_test = test_data[feature_names_90th_percentile]
          y_train = train_data[_trainings[i][1]]
          y_test = test_data[_trainings[i][1]]
          print(f"Shapes for 90th perc. dataset training {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
          rfc10, grid10 = tune_hyperparams_and_return_RFC(param_grid, X_train, X_test, y_train, y_test)
          log_grid_search_predictions(grid10, log_file_path)
          # Save model trained on all parameters
          path_90th_perc_model = os.path.join(cwd, 'Models', dt_string + "_" + _trainings[i][3].split('.joblib')[0] + "_90th_percentile.joblib")
          joblib.dump(rfc10, path_90th_perc_model)
          
          print(f"[INFO:] Done with experiment No.{i}!\n")
