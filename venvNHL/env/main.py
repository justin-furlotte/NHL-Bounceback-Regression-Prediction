# Introduction

# The purpose of this project is to build a machine learning model which uses data from https://moneypuck.com and, intuitively, attempts to learn how many goals a player "deserved" to score in a given season. This is in order to perform outlier detection. An "underperforming" player is a player who scored significantly less goals than the model predicted they should score, and an "overperforming" player would be a player who scored more goals than the model predicted. 
# 
# It is important to note that this model is not predicting how many goals a player will score *next* season. Instead, the model is learning how many goals a player *should have scored this season*, and comparing this prediction with the number of goals the player *actually* scored. 
# 
# From this, we will see that most players who are deemed as underperformers by the model do indeed have bounce back seasons (purely in terms of number of goals scored) the following year, and similarly most overperformers tend to score less goals the following year.
# 
# This type of model might be useful tool when trying to evaluate, say, the value of a player in a trade (e.g. selling high and buying low), negotiating contracts, or creating draft lists.
import utils 
import data_cleaning
import model_tuning

import os
import csv
import sys
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    plot_confusion_matrix,
    recall_score,
    mean_absolute_error
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
import pickle

train_new_models = False

# Seasons used as training data (i.e. "previous seaons")
season_start = "10_11"
season_ends = ["11_12", "12_13", "13_14", "14_15", "15_16", "16_17", "17_18", "18_19", "19_20", "20_21", "21_22"]

# Seasons used for predicting over/underperformers (i.e. "this year" and "next year")
this_seasons = ["12_13", "13_14", "14_15", "15_16", "16_17", "17_18", "18_19", "19_20", "20_21", "21_22"]
next_seasons = ["13_14", "14_15", "15_16", "16_17", "17_18", "18_19", "19_20", "20_21", "21_22"]

# Load the data from each year into a dictionary
dfs = {}
for i in np.arange(10,22):
    year = str(i)+"_"+str(i+1)
    csv_name = "./env/PlayerData/Players_"+year+".csv"
    dfs[year] = pd.read_csv(csv_name, index_col="playerId")

for i in [1]:

    season_end = season_ends[i]
    this_season = this_seasons[i]
    next_season = next_seasons[i]

    # Preprocess the data

    cleaner = data_cleaning.Cleaner(dfs)
    cleaner.CreateAllSituationsDF()

    # X_train and y_train are all the data from the year
    # season_start to the year season_end
    X_train, y_train = cleaner.CreateXytrain(season_start, season_end)

    cleaner.CreatePreprocessor(X_train)
    cleaner.GetNewFeatureNames(X_train)
    preprocessor = cleaner.preprocessor
    new_feature_names = cleaner.transformed_feature_names

    dfs = cleaner.dfs

    playerid_this_season = {playerid: str(playerid)+"_"+this_season for playerid in list(dfs[this_season].index)}
    X_this_season = dfs[this_season].drop(columns="I_F_goals").rename(index=playerid_this_season)
    y_this_season = dfs[this_season]["I_F_goals"].rename(index=playerid_this_season)

    playerid_next_season = {playerid: str(playerid)+"_"+next_season for playerid in list(dfs[next_season].index)}
    X_next_season = dfs[next_season].drop(columns="I_F_goals").rename(index=playerid_next_season)
    y_next_season = dfs[next_season]["I_F_goals"].rename(index=playerid_next_season)


    if train_new_models == True:

        # Create a pipeline for some models

        # Lasso Regression
        pipe_lasso = make_pipeline(preprocessor, StandardScaler(), Lasso(max_iter=1000))

        # Random forest regression
        pipe_rfr = make_pipeline(preprocessor, RandomForestRegressor())

        # SVM Regression
        # Unfortunately my laptop is too slow to do hyperparameter optimization on svr but it is still giving
        # decent results with a regularization strength of 1 and Gaussian RBF kernel
        pipe_svr = make_pipeline(preprocessor, StandardScaler(), SVR(kernel="rbf", C=1.0))

        tuner = model_tuning.Tuner(cleaner, pipe_svr, pipe_lasso, pipe_rfr, season_start, season_end)
        tuner.TuneLasso(X_train, y_train)
        tuner.TuneRFR(X_train, y_train)
        tuner.TuneSVR(X_train, y_train)
        tuner.TuneEnsemble(X_train, y_train)
        

    # The final model is the stacking regressor, which is an ensemble
    # method combining lasso, support vector regressor, and 
    # random forest regressor using ridge as the final estimator
    #model = tuner.sr_ridge


    with open("./env/PickledModels/SR_pickle_"+season_start+"_to_"+season_end,"rb") as f:
       model = pickle.load(f)

    chart = utils.Chart()

    scatter_df = chart.CreateScatterDF(model, dfs)
    print(scatter_df.head(5))

    # Plot the results
    chart.Find(chart.OverPerformer, X_this_season, X_next_season, pd.DataFrame(y_this_season), pd.DataFrame(y_next_season), model, print_details=False, produce_plot=False)
    chart.Find(chart.UnderPerformer, X_this_season, X_next_season, pd.DataFrame(y_this_season), pd.DataFrame(y_next_season), model, print_details=False, produce_plot=False)

class Scatter:
    def __init__(self):
        self.scatter_df = scatter_df