import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Cleaner:

    def __init__(self, dfs):
        self.dfs = dfs
        self.preprocessor = None
        self.num_features = None
        self.categorical_features = None
        self.drop_features = None
        self.pass_features = None
        self.transformed_feature_names = None

    # Create a dataframe for all-situations scoring (as opposed to just 5 on 5, or 5 on 4, etc)
    def CreateAllSituationsDF(self):
        for key in self.dfs.keys():
            self.dfs[key] = self.dfs[key].loc[self.dfs[key]["situation"] == "all"].drop(columns="situation")

    # Split the Data into train and test sets
    def CreateXytrain(self, season_start, season_end):

        X_all_years = [self.dfs[key].drop(columns="I_F_goals") for key in self.dfs.keys()]
        y_all_years = [self.dfs[key]["I_F_goals"] for key in self.dfs.keys()]
        
        start = int(season_start[:2])
        finish = int(season_end[:2])
        
        X = []
        y = []
        
        for i in np.arange(start,finish+1):
            
            dfX = X_all_years[i-start]
            
            playerid_new = {playerid: str(playerid)+"_"+str(i)+"_"+str(i+1) for playerid in list(dfX.index)}
            dfX = dfX.rename(index=playerid_new)
            X.append(dfX)
            
            dfy = y_all_years[i-start]
            dfy = dfy.rename(index=playerid_new)
            y.append(dfy)
            
            
        return pd.concat(X), pd.concat(y)

    # Create a preprocessor
    def CreatePreprocessor(self, X_train):
        # Some Careful Feature Selection
        self.num_features = ['games_played', 'icetime', 
                        'gameScore', 
                        'onIce_corsiPercentage', 'onIce_fenwickPercentage', 
                        'I_F_primaryAssists', 'I_F_secondaryAssists', 
                        'I_F_shotsOnGoal', 'I_F_missedShots', 'I_F_blockedShotAttempts', 'I_F_shotAttempts', 
                        'I_F_rebounds', 'I_F_freeze', 'I_F_playStopped', 
                        'I_F_playContinuedInZone', 'I_F_playContinuedOutsideZone',  
                        'I_F_hits', 'I_F_takeaways', 'I_F_giveaways', 'I_F_lowDangerShots', 'I_F_mediumDangerShots', 
                        'I_F_highDangerShots', 'I_F_scoreAdjustedShotsAttempts', 'I_F_unblockedShotAttempts', 
                        'I_F_shifts', 
                        'I_F_oZoneShiftStarts', 'I_F_dZoneShiftStarts', 'I_F_neutralZoneShiftStarts', 
                        'I_F_flyShiftStarts', 'I_F_oZoneShiftEnds', 'I_F_dZoneShiftEnds', 'I_F_neutralZoneShiftEnds', 
                        'I_F_flyShiftEnds', 'faceoffsWon', 'faceoffsLost', 
                        'penalityMinutesDrawn', 'penaltiesDrawn', 'OnIce_F_shotsOnGoal', 'OnIce_F_missedShots', 
                        'OnIce_F_blockedShotAttempts', 'OnIce_F_shotAttempts', 'OnIce_F_rebounds', 
                        'OnIce_F_lowDangerShots', 'OnIce_F_mediumDangerShots', 
                        'OnIce_F_highDangerShots', 
                        'OnIce_F_unblockedShotAttempts', 'corsiForAfterShifts', 
                        'corsiAgainstAfterShifts', 'fenwickForAfterShifts', 'fenwickAgainstAfterShifts']

        self.categorical_features = ["position"]

        # drop everything else
        self.drop_features = []
        for feat in X_train.columns:
            if feat not in self.num_features+self.categorical_features:
                self.drop_features.append(feat)

        self.pass_features = X_train.columns.tolist()
        for feat in self.drop_features + self.categorical_features: #everything else is passed through
            self.pass_features.remove(feat)

        # Keep track of new feature names
        new_categorical_features = []
        for feat_name in X_train["position"].unique().tolist():
            new_categorical_features.append(feat_name)

                
        preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), self.categorical_features),
                                            (StandardScaler(), self.num_features),
                                            ("drop", self.drop_features))

        # Note that we will apply standard scaler later in a pipeline

        self.preprocessor = preprocessor

    def GetNewFeatureNames(self, X_train):
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        ohe_column_names = self.preprocessor.named_transformers_["onehotencoder"].get_feature_names_out().tolist()#[0:51]
        new_feature_names = ohe_column_names + self.num_features

        self.transformed_feature_names = new_feature_names
