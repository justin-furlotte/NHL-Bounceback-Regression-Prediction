import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

class Tuner:

    def __init__(self, cleaner, pipe_svr, pipe_lasso, pipe_rfr, season_start, season_end):
        self.pipe_svr = pipe_svr
        self.pipe_lasso = pipe_lasso 
        self.pipe_rfr = pipe_rfr
        self.cleaner = cleaner
        self.season_start = season_start 
        self.season_end = season_end

    def TuneSVR(self, X_train, y_train, verbose = True, save = True):
        # For now, "Tune" SVR actually just trains SVR as my laptop is quite slow
        # Support Vector Regressor
        if verbose == True:
            print("Training - SVR")
        
        if verbose == True:
            print("Cross validation results - untuned SVR:\n")
            print(pd.DataFrame(cross_validate(self.pipe_svr, X_train, np.ravel(y_train), scoring="neg_mean_absolute_error")))
        
        self.pipe_svr.fit(X_train, y_train)

        # save model
        if save == True:
            print("Saving SVR model...\n")
            with open("SVR_pickle_"+self.season_start+"_to_"+self.season_end, "wb") as f:
                pickle.dump(self.pipe_svr, f)
            print("SVR model saved")



    def TuneLasso(self, X_train, y_train, verbose = True, save = True):

        # Hyperparameter Tuning - LASSO

        if verbose == True:
            print("\nHyperparameter Tuning - LASSO")

        parameters_lasso = {"lasso__alpha": np.linspace(0.1, 5, 100)} # Some values for the regularization strength
        gs_lasso = GridSearchCV(
                                    self.pipe_lasso, 
                                    parameters_lasso,
                                    scoring="neg_mean_absolute_error"
        )

        # Older laptop that doesn't take well to parallel processing so n_jobs is not -1 here for grid search! 
        gs_lasso.fit(X_train, y_train)
        alpha_best_lasso = gs_lasso.best_params_["lasso__alpha"]
        alpha_best_lasso

        if verbose == True:
            print("\nGrid Search Results:\n")
            print(pd.DataFrame(gs_lasso.cv_results_).head(5))

        pipe_Lasso = make_pipeline(self.cleaner.preprocessor, StandardScaler(), Lasso(alpha = alpha_best_lasso))
        
        if verbose == True:
            print("\nCross validation results - tuned LASSO:")
            print(pd.DataFrame(cross_validate(pipe_Lasso, X_train, y_train, scoring="neg_mean_absolute_error")))

        pipe_Lasso.fit(X_train, y_train)
        self.pipe_lasso = pipe_Lasso

        if verbose == True:
            sorted_coefs = np.sort(np.array(pipe_Lasso.named_steps["lasso"].coef_))
            locs = np.argsort(np.array(pipe_Lasso.named_steps["lasso"].coef_))
            print("LASSO Regression coefficients (sorted)")
            for i, coef in enumerate(sorted_coefs):
                print(i)
                print("Feature:", self.cleaner.transformed_feature_names[locs[i]])
                print("Coefficient:", coef)
                print("\n")

        if save == True:
            # save model
            print("Saving LASSO model...\n")
            with open("LASSO_pickle_"+self.season_start+"_to_"+self.season_end, "wb") as f:
                pickle.dump(self.pipe_lasso, f)
            print("Lasso model saved")



    
    def TuneRFR(self, X_train, y_train, verbose = True, save = True):

        # Hyperparameter Tuning - Random Forest

        if verbose == True:
            print("Hyperparameter Tuning - Random Forest")

        d = self.cleaner.preprocessor.fit_transform(X_train).shape[1]
        parameters_rfr = {
                            "randomforestregressor__max_depth": np.arange(np.floor(np.sqrt(d)/2), np.floor(np.sqrt(d)*2)),
                            "randomforestregressor__n_estimators": np.arange(20,100)
                        }
        rs_rfr = RandomizedSearchCV(
                                    self.pipe_rfr, 
                                    parameters_rfr,
                                    scoring = "neg_mean_absolute_error",
                                    n_jobs=-1
        )
        rs_rfr.fit(X_train, np.ravel(y_train))

        max_depth_best = rs_rfr.best_params_["randomforestregressor__max_depth"]
        n_estimators_best = rs_rfr.best_params_["randomforestregressor__n_estimators"]

        if verbose == True:
            print("Best maximum depth:", max_depth_best)
            print("\nBest number of estimators:", n_estimators_best)

        pipe_rfr = make_pipeline(self.cleaner.preprocessor, 
                                        StandardScaler(), 
                                        RandomForestRegressor(max_depth = max_depth_best, n_estimators = n_estimators_best)
                                        )

        pipe_rfr.fit(X_train, y_train)
        self.pipe_rfr = pipe_rfr
        
        if verbose == True:
            print("RFR cross validation results - tuned RFR:\n")
            print(pd.DataFrame(cross_validate(pipe_rfr, 
                                            X_train, 
                                            np.ravel(y_train), 
                                            scoring="neg_mean_absolute_error")))
        
        if save == True:
            # save model
            print("Saving RFR model...\n")
            with open("RFR_pickle_"+self.season_start+"_to_"+self.season_end, "wb") as f:
                pickle.dump(self.pipe_rfr, f)
            print("Random forest model saved")




    def TuneEnsemble(self, X_train, y_train, verbose = True, save = True):

        # Ensemble Model (Stacking)
        self.sr_ridge = StackingRegressor(estimators = [("lasso", self.pipe_lasso), ("rfr", self.pipe_rfr), ("svr", self.pipe_svr)],
                                    final_estimator = Ridge(alpha=1))

        if verbose == True:
            print("Mean of cross validation results - stacking model:")
            print(np.mean(pd.DataFrame(cross_validate(self.sr_ridge, X_train, np.ravel(y_train), scoring = "neg_mean_absolute_error"))))
        
        self.sr_ridge.fit(X_train, np.ravel(y_train))
        
        if verbose == True:
            print("The stacking model coefficients are:\n")
            print(self.sr_ridge.final_estimator_.coef_)

        if save == True:
            # save model
            print("Saving ensemble model...\n")
            with open("SR_pickle_"+self.season_start+"_to_"+self.season_end, "wb") as f:
                pickle.dump(self.sr_ridge, f)
            print("ensemble model saved")