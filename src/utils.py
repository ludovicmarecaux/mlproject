import os
import sys
import dill
from src.exception import CustomException
from logger import logging

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from optuna.samplers import TPESampler

from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        CustomException(e,sys)




def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        report={}
        para={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            name=list(models.keys())[i]
            logging.info(f"entraînement du modèle: {name}")
            if name=="Random Forest":
                
                def objective(trial):
                    algo = RandomForestRegressor(
                                        criterion=trial.suggest_categorical('criterion',['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                                        max_features=trial.suggest_categorical('max_features',['sqrt','log2',None]),
                                        n_estimators=trial.suggest_int('n_estimators',8,256)
                                        )
                    algo.fit(X_train, y_train)
                    y_pred = algo.predict(X_test)
                    return r2_score(y_test, y_pred)
                
            elif name=="Decision Tree" :
                def objective(trial):
                    algo = DecisionTreeRegressor(
                        criterion=trial.suggest_categorical('criterion',['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
                        splitter=trial.suggest_categorical('splitter',['best','random']),
                        max_features=trial.suggest_categorical('max_features',['sqrt','log2']),
                        )
                    algo.fit(X_train, y_train)
                    y_pred = algo.predict(X_test)
                    return r2_score(y_test, y_pred)
                
            elif name=="Gradient Boosting":
                    def objective(trial):
                        algo = GradientBoostingRegressor(
                            loss=trial.suggest_categorical('loss',['squared_error', 'huber', 'absolute_error', 'quantile']),
                            learning_rate=trial.suggest_float('learning_rate',1e-3, 1e-1, log=True),
                            subsample=trial.suggest_float('subsample',0.6,0.9,log=True),
                            criterion=trial.suggest_categorical('criterion',['squared_error', 'friedman_mse']),
                            max_features=trial.suggest_categorical('max_features',['auto','sqrt','log2']),
                            n_estimators=trial.suggest_int('n_estimators',8,256)
                            )
                        algo.fit(X_train, y_train)
                        y_pred = algo.predict(X_test)
                        return r2_score(y_test, y_pred)
                
            elif name=="Linear Regression":
                    def objective(trial):
                        algo = LinearRegression()
                        algo.fit(X_train, y_train)
                        y_pred = algo.predict(X_test)
                        return r2_score(y_test, y_pred)
            
            elif name=="XGBRegressor":
                    def objective(trial):
                        algo = XGBRegressor(
                            learning_rate=trial.suggest_float('learning_rate',1e-3, 1e-1, log=True),
                            n_estimators=trial.suggest_int('n_estimators',8,256)
                            )
                        algo.fit(X_train, y_train)
                        y_pred = algo.predict(X_test)
                        return r2_score(y_test, y_pred)
            
            elif name=="CatBoosting Regressor":
                    def objective(trial):
                        algo = CatBoostRegressor(
                            iterations=trial.suggest_int("iterations", 100, 1000),
                            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                            depth=trial.suggest_int("depth", 4, 10),
                            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
                            bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
                            random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
                            od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
                            od_wait=trial.suggest_int("od_wait", 10, 50),
                            verbose=False
                            )
                        algo.fit(X_train, y_train)
                        y_pred = algo.predict(X_test)
                        return r2_score(y_test, y_pred)
                    
            else:
                   def objective(trial):
                        algo = AdaBoostRegressor(
                            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                            loss=trial.suggest_categorical('loss',['linear','square','exponential']),
                            n_estimators=trial.suggest_int('n_estimators',8,256)

                            )
                        algo.fit(X_train, y_train)
                        y_pred = algo.predict(X_test)
                        return r2_score(y_test, y_pred)

            sampler = TPESampler(seed=1)
            study = optuna.create_study(study_name=name, direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=100)

            
            trial = study.best_params
            
            logging.info(f"Recherche des hyper-paramètres ")
                        
            best_model=model.set_params(**trial)     
                
            best_model.fit(X_train, y_train)
            logging.info(f"prediction avec les meilleurs paramètres avec le modèle {name}")
            y_train_pred = best_model.predict(X_train)

            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
            
            para[list(models.keys())[i]]=trial
            
            
        return report,para    
            

    except Exception as e:
        CustomException(e,sys)



def load_object(file_path):
    try:
        with open(file_path,"rb") as object:
            return dill.load(object)

    except Exception as e:
        CustomException(e,sys)

