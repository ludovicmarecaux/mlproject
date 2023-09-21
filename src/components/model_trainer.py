from src.logger import logging
from src.exception import CustomException

import os
import sys

from src.utils import save_object
from src.utils import evaluate_models
from dataclasses import dataclass

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

@dataclass
class ModelTrainerConfig:
    model_trainer_path=os.path.join("artefacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("Separations des jeu de données et test préprocessé")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],

            )

            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }



            logging.info(f"evaluation des models")
            model_report={}
            para={}
            model_report,para=evaluate_models(X_train,y_train,X_test,y_test,models)
            
            #best model score from dictionnary
            
            best_model_score=max(sorted(model_report.values()))
            
            #best model name

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model=models[best_model_name]
            
            #if best_model_score<0.6:
                #raise CustomException("Pas de modèles pertinents trouvés")
            best_para=para[best_model_name]
                    

            logging.info(f"Meilleur modèle trouvé sur le train et le test dataset")

            save_object(
                file_path=self.model_trainer_config.model_trainer_path,
                obj=best_model

            )
            best_model.set_params(**best_para)
            best_model.fit(X_train,y_train)
            predicted=best_model.predict(X_test)
            
            r2_square=r2_score(y_test,predicted)
            
            return r2_square



        except Exception as e:
            CustomException(e,sys)
