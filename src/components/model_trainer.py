import os 
import sys 
import pandas as pd

from src.exception import Custom_Exception
from src.logger import logging 
from src.utils import evaluate_models, save_object
from dataclasses import dataclass

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)


@dataclass
class ModelTrainerConfig:
    model_trainer_path: str=os.path.join("artifacts", "model_trainer.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        
        try:
            
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )         
            
            models = {
                "Linear_Regression" : LinearRegression(),
                "Decision_Tree" : DecisionTreeRegressor(),
                "SVM" : SVR(),
                "KNN" : KNeighborsRegressor(),
                "Random_Forest" : RandomForestRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "GradientBoosting" : GradientBoostingRegressor()
            }
            
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,
                                                y_test=y_test, models=models)
            
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            if best_model_score < 0.6:  # Example threshold
                raise Custom_Exception("No model performed well enough (R2 < 0.6)")
            
            best_model = models[best_model_name]
            
            predicted = best_model.predict(X_test)
            
            r2score = r2_score(y_test, predicted)
            
            save_object(
                file_path=self.model_trainer_config.model_trainer_path,
                obj = best_model
            )
            
            logging.info(f"Model saved at {self.model_trainer_config.model_trainer_path}")
            
            return r2score
        
        except Exception as e:
            raise Custom_Exception(e, sys)
        
        
        
        
        
        
        