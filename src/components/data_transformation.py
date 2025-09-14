import os
import sys
import pandas as pd
import numpy as np 
from dataclasses import dataclass

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessed_data_path: str=os.path.join("artifacts", "preprocessing.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config =  DataTransformationConfig()
        
    def get_preprocessing_obj(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(steps=[
                ("Impute", SimpleImputer(strategy="median")),
                ("Scale", StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps=[
                ("Impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder())
            ])
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_features),
                    ("categorical_pipeline", cat_pipeline, categorical_features)
                ]

            )
            
            return preprocessor
        
        except Exception as e:
            raise Custom_Exception(e, sys)
        
        
    def initiate_data_transformation(self, train_data, test_data):
        
        try:
        
            preprocessing_obj = self.get_preprocessing_obj()
            logging.info("preprocessing obj imported")
            
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)
            
            target_feature = "math_score"
            
            train_features = train_df.drop(columns=[target_feature])
            train_target = train_df[target_feature]
            
            test_features = test_df.drop(columns=[target_feature])
            test_target = test_df[target_feature]
            
            preprocessed_train_features = preprocessing_obj.fit_transform(train_features)
            preprocessed_test_features = preprocessing_obj.transform(test_features)
            
            train_arr = np.c_[
                preprocessed_train_features, np.array(train_target)
            ]
            
            test_arr = np.c_[
                preprocessed_test_features, np.array(test_target)
            ]
            
            save_object(
                file_path = self.data_transformation_config.preprocessed_data_path,
                obj = preprocessing_obj
            )
            
            logging.info("preprocessing file path and obj saved")
            
            return (
                train_arr, test_arr
            )
        
        except Exception as e:
            raise Custom_Exception(e, sys)
        
        
            