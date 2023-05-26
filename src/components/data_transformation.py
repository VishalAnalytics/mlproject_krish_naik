import os
import sys
from dataclasses import dataclass

'''
Here we will be focusing on doing data transformation. 
It means that any dataset that we have. 
We will be using all the transformation techniques. 
If there is a categorical feature, of a numerical feature, or handling missing values.
Everything will be handled over here.
'''
import numpy as np
import pandas as pd

#ColumnTransformer is used to create pipeline, like onehot encoding, standard scaling
from sklearn.compose import ColumnTransformer

# For missing values use imputers
from sklearn.impute import SimpleImputer

#Implementing pipelines
from sklearn.pipeline import Pipeline

# Using Scaling techniques
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Exception Handling
from src.exception import CustomException
from src.logger import logging

@dataclass
# Below will be giving us any path that we will be requiring any inputs for any data transformation components
class DataTransformationConfig:
    
    # Giving input to data transformation components
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
    
class DataTransformation:
        def __init__(self):
            self.data_transformation_config=DataTransformationConfig()
            
        def get_data_transformer_object(self):
            try:
                numerical_columns = ["writing_score", "reading_score"]
                categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
                
                #Below we will be doing for Numerical Columns
                num_pipeline= Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]
                )
                
                logging.info("Numerical columns standard scaling completed")
                
                #Below we will be doing for Categorical Columns
                cat_pipeline= Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", OneHotEncoder()),
                        ("scaler", StandardScaler())
                                                
                    ]
                )
                logging.info("Categorical columns standard scaling completed")
                
                logging.info("Numerical columns standard scaling completed")
                
            except:
                pass
    