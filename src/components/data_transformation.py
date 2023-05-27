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
from src.utils import save_object

@dataclass
# Below will be giving us any path that we will be requiring any inputs for any data transformation components
class DataTransformationConfig:
    
    # Giving input to data transformation components
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
    
class DataTransformation:
        def __init__(self):
            self.data_transformation_config=DataTransformationConfig()
            
        def get_data_transformer_object(self):
            
            ''' 
            This function is responsible for data transformation
            '''
            
            try:
                numerical_columns = ["writing_score", "reading_score"]
                categorical_columns = ["gender",
                                       "race_ethnicity",
                                       "parental_level_of_education",
                                       "lunch",
                                       "test_preparation_course"]
                
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
               
                
                preprocessor= ColumnTransformer(
                    [
                        ("num_pipeline", num_pipeline, numerical_columns),
                        ("cat_pipeline", cat_pipeline, categorical_columns)
                    ]
                )
                
                logging.info(f"Transformed Categorical columns :{categorical_columns}")
                logging.info(f"Transformed Numerical columns :{numerical_columns}")
                
                return preprocessor
            
            
            except Exception as e:
                raise CustomException(e, sys)
            
        def initiate_data_transformation(self, train_path, test_path):
    
            try:
                train_df = pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)
        
                logging.info("Read train and test data completed")
                logging.info("Obtained preprocessing object")
        
                preprocessing_obj = self.get_data_transformer_object()
        
                target_column_name="math_score"
                numerical_columns= ["writing_score", "reading_score"]
                
                input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
                target_feature_train_df= train_df[target_column_name]
            
        
                input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 1)
                target_feature_test_df= test_df[target_column_name]
            
                logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )

                input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
            
                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
            
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
                logging.info(f"Saved preprocessing object.")
            
                save_object(
                    file_path= self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )
            
                return(
                    train_arr,
                    test_arr,
                    self_data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomException(e,sys)
    