import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    # These are the input which we are given to Data Ingestion Configs, where they need to save data
    train_data_path: str = os.path.join('articfacts', "train.csv")
    test_data_path: str = os.path.join('articfacts', "test.csv")
    raw_data_path: str = os.path.join('articfacts', "data.csv")

class DataIngestion:
    def __init__(self):
        # As soon as we call above DataIngestion Class, above mentioned three variable train_data_path, test_data_path, raw_data_path, will get save into self.ingestion_config. 
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            
            # Getting data into dataframe
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            
            # We already know the path of training data, test data and raw data. My path would be artifact/train.csv. So, we will be creating folders.
            # Below we will be creating folders like artifacts/train.csv, artifacts/test.csv, artifacts/data.csv
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True )
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Now we will be spliting our data into train and test.
            logging.info('Train Test Split initiated')
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42 )
            
            train_set.to_csv(self.ingestion_config.train_data_path,index= False, header=True )
            
            test_set.to_csv(self.ingestion_config.test_data_path,index= False, header=True )
            
            logging.info("Ingestion of data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data= obj.initiate_data_ingestion()
            
    data_transformation= DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
        
        