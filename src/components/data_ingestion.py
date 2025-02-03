""" 
Read data from the cloud, database, internet etc source.
Includes some inputs that may be required for this Data Ingestion Component:
    1. where I have to the save the training path/data
    2. where I have to save the test data 
    3. where I have to save the raw data
    All these above can be saved in another class DataIngestionConfig
"""

import os
import sys # For custom exception

# Get the project root directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # From python 3.8 to create class variables

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# @dataclass decorator- Helps in creating all the class variables without having to create '__init__' constructor
@dataclass
class DataIngestionConfig:
    "Saving the files that are the output of the data ingestion in the artifact folder"
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\\data\\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys) from e 
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_array,test_array,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array,test_array))
    