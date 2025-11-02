from  src.logger import logging
from src.exceptons import CustomException
import sys 
from src.config import DataIngestionConfig
import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion Engine")
        
        try:
            data = pd.read_csv("artifacts\data.csv")
            logging.info("Read the dataset as dataframe completed")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data is saved")
            
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            logging.info("Train test split is done")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train data is done")
            logging.info("Test data is done")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.info("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

            

            
            
            