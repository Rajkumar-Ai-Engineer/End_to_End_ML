from  src.logger import logging
from src.exceptons import CustomException
from src.utils import save_object
import sys 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.config import DataTransformationConfig


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self,numerical_columns,categorical_columns):
        try:
            logging.info("Data Transformation initiated")
            
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False,handle_unknown="ignore")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Combine Pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            target = "math_score"
            input_feature_train_df = train_df.drop(columns=[target],axis=1)
            target_feature_train_df = train_df[target]
            
            input_feature_test_df = test_df.drop(columns=[target],axis=1)
            target_feature_test_df = test_df[target]
            
            numerical_columns = input_feature_train_df.select_dtypes(include=['int64','float64']).columns
            categorical_columns = input_feature_train_df.select_dtypes(include=['object']).columns
            
            preprocessor = self.get_data_transformer_object(numerical_columns,categorical_columns)
            logging.info("Applying preprocessing object on training and testing datasets")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Error in initiate_data_transformation")
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation(train_path="artifacts/train.csv",test_path="artifacts/test.csv")