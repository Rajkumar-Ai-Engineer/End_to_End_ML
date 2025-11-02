from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.exceptons import CustomException
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.utils import save_object,evaluate_models
from src.config import ModelTrainerConfig


from src.components import  data_ingestion,data_transformation


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "SVR": SVR(),
                "GaussianNB": GaussianNB(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGB Regressor": XGBRegressor()
            }
            
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test,predicted)
            return (r2_square,best_model_name,best_model_score)
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = data_ingestion.DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    obj1 = data_transformation.DataTransformation()
    train_array,test_array,preprocessor_path = obj1.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)
    obj2 = ModelTrainer()
    r2_square,model_name,model_score = obj2.initiate_model_trainer(train_array=test_array,test_array=test_array)
    print("R2 Square Score :",r2_square)
    print("Model Name :",model_name)
    print("Model Score :",model_score)
    
    

