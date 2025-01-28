import os
import sys
from src.Datascienceproject.exception import CustomException
from src.Datascienceproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


load_dotenv()
host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db   
        )
        logging.info("Connection Established",mydb)
        df=pd.read_sql_query("Select * from train",mydb)
        print(df.head())
        
        return df 
        
    except Exception as ex:
        raise CustomException(ex)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

from sklearn.model_selection import GridSearchCV

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            try:
                # Get parameter grid (fallback to empty dictionary if none exists)
                param_grid = params.get(model_name, {})
                if param_grid:
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                else:
                    # Directly fit if no parameter grid
                    model.fit(X_train, y_train)
                    best_model = model

                y_pred = best_model.predict(X_test)
                r2_square = r2_score(y_test, y_pred)
                report[model_name] = r2_square
                logging.info(f"{model_name} - R2 Score: {r2_square}")

            except Exception as model_error:
                logging.error(f"Error training {model_name}: {model_error}", exc_info=True)

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)