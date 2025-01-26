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

from sklearn.exceptions import NotFittedError

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple models with hyperparameter tuning using GridSearchCV.
    """
    report = {}

    for model_name, model in models.items():
        try:
            logging.info(f"Evaluating model: {model_name}")

            # Get hyperparameters for the model (if any)
            param_grid = params.get(model_name, {})
            
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="r2",
                cv=5,
                n_jobs=-1,
                verbose=0,
            )

            # Fit the model on training data
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Evaluate the best model on test data
            y_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_pred)

            # Log the results
            report[model_name] = test_score
            logging.info(f"{model_name} R2 score: {test_score}")

        except Exception as e:
            # Handle fit failures
            logging.error(f"Error evaluating model {model_name}: {e}")
            report[model_name] = np.nan  # Assign NaN if evaluation fails

    return report
