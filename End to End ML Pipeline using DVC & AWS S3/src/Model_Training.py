import os 
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import logging
## ENSURE THE logs directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

## LOGGLING CONFIGUARTION
logger=logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_building.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) ->pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df=pd.read_csv(file_path)
        logger.info(f"Data loaded from %s with Shape %s ",file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed parse  the csv file  %s",e)
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found %s",e)
        raise
    except Exception as e:
        logger.error(f"Unexpected  error occurred while loading the data %s",e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model,file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        ## Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open (file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model Saved To %s',file_path)
    except FileNotFoundError as e:
        logger.error(f"File not found %s",e)
        raise
    except Exception as e:
        logger.error(f"Error occurred while saving the model %s",e)
        raise

def main():
    try:
        params={'n_estimators':25,'random_state':2}
        train_data=load_data('./data/processed/train_tfidf.csv')
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values

        clf=train_model(x_train,y_train,params)
        save_model(clf,'./models/random_forest_model.pkl')
    except Exception as e:
        logger.error(f"Failed to compelte to model building process %s",e)
        print(f"Error: {e}")


if __name__=="__main__":
    main()