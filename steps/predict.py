import yaml
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

TARGET_COL = 'trip_count'


class Predictor:
    def __init__(self):
        self.config = self.load_config()
        self.model = self.load_model()

    def load_config(self):
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)

    def load_model(self):
        store_path = self.config['model']['store_path']
        return joblib.load(os.path.join(store_path, 'model.pkl'))

    def feature_target_separator(self, data):
        X = data.drop(columns=[TARGET_COL])
        y = data[TARGET_COL]
        return X, y

    def evaluate_model(self, X_test, y_test):
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return mae, rmse, r2
