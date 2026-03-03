import yaml
import joblib
import os
from lightgbm import LGBMRegressor


MODELS = {
    'LGBMRegressor': LGBMRegressor,
}

TARGET_COL = 'trip_count'


class Trainer:
    def __init__(self):
        self.config = self.load_config()
        model_cfg = self.config['model']
        self.model_name = model_cfg['name']
        self.store_path = model_cfg['store_path']
        self.model = MODELS[self.model_name](**model_cfg.get('params', {}))

    def load_config(self):
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)

    def feature_target_separator(self, data):
        X = data.drop(columns=[TARGET_COL])
        y = data[TARGET_COL]
        return X, y

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def save_model(self):
        os.makedirs(self.store_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(self.store_path, 'model.pkl'))
