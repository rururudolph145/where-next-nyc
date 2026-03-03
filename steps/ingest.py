import yaml
import pandas as pd


class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)

    def load_data(self):
        train_path = self.config['data']['train_path']
        test_path = self.config['data']['test_path']
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        return train, test
