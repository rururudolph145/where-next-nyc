import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def main():
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data cleaning completed successfully")

    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    mae, rmse, r2 = predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")

    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"MAE: {mae:.4f}  |  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")
    print("=====================================================\n")


def train_with_mlflow():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment("NY Taxi Demand Forecasting")

    with mlflow.start_run() as run:
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed successfully")

        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        mae, rmse, r2 = predictor.evaluate_model(X_test, y_test)
        logging.info("Model evaluation completed successfully")

        mlflow.set_tag('Model developer', 'jason')
        mlflow.set_tag('task', 'demand_forecasting')

        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(trainer.model, "model")

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "ny_taxi_demand_model")
        logging.info("MLflow tracking completed successfully")

        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"MAE: {mae:.4f}  |  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")
        print("=====================================================\n")


if __name__ == "__main__":
    # main()
    train_with_mlflow()
