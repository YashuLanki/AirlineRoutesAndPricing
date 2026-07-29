import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline(data_path=None):
    """Ingest -> transform -> train, end to end. Returns the test-set R^2."""
    try:
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(data_path)

        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )

        trainer = ModelTrainer()
        r2_square = trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)

        logging.info(f"Training pipeline complete. Test R^2: {r2_square:.4f}")
        return r2_square
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    print(f"Test R^2: {run_training_pipeline():.4f}")
