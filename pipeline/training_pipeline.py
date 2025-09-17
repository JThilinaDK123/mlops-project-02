from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessing
from src.model_training import ModelTraining
from src.feature_store import RedisFeatureStore
from config.paths_config import *
from config.database_config import DB_CONFIG


if __name__=="__main__":

    ## Data ingestion pipeline
    data_ingestion = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion.run()

    feature_store = RedisFeatureStore()

    ## Data processing pipeline
    data_processor = DataProcessing(TRAIN_PATH, TEST_PATH, feature_store)
    data_processor.run()

    ## Model Training pipeline
    feature_store = RedisFeatureStore()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()