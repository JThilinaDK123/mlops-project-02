import os

########### Path for DATA INGESTION ###########

## 01) Where we have to save the data

RAW_DIR = "artifacts/raw"
TRAIN_PATH = os.path.join(RAW_DIR,'titanic_train.csv')
TEST_PATH = os.path.join(RAW_DIR,'titanic_test.csv')
CLEANED_DATA_PATH = os.path.join(RAW_DIR,'final_train_data.csv')


########### Data Processing ###########
PROCESSED_DIR = "artifacts/processed"

