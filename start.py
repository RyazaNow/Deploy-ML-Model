import src

RAW_DATA_PATH = 'data/raw/all_data.csv'
CLEANED_DATA_PATH = 'data/interim/data_cleaned.csv'
FEATURES_DATA_PATH = 'data/interim/data_features.csv'


if __name__ == '__main__':
    src.clean_data([RAW_DATA_PATH, CLEANED_DATA_PATH])
    src.add_features([CLEANED_DATA_PATH, FEATURES_DATA_PATH])


#data/interim/data_features.csv

#python src/models/prepare_datasets.py data/interim/data_features.csv data/processed/train.csv data/processed/test.csv