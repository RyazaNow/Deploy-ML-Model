import src

RAW_DATA_PATH = 'data/raw/all_data.csv'
CLEANED_DATA_PATH = 'data/interim/data_cleaned.csv'

if __name__ == '__main__':
    src.clean_data([RAW_DATA_PATH, CLEANED_DATA_PATH])