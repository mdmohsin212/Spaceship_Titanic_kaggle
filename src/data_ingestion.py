import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(data_url : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.ParserError as e:
        print(f'Error : Failed to parse the csv file from {data_url}')
        print(e)
        raise

def save_data(train_data : pd.DataFrame, test_data : pd.DataFrame, data_path : str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "raw_train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, 'raw_test.csv'), index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise
    

def main():
    try:
        df = load_data('https://raw.githubusercontent.com/mdmohsin212/Machine-Learning/refs/heads/main/dataset/Titanic.csv')
        train_data, test_data = train_test_split(df, test_size=.25, random_state=42)
        save_data(train_data, test_data, data_path='data')
    
    except Exception as e:
        print(f'Error : {e}')
        print("Failed to complete the data ingestion process")
        
if __name__ == '__main__':
    main()