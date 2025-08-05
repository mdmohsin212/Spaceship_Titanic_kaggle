import pandas as pd
import os

train_data = pd.read_csv('../data/raw/raw_train.csv')
test_data = pd.read_csv('../data/raw/raw_test.csv')

def data_preprocess(df : pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        return df
    except KeyError as e:
        print(f'Error : Missing Column {e} in Dataframe')
        raise
        

train_processed_data = data_preprocess(train_data)
test_processed_data = data_preprocess(test_data)

data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)