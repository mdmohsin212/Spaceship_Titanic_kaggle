import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

train_data = pd.read_csv('../data/processed/train_processed.csv')
test_data = pd.read_csv('../data/processed/test_processed.csv')

for col in ['Sex', 'Embarked']:
    enc = LabelEncoder()
    train_data[col] = enc.fit_transform(train_data[col])
    test_data[col] = enc.transform(test_data[col])
    
data_path = os.path.join("data", "features")

os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)