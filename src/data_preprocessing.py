import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess_data(file_path, seed=42):
    data = pd.read_csv(file_path, sep=',', header=0)
    data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
                   'thalach','exang','oldpeak','slope','ca','thal','DISEASE']
    
    data.replace('?', np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    
    data.fillna(data.mean(), inplace=True)
    
    numerical_cols = data.columns.drop(['sex','fbs','exang','DISEASE'])
    data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()
    
    data['DISEASE'] = (data['DISEASE'] > 0).astype(float)
    
    features = data.drop('DISEASE', axis=1).values
    labels = data['DISEASE'].values
    
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)
    
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True)
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std
    
    batch_size = 20
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])
    
    return train_loader, test_loader

