import unittest
import torch
from src.data_preprocessing import load_and_preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_load_and_preprocess_data(self):
        train_loader, test_loader = load_and_preprocess_data('data/raw/heart_disease_uci.csv')
        # Check if loaders are not empty
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)
        # Check batch size
        for X, y in train_loader:
            self.assertEqual(X.shape[1], 13)  # 13 features
            break

if __name__ == '__main__':
    unittest.main()

