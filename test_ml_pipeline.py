import unittest
import os
import pandas as pd

from train_model import df, X, y, X_train, X_test, y_train, y_test
from train_model import model, scaler, y_pred, accuracy


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("train.csv"))

    def test_dataset_shape(self):
        self.assertEqual(df.shape, (2000, 21))

    def test_target_column_exists(self):
        self.assertIn("price_range", df.columns)

    def test_target_classes(self):
        self.assertEqual(sorted(y.unique().tolist()), [0, 1, 2, 3])

    def test_train_test_split(self):
        self.assertEqual(len(X_train), 1600)
        self.assertEqual(len(X_test), 400)

    def test_prediction_length(self):
        self.assertEqual(len(y_pred), 400)

    def test_accuracy(self):
        self.assertGreaterEqual(accuracy, 0.80)


if __name__ == "__main__":
    unittest.main()
