import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import *
import numpy as np
class DumbModel:
    """Dumb model always predicts 0"""

    def __init__(self, X_train, y_train, X_test, y_test, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = 0

    def fit(self):
        self.clf = RandomForestClassifier().fit(self.X_train, self.y_train)

    def predict(self, X):
        # This function predicts using the trained classifier
        if self.clf is None:
            raise ValueError("Classifier not trained. Call the 'fit' method first.")

        # Predict
        predictions = self.clf.predict(X)

        return predictions

    def dump(self, filename_output):
        # Save the trained classifier to a file
        if self.clf is not None:
            dump(self.clf, filename_output)
        else:
            raise ValueError("Classifier not trained. Call the 'fit' method first.")
