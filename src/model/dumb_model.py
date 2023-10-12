import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from joblib import *
import numpy as np
class DumbModel:
    """Dumb model always predicts 0"""

    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y, test_size, random_state)
        self.clf = None  # Store the classifier
        self.cv_scores = None  # Store cross-validation scores

    def split_data(self, X, y, test_size, random_state):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def fit(self):
        self.clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1).fit(self.X_train, self.y_train)

    def predict(self, X):
        # This function predicts using the trained classifier
        if self.clf is None:
            raise ValueError("Classifier not trained. Call the 'fit' method first.")
        # vectorizer = CountVectorizer()
        # X_new_counts = vectorizer.transform(X)
        # tfidf_transformer = TfidfTransformer()
        # X_new_tfidf = tfidf_transformer.transform(X_new_counts)



        # Predict
        predictions = self.clf.predict(X)

        return predictions

    def calculate_accuracy(self, cv=5):
        # Calculate accuracy using cross-validation
        if self.clf is None:
            raise ValueError("Classifier not trained. Call the 'fit' method first.")

        predictions = self.predict(self.X_test)
        predictions.reshape(-1, 1)
        predictions = predictions[:, np.newaxis]  # Convert predictions to a 2D array
        y_test_2d = self.y_test.to_numpy()[:, np.newaxis]

        self.cv_scores = cross_val_score(self.clf, predictions, y_test_2d, cv=cv)

    def dump(self, filename_output):
        # Save the trained classifier to a file
        if self.clf is not None:
            dump(self.clf, filename_output)
        else:
            raise ValueError("Classifier not trained. Call the 'fit' method first.")
