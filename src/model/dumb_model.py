import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import *
import numpy as np
class DumbModel:
    """Dumb model always predicts 0"""

    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y, test_size, random_state)
        self.clf = None  # Store the classifier
        self.accuracy = None  # Store cross-validation scores

    def split_data(self, X, y, test_size, random_state):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def fit(self):
        self.clf = RandomForestClassifier().fit(self.X_train, self.y_train)

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

    def calculate_accuracy(self):
        # Calculate and return the accuracy of the model
        if self.clf is None:
            raise ValueError("Classifier not trained. Call the 'fit' method first.")
        y_pred = self.predict(self.X_test)
        correct_predictions = (y_pred == self.y_test).sum()
        total_predictions = len(self.y_test)
        self.accuracy = correct_predictions / total_predictions

    # def cross_validate(self, cv=5):
    #     # Perform cross-validation and return an array of accuracy scores
    #     if self.clf is None:
    #         raise ValueError("Classifier not trained. Call the 'fit' method first.")
    #     cv_scores = cross_val_score(self.clf, self.X_train, self.y_train, cv=cv)
    #     return cv_scores


    def dump(self, filename_output):
        # Save the trained classifier to a file
        if self.clf is not None:
            dump(self.clf, filename_output)
        else:
            raise ValueError("Classifier not trained. Call the 'fit' method first.")
