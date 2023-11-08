from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier

def make_model(task):
    if task == "is_comic_video":
        log = "Choose the configs you want to use: \n"
        log += "1. Stemming \n"
        log += "2. Stop words \n"
        log += "3. TF-IDF \n"
        log += "4. Default \n"
        log += "5. Exit \n"

        print(log)

        Config = []
        while True:
            choice = input("Enter your choice: ")
            if choice == "1":
                Config.append("stemming")
            elif choice == "2":
                Config.append("stop words")
            elif choice == "3":
                Config.append("TF-IDF")
            elif choice == "4":
                if Config == []:
                    Config.append("count_vectorizer")
                break
            elif choice == "5":
                break
            else:
                print("Invalid choice")

        if "TF-IDF" in Config:
            return Pipeline([
                ("tfidf_vectorizer", TfidfVectorizer()),
                ("random_forest", AdaBoostClassifier(random_state=42)),
            ]), Config
        else:
            return Pipeline([
                ("count_vectorizer", CountVectorizer()),
                ("random_forest", AdaBoostClassifier(random_state=42)),
            ]), Config

    elif task == "is_name":
        return Pipeline([
            ("Dict_Vectorizer", DictVectorizer()),
            ("random_forest", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42)),
        ]), []
    elif task == "find_comic_name":
        return Pipeline([
            ("count_vectorizer", CountVectorizer()),
            ("random_forest", AdaBoostClassifier(random_state=42)),
        ]), []


def dump(model, Config, filename_output):

    save_data = {
        "model": model,
        "CONFIG": Config
    }

    # Save the data to a file
    with open(filename_output, 'wb') as file:
        pickle.dump(save_data, file)