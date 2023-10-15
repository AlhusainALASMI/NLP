from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def make_model():
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

    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", AdaBoostClassifier(random_state=42)),
    ]), Config


def dump(model, Config, filename_output):

    save_data = {
        "model": model,
        "CONFIG": Config
    }

    # Save the data to a file
    with open(filename_output, 'wb') as file:
        pickle.dump(save_data, file)