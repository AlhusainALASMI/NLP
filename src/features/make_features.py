from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB
from nltk import *
from nltk.corpus import stopwords

stop = stopwords.words('french')


def make_features(df, task, Config):
    y = get_output(df, task)

    X = df["video_name"]


    if task == "is_comic_video":
        X = X.apply(lambda x: " ".join(x.lower() for x in x.split())) #chagner le code
    if "stop words" in Config:
        X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

    if "stemming" in Config:
        stemmer = SnowballStemmer("french")
        X = X.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


    return X, y
def get_output(df, task):
        if task == "is_comic_video":
            y = df["is_comic"]
        elif task == "is_name":
            y = df["is_name"]
        elif task == "find_comic_name":
            y = df["comic_name"]
        else:
            raise ValueError("Unknown task")

        return y
