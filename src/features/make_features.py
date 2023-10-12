from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB
from nltk import *
from nltk.corpus import stopwords

stop = stopwords.words('french')


def make_features(df, task, stop_words=False, stemming=False):
    X = df["video_name"]
    y = df["is_comic"]

    if task == "is_comic_video":
        X = X.apply(lambda x: " ".join(x.lower() for x in x.split())) #chagner le code
    if stop_words:
        X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

    if stemming:
        stemmer = SnowballStemmer("french")
        X = X.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    # tfidf_transformer = TfidfTransformer()
    # X = tfidf_transformer.fit_transform(X)

    return X, y
