from nltk import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import word_tokenize
import ast

stop = stopwords.words('french')

def make_features(df, task, Config):
    y = get_output(df, task)

    X = df["video_name"]

    if task == "is_name":
        X = df["tokens"]
        y = y.apply(lambda x: ast.literal_eval(x))
        labels_ds, features = [], []
        sentences_with_id = {}
        for sentence_id, (sentence, labels) in enumerate(zip(X, y)):  # Pair each word with its corresponding label
            for i, (word, word_label) in enumerate(zip(sentence, labels)):
                feature = {
                    'word': word,
                    'is_final_word': i == len(sentence) - 1,
                    'is_starting_word': i == 0,
                    'is_capitalized': word.istitle()}

                features.append(feature)
                labels_ds.append(word_label)
            sentences_with_id[sentence_id] = feature


        return features, labels_ds, sentences_with_id


    elif task == "is_comic_video":
        # Apply lowercase transformation
        X = X.apply(lambda x: " ".join(x.lower() for x in x.split()))
        if "stop words" in Config:
            X = X.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

        if "stemming" in Config:
            stemmer = SnowballStemmer("french")
            X = X.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

        return X, y

    else:
        raise ValueError("Unknown task")

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
