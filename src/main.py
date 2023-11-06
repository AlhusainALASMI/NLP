import click
from sklearn.model_selection import cross_val_score
import pickle
from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model.main import make_model
from src.model.main import dump
import pandas as pd
import os
import ast
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import ast


@click.group()
def cli():
    pass

def obtenir_prochain_numero_modele(fichier):
    try:
        with open(fichier, 'r') as f:
            lignes = f.readlines()
            if not lignes:
                return 1
            derniere_ligne_modele = None
            for ligne in reversed(lignes):
                if ligne.startswith("Model"):
                    try:
                        dernier_modele = int(ligne.split()[1])
                        return dernier_modele + 1
                    except ValueError:
                        continue
            return 1  # Si aucune ligne "Model X" valide n'est trouvée
    except FileNotFoundError:
        return 1
def sauvegarder_scores(task, fichier,model_number, model, configuration, scores, moyenne_scores):
    if task == "is_comic_video":
        with open(fichier, 'a') as f:
            f.write(f"Model {model_number}\n")
            f.write(f"Model {model}\n")
            f.write(f"Modele utilise: {model}\n")
            f.write(f"Configuration utilisee: {', '.join(configuration)}\n")
            f.write(f"Scores: {', '.join(map(str, scores))}\n")
            f.write(f"Moyenne des scores: {moyenne_scores}\n\n")
    elif task == "is_name":
        with open(fichier, 'a') as f:
            f.write(f"Model {model_number}\n")
            f.write(f"Model {model}\n")
            f.write(f"Modele utilise: {model}\n")
            f.write(f"Scores: {', '.join(map(str, scores))}\n")
            f.write(f"Moyenne des scores: {moyenne_scores}\n\n")
@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    model, CONFIG = make_model(task)

    df = make_dataset(input_filename)

    X, y = make_features(df, task, CONFIG)

    model.fit(X, y)
    ("i love sam")
    [0, 0, 1]

    return dump(model, CONFIG, model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/models/dump.json", help="File to dump model")
@click.option("--output_filename", default="src/data/processed/prediction.csv", help="Output file for predictions")
def predict(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)

    CONFIG = None
    Model = None

    load = pickle.load(open(model_dump_filename, 'rb'))
    if task == "is_name":
        Model = load["model"]

    elif task == "is_comic_video":
        CONFIG = load["CONFIG"]
        Model = load["model"]

    X, y, sentences_with_id = make_features(df, task, CONFIG)
    predictions_lists = []
    cpt = 0

    for row in df["video_name"]:
        liste = []
        for x in range(len(row.split())):
            if cpt <= (len(X) - 1):
                liste.append(Model.predict(X[cpt])[0])
                cpt += 1
        predictions_lists.append((liste, row.split()))

    predictions_df = pd.DataFrame({'Prediction': predictions_lists})

    # Save the DataFrame to a CSV file
    return predictions_df.to_csv(output_filename, index=False)




@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    df = make_dataset(input_filename)

    CONFIG = None
    Model = None

    if task == "is_name":
        Model = make_model(task)
    elif task == "is_comic_video":
        Model, CONFIG = make_model(task)
    elif task == "find_comic_name":
        df_test = pd.read_csv("src/data/raw/train.csv")
        Model, CONFIG = make_model("is_comic_video")
        X_is_comic, y_is_comic = make_features(df, "is_comic_video", CONFIG)
        Model.fit(X_is_comic, y_is_comic)

        # X_is_comic_test, y_is_comic_test = make_features(df_test, "is_comic_video", CONFIG)
        is_comic_predictions = Model.predict(X_is_comic)

        Model = make_model("is_name")
        X_is_name, y_is_name, sentences_with_id = make_features(df, "is_name", CONFIG)
        Model.fit(X_is_name, y_is_name)

        # X_is_name_test, y_is_name_test = make_features(df_test, "is_name", CONFIG)
        predictions_lists = []
        cpt = 0


        for step, x in enumerate(is_comic_predictions):
            if x == 1: # if comics sentence
                for row in df_test["video_name"]:
                    liste = []
                    for x in range(len(row.split())):
                        if cpt <= (len(X_is_name) - 1):
                            liste.append(Model.predict(X_is_name[cpt])[0])
                            cpt += 1
                    predictions_lists.append((liste, row.split()))
            else:
                predictions_lists.append(([], []))
        predictions_df = pd.DataFrame({'Prediction': predictions_lists})
        print("here")


            #     print(X_is_name_test[step])
            #     a = Model.predict(X_is_name_test[step])
            #     print(a)
            #     predictions_lists.append((a, True))
            # else: # if not comics sentence
            #     predictions_lists.append((None, False))

        print(predictions_lists)

        # for x in range(len(is_comic_predictions)):
        #     if is_comic_predictions[x] == 1:
        #         for row, row2 in zip(df_test["video_name"], df_test["is_name"]):
        #             liste = []
        #             for x in range(len(row.split())):
        #                 if cpt <= (len(X_is_name_test) - 1):
        #                     liste.append(Model.predict(X_is_name_test[cpt])[0])
        #                     cpt += 1
        #             row2 = ast.literal_eval(row2)
        #             predictions_lists.append((liste, row.split(), row2))
        #     else:
        #         predictions_lists.append(([], [], []))
        #
        # cpt = 0
        # predictions_df = pd.DataFrame({'Prediction': predictions_lists})
        #
        # find_comic_name_Accuracy = 0
        # for i in range(len(predictions_df)):
        #     if predictions_df.iloc[i][0][0] != [] and predictions_df.iloc[i][0][2] != []:
        #         cpt += 1
        #         if predictions_df.iloc[i][0][0] == predictions_df.iloc[i][0][2]:
        #             print(predictions_df.iloc[i][0][0], predictions_df.iloc[i][0][2])
        #             find_comic_name_Accuracy += 1
        # find_comic_name_Accuracy = find_comic_name_Accuracy / cpt
        # print(find_comic_name_Accuracy)
    model_name = "Stacking Classifier (Random Forest + MLP Classifier + Logistic Regression)"

    X, y = make_features(df, task, CONFIG)

    cv_scores = cross_val_score(Model, X, y)
    print(cv_scores)

    fichier_modeles = "Resultats/modeles_is_name.txt"

    prochain_modele = obtenir_prochain_numero_modele(fichier_modeles)

    sauvegarder_scores(task, fichier_modeles, prochain_modele, model_name, CONFIG, cv_scores, cv_scores.mean())





cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
