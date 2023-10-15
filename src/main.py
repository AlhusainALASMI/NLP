import click
from sklearn.model_selection import cross_val_score
import pickle
from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model.main import make_model
from src.model.main import dump
import pandas as pd
import os


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
def sauvegarder_scores(fichier,model_number, model, configuration, scores, moyenne_scores):
    with open(fichier, 'a') as f:
        f.write(f"Model {model_number}\n")
        f.write(f"Model {model}\n")
        f.write(f"Modele utilise: {model}\n")
        f.write(f"Configuration utilisee: {', '.join(configuration)}\n")
        f.write(f"Scores: {', '.join(map(str, scores))}\n")
        f.write(f"Moyenne des scores: {moyenne_scores}\n\n")
@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    model, CONFIG = make_model()

    df = make_dataset(input_filename)

    X, y = make_features(df, task, CONFIG)

    model.fit(X, y)

    return dump(model, CONFIG, model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="src/models/dump.json", help="File to dump model")
@click.option("--output_filename", default="src/data/processed/prediction.csv", help="Output file for predictions")
def predict(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)

    load  = pickle.load(open(model_dump_filename, 'rb'))

    CONFIG = load["CONFIG"]
    Model = load["model"]

    X, y = make_features(df, task, CONFIG)

    predictions = Model.predict(X)

    predictions_df = pd.DataFrame({'Prediction': predictions})

    # Save the DataFrame to a CSV file
    return predictions_df.to_csv(output_filename, index=False)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="src/data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    df = make_dataset(input_filename)

    Model, CONFIG = make_model()
    model_name = "Ada Boost Classifier"
    X, y = make_features(df, task, CONFIG)

    cv_scores = cross_val_score(Model, X, y)

    fichier_modeles = "Resultats/modeles_AdaBoostClassifier.txt"

    prochain_modele = obtenir_prochain_numero_modele(fichier_modeles)

    sauvegarder_scores(fichier_modeles, prochain_modele, model_name, CONFIG, cv_scores, cv_scores.mean())





cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
