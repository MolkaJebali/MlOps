"""
main.py
=======

Point d'entrée : exécute les étapes de model_pipeline.py via des arguments CLI.

Exemples :
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --train --evaluate
    python main.py --all
    python main.py --train --data diabetic_data.csv --model-path model.joblib
"""

import argparse
import sys

from model_pipeline import (DEFAULT_CSV, evaluate_model, load_model,
                            prepare_data, save_model, train_model)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pipeline ML — prédiction du résultat HbA1c (A1Cresult)")
    parser.add_argument("--data", default=DEFAULT_CSV,
                        help="chemin du fichier CSV (défaut : %(default)s)")
    parser.add_argument("--model-path", default="model.joblib",
                        help="fichier du modèle sauvegardé (défaut : %(default)s)")
    parser.add_argument("--prepare", action="store_true",
                        help="charger et prétraiter les données (affiche les dimensions)")
    parser.add_argument("--train", action="store_true",
                        help="entraîner le modèle puis le sauvegarder")
    parser.add_argument("--evaluate", action="store_true",
                        help="charger le modèle sauvegardé et l'évaluer sur le jeu de test")
    parser.add_argument("--all", action="store_true",
                        help="exécuter toutes les étapes : préparer, entraîner, sauvegarder, évaluer")
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    if args.all:
        args.prepare = args.train = args.evaluate = True
    if not (args.prepare or args.train or args.evaluate):
        parser.print_help()
        return 1

    print(f"[prepare] Chargement et prétraitement de {args.data} ...")
    X_train, X_test, y_train, y_test = prepare_data(args.data)
    print(f"[prepare] Train : {X_train.shape} | Test : {X_test.shape}")
    print(y_train.value_counts(normalize=True).round(3).to_string())

    if args.train:
        print("[train] Entraînement du modèle (deux étages CatBoost) ...")
        model = train_model(X_train, y_train)
        path = save_model(model, args.model_path)
        print(f"[save] Modèle sauvegardé : {path}")

    if args.evaluate:
        print(f"[load] Chargement du modèle : {args.model_path}")
        model = load_model(args.model_path)
        print("[evaluate] Évaluation sur le jeu de test :")
        evaluate_model(model, X_test, y_test)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as err:
        print(f"Erreur : {err}", file=sys.stderr)
        sys.exit(1)
