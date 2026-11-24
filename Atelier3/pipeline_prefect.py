"""
pipeline_prefect.py — Orchestration du pipeline ML avec Prefect (Atelier 3)

Projet : Alerte Suivi Glycémique (prédiction de A1Cresult)

Lancement :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
"""
import argparse
import subprocess
import sys

from prefect import flow, task

# Les fonctions sont appelées via main.py (qui les importe de model_pipeline.py)
from main import (
    DEFAULT_CSV,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

MODEL_PATH = "model.joblib"

# Seuls ces fichiers sont analysés (sinon qualité/sécurité scannent tout le venv)
FICHIERS = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd, tolerant=False):
    """Exécute une commande ; lève une erreur si elle échoue (sauf tolerant)."""
    print(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    if res.returncode != 0 and not tolerant:
        raise RuntimeError(f"Échec ({res.returncode}) : {' '.join(cmd)}")


# ----------------------------- TASKS (2.1) -----------------------------
@task(name="Installer les dépendances")
def installer_dependances():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Préparation des données")
def preparer_donnees(csv_path=DEFAULT_CSV):
    return prepare_data(csv_path)  # X_train, X_test, y_train, y_test


@task(name="Entraînement du modèle")
def entrainer_modele(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarder le modèle")
def sauvegarder_modele(model, path=MODEL_PATH):
    return save_model(model, path)


@task(name="Charger le modèle")
def charger_modele(path=MODEL_PATH):
    return load_model(path)


@task(name="Évaluer le modèle")
def evaluer_modele(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


# ----------------------- TASKS code (étape 4) -----------------------
@task(name="Formatage du code (black)")
def formater_code():
    run([sys.executable, "-m", "black", *FICHIERS])


@task(name="Qualité du code (flake8)")
def qualite_code():
    # tolerant : on affiche les avertissements sans bloquer le pipeline
    run(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *FICHIERS],
        tolerant=True,
    )


@task(name="Sécurité du code (bandit)")
def securite_code():
    run([sys.executable, "-m", "bandit", "-q", *FICHIERS], tolerant=True)


@task(name="Tests unitaires (pytest)")
def tests_unitaires():
    run([sys.executable, "-m", "pytest", "-q", "test_model_pipeline.py"])


# ----------------------------- FLOWS (2.2 + étape 4) -----------------------------
@flow(name="code")
def flow_code():
    formater_code()
    qualite_code()
    securite_code()
    tests_unitaires()


@flow(name="train")
def flow_train():
    X_train, X_test, y_train, y_test = preparer_donnees()
    model = entrainer_modele(X_train, y_train)
    sauvegarder_modele(model)  # nécessaire pour que le flow evaluate puisse le recharger


@flow(name="evaluate")
def flow_evaluate():
    # les données de test sont recalculées (même random_state=42 -> même découpage)
    X_train, X_test, y_train, y_test = preparer_donnees()
    model = charger_modele()
    evaluer_modele(model, X_test, y_test)


@flow(name="all")
def flow_all():
    installer_dependances()
    flow_code()  # étape 4 : avant la préparation des données
    X_train, X_test, y_train, y_test = preparer_donnees()
    model = entrainer_modele(X_train, y_train)
    sauvegarder_modele(model)
    evaluer_modele(model, X_test, y_test)


FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "entrainement": flow_train,  # alias : l'énoncé utilise les deux noms
    "evaluate": flow_evaluate,
    "code": flow_code,
}

# ----------------------------- SCRIPT (2.3) -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML orchestré avec Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
