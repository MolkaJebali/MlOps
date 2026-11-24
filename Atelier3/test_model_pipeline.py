"""
test_model_pipeline.py — Tests du projet « Alerte Suivi Glycémique » (Atelier 3, étape 4)

  1. Tests unitaires    : chaque fonction isolée, sans le vrai jeu de données
  2. Tests fonctionnels : le pipeline complet sur un échantillon du vrai CSV
  3. Tests de robustesse : entrées invalides rejetées proprement
  4. Tests qualité/sécurité : Flake8 (qualité) et Bandit (sécurité du code)

Lancement :  python -m pytest -q test_model_pipeline.py
"""

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import main as main_module
from model_pipeline import (
    DEFAULT_CSV,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

ERREURS_ATTENDUES = (FileNotFoundError, ValueError)

ROOT = Path(__file__).parent
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TEST_FILE = "test_model_pipeline.py"


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def echantillon_csv(tmp_path_factory):
    """Petit échantillon (3000 lignes) du vrai CSV, pour des tests rapides."""
    if not os.path.exists(DEFAULT_CSV):
        pytest.skip(f"{DEFAULT_CSV} introuvable : tests fonctionnels ignorés")
    chemin = tmp_path_factory.mktemp("data") / "echantillon.csv"
    # dtype=str + keep_default_na=False : pandas ne doit pas transformer le texte
    # "None" (valeur de A1Cresult) en valeur manquante, puis en chaîne vide.
    complet = pd.read_csv(DEFAULT_CSV, dtype=str, keep_default_na=False)
    echantillon = complet.sample(3000, random_state=42)

    # Un tirage aléatoire peut perdre des valeurs rares (ex. un médicament presque
    # jamais prescrit) : prepare_data supprimerait alors la colonne devenue constante.
    # On rajoute donc une ligne pour chaque valeur absente des colonnes catégorielles.
    ajouts = []
    for col in complet.columns:
        if complet[col].nunique() <= 30:
            manquantes = set(complet[col].dropna().unique()) - set(
                echantillon[col].dropna().unique()
            )
            for valeur in manquantes:
                ajouts.append(complet[complet[col] == valeur].head(1))
    if ajouts:
        echantillon = pd.concat([echantillon, *ajouts]).drop_duplicates()

    echantillon.to_csv(chemin, index=False)
    return str(chemin)


@pytest.fixture(scope="module")
def donnees(echantillon_csv):
    return prepare_data(echantillon_csv)  # X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def modele(donnees):
    X_train, _, y_train, _ = donnees
    return train_model(X_train, y_train)


# ----------------------------------------------------------------------------
# 1. TESTS UNITAIRES
# ----------------------------------------------------------------------------
class TestUnitaires:
    def test_prepare_data_fichier_absent(self):
        with pytest.raises(FileNotFoundError):
            prepare_data("fichier_inexistant.csv")

    def test_prepare_data_colonnes_manquantes(self, tmp_path):
        csv = tmp_path / "mauvais.csv"
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(csv, index=False)
        with pytest.raises(ERREURS_ATTENDUES):
            prepare_data(str(csv))

    def test_train_model_longueurs_differentes(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series(["None", "Norm", "Anormal"])  # 3 valeurs pour 4 lignes
        with pytest.raises(ERREURS_ATTENDUES):
            train_model(X, y)

    def test_load_model_fichier_absent(self, tmp_path):
        with pytest.raises(ERREURS_ATTENDUES):
            load_model(str(tmp_path / "absent.joblib"))

    def test_save_puis_load_conserve_les_predictions(self, tmp_path):
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        y = ["None", "Norm", "None", "Anormal"]
        petit_modele = DummyClassifier(strategy="most_frequent").fit(X, y)

        chemin = save_model(petit_modele, str(tmp_path / "m.joblib"))
        assert os.path.exists(chemin)

        recharge = load_model(chemin)
        assert list(recharge.predict(X)) == list(petit_modele.predict(X))


# ----------------------------------------------------------------------------
# 2. TESTS FONCTIONNELS (pipeline complet sur échantillon)
# ----------------------------------------------------------------------------
class TestFonctionnels:
    def test_prepare_data_dimensions_coherentes(self, donnees):
        X_train, X_test, y_train, y_test = donnees
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]
        # découpage 80/20 (tolérance pour l'arrondi)
        part_test = len(X_test) / (len(X_train) + len(X_test))
        assert 0.15 < part_test < 0.25

    def test_les_trois_classes_sont_presentes(self, donnees):
        _, _, y_train, _ = donnees
        assert set(y_train.unique()) == {"None", "Norm", "Anormal"}

    def test_entrainement_et_prediction(self, modele, donnees):
        _, X_test, _, _ = donnees
        predictions = np.ravel(modele.predict(X_test))
        assert len(predictions) == len(X_test)
        assert set(predictions) <= {"None", "Norm", "Anormal"}

    def test_evaluate_model_renvoie_des_metriques(self, modele, donnees):
        _, X_test, _, y_test = donnees
        metriques = evaluate_model(modele, X_test, y_test, verbose=False)
        assert isinstance(metriques, dict) and len(metriques) > 0

    def test_sauvegarde_et_rechargement_du_vrai_modele(self, modele, donnees, tmp_path):
        _, X_test, _, _ = donnees
        chemin = save_model(modele, str(tmp_path / "model.joblib"))
        recharge = load_model(chemin)
        assert list(np.ravel(recharge.predict(X_test))) == list(
            np.ravel(modele.predict(X_test))
        )

    def test_main_prepare_via_la_ligne_de_commande(self, echantillon_csv):
        code = main_module.main(["--prepare", "--data", echantillon_csv])
        assert code == 0

    def test_main_sans_argument_affiche_l_aide(self, capsys):
        code = main_module.main([])
        assert code == 1
        assert "usage" in capsys.readouterr().out.lower()


# ----------------------------------------------------------------------------
# 3. TESTS DE ROBUSTESSE (validation des entrées ; sécurité du code = Bandit)
# ----------------------------------------------------------------------------
class TestRobustesse:
    def test_csv_vide_rejete(self, tmp_path):
        vide = tmp_path / "vide.csv"
        vide.write_text("")
        with pytest.raises(Exception):
            prepare_data(str(vide))

    def test_chemin_avec_remontee_de_repertoire(self):
        # un chemin de type ../../ vers un fichier inexistant doit échouer proprement
        with pytest.raises(ERREURS_ATTENDUES):
            prepare_data("../../../etc/fichier_inexistant.csv")

    def test_chemin_avec_injection_de_commande_non_execute(self, tmp_path):
        piege = tmp_path / "pwned"
        with pytest.raises(ERREURS_ATTENDUES):
            prepare_data(f"x.csv; touch {piege}")
        assert not piege.exists()  # la commande injectée n'a pas été exécutée

    def test_fichier_modele_corrompu_rejete(self, tmp_path):
        faux = tmp_path / "faux.joblib"
        faux.write_bytes(b"ceci n'est pas un modele")
        with pytest.raises(Exception):
            load_model(str(faux))

    def test_arguments_cli_inconnus_rejetes(self):
        with pytest.raises(SystemExit):
            main_module.main(["--option-inexistante"])


# ----------------------------------------------------------------------------
# 4. QUALITÉ (Flake8) ET SÉCURITÉ DU CODE (Bandit)
# ----------------------------------------------------------------------------
def _outil(*args):
    """Lance un outil Python (flake8, bandit) sur le projet et renvoie le résultat."""
    return subprocess.run(  # nosec B603
        [sys.executable, "-m", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _bandit(*args):
    """Lance bandit en sortie JSON et renvoie la liste des problèmes trouvés."""
    result = _outil("bandit", "-f", "json", *args)
    return json.loads(result.stdout)["results"]


def _format_problemes(problemes):
    return "Failles bandit :\n" + "\n".join(
        f"{p['filename']}:{p['line_number']} [{p['test_id']}] {p['issue_text']}"
        for p in problemes
    )


class TestQualiteSecurite:
    def test_flake8_aucune_erreur_de_style(self):
        result = _outil("flake8", *PROJECT_FILES, TEST_FILE)
        assert result.returncode == 0, f"Erreurs flake8 :\n{result.stdout}"
        print("\n  flake8 : aucune erreur de style")

    def test_bandit_aucune_faille_dans_le_code(self):
        problemes = _bandit(*PROJECT_FILES)
        assert problemes == [], _format_problemes(problemes)
        print("\n  bandit : aucune faille dans le code")

    def test_bandit_aucune_faille_dans_les_tests(self):
        # B101 (assert) ignoré : pytest repose sur assert
        problemes = _bandit("-s", "B101", TEST_FILE)
        assert problemes == [], _format_problemes(problemes)
        print("\n  bandit : aucune faille dans les tests")
