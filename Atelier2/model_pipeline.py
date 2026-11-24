"""
model_pipeline.py
=================

Pipeline ML modularisé — module « Alerte Suivi Glycémique » :
prédiction du résultat du test HbA1c (variable cible ``A1Cresult``).

Ce fichier regroupe, sous forme de fonctions réutilisables, les étapes du
notebook « A1Cresult - Molka Jebali » :

    - prepare_data()   : charger et prétraiter les données (Lab 1) ;
    - train_model()    : entraîner le modèle final (Lab 2, section 21) ;
    - evaluate_model() : évaluer les performances sur le jeu de test (Lab 3) ;
    - save_model()     : sauvegarder le modèle entraîné avec joblib ;
    - load_model()     : charger un modèle sauvegardé.

Le modèle final retenu dans le notebook est le « Deux étages CatBoost » :
    étage 1 : l'examen HbA1c est-il réalisé ou non ?
    étage 2 : Anormal contre Norm, parmi les patients testés.

Tous les paramètres (graine 42, découpage 80/20 stratifié, 300 itérations
CatBoost par étage, etc.) sont repris à l'identique du notebook.
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                   StandardScaler, label_binarize)

# --------------------------------------------------------------------------
# Constantes (reprises du notebook)
# --------------------------------------------------------------------------
DEFAULT_CSV = "diabetic_data.csv"
RANDOM_STATE = 42
CLASSES = ["Anormal", "None", "Norm"]

# Colonnes catégorielles brutes encodées nativement par CatBoost (notebook, section 17)
TE_COLS = ["diag_1_3", "diag_2_3", "diag_3_3", "specialty_cat", "discharge_cat"]

AGE_ORDER = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]

CATEGORICAL_NOMINAL = ["race", "gender", "medical_specialty",
                       "diag_1_group", "diag_2_group", "diag_3_group",
                       "max_glu_serum", "change", "diabetesMed"]

NUM_COLS = ["num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient",
            "number_diagnoses", "time_in_hospital"]

MED_COLS_RAW = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
                "glimepiride", "glipizide", "glyburide", "pioglitazone",
                "rosiglitazone", "insulin"]

# Colonnes du fichier CSV lues directement par le prétraitement
REQUIRED_COLUMNS = (
    ["encounter_id", "patient_nbr", "race", "gender", "age", "weight",
     "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3",
     "max_glu_serum", "A1Cresult", "change", "diabetesMed",
     "admission_type_id", "discharge_disposition_id"]
    + NUM_COLS + MED_COLS_RAW
)


# --------------------------------------------------------------------------
# Fonctions utilitaires de prétraitement
# --------------------------------------------------------------------------
def merge_target(value):
    """Simplifie la cible : fusionne ``>7`` et ``>8`` en ``Anormal``.

    Args:
        value (str): valeur brute de ``A1Cresult`` (None, Norm, >7, >8).

    Returns:
        str: ``'Anormal'`` pour ``>7`` / ``>8``, sinon la valeur inchangée.
    """
    if value in [">7", ">8"]:
        return "Anormal"
    return value  # 'None' et 'Norm' restent inchangés


def group_diag(code):
    """Regroupe un code diagnostic CIM-9 en grande catégorie clinique.

    Args:
        code (str): code diagnostic (ex. ``'250.83'``, ``'V27'``, ``'Missing'``).

    Returns:
        str: nom de la catégorie clinique.
    """
    if code == "Missing":
        return "Missing"
    try:
        code_f = float(code)
    except ValueError:
        return "Autre"
    if 390 <= code_f <= 459 or code_f == 785:
        return "Circulatoire"
    elif 250 <= code_f < 251:
        return "Diabete"
    elif 460 <= code_f <= 519 or code_f == 786:
        return "Respiratoire"
    elif 520 <= code_f <= 579 or code_f == 787:
        return "Digestif"
    elif 800 <= code_f <= 999:
        return "Traumatisme"
    elif 710 <= code_f <= 739:
        return "Musculo_squelettique"
    elif 580 <= code_f <= 629:
        return "Genito_urinaire"
    elif 140 <= code_f <= 239:
        return "Neoplasme"
    else:
        return "Autre"


# --------------------------------------------------------------------------
# Modèles (classes reprises du notebook, sections 18 et 20)
# --------------------------------------------------------------------------
class TwoStageClassifier(ClassifierMixin, BaseEstimator):
    """Classifieur en deux étages.

    Étage 1 : examen réalisé ou non.
    Étage 2 : Anormal contre Norm, parmi les patients testés.
    Les classes sont supposées triées dans l'ordre (Anormal, None, Norm).
    """

    def __init__(self, stage1=None, stage2=None):
        self.stage1 = stage1
        self.stage2 = stage2

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 3
        c_anormal, c_none, c_norm = self.classes_
        tested = y != c_none
        self.s1_ = clone(self.stage1).fit(X, tested.astype(int))
        self.s2_ = clone(self.stage2).fit(X[tested], (y[tested] == c_anormal).astype(int))
        return self

    def predict_proba(self, X):
        p_tested = self.s1_.predict_proba(X)[:, 1]
        q = self.s2_.predict_proba(X)[:, 1]
        return np.column_stack([p_tested * q, 1 - p_tested, p_tested * (1 - q)])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class CatWrap(ClassifierMixin, BaseEstimator):
    """CatBoost avec classes pondérées ; les variables catégorielles brutes
    (``TE_COLS``) sont traitées nativement."""

    def __init__(self, iterations=400, depth=6, learning_rate=0.06):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        cats = [c for c in TE_COLS if c in X.columns]
        self.m_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            auto_class_weights="Balanced", cat_features=cats, verbose=0, thread_count=2,
            random_seed=42, allow_writing_files=False)
        self.m_.fit(X, self.le_.transform(y))
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model():
    """Construit (sans l'entraîner) le modèle final du notebook :
    « Deux étages CatBoost » (300 itérations par étage).

    Returns:
        TwoStageClassifier: modèle non entraîné.
    """
    return TwoStageClassifier(stage1=CatWrap(iterations=300),
                              stage2=CatWrap(iterations=300))


# --------------------------------------------------------------------------
# 1) prepare_data
# --------------------------------------------------------------------------
def prepare_data(csv_path=DEFAULT_CSV, test_size=0.2, random_state=RANDOM_STATE):
    """Charge et prétraite les données (Lab 1 du notebook), puis les découpe.

    Étapes : fusion de la cible en 3 classes, dédoublonnage des patients,
    suppression des colonnes inutiles, traitement des valeurs manquantes,
    suppression des colonnes constantes, regroupement des diagnostics,
    encodage (ordinal de ``age``, one-hot), mise à l'échelle, feature
    engineering, renommage des colonnes, découpage train/test stratifié,
    ajout des 5 colonnes catégorielles brutes (``TE_COLS``) utilisées
    nativement par CatBoost.

    Args:
        csv_path (str): chemin du fichier ``diabetic_data.csv``.
        test_size (float): part du jeu de test (0.2 dans le notebook).
        random_state (int): graine (42 dans le notebook).

    Returns:
        tuple: ``(X_train, X_test, y_train, y_test)``.

    Raises:
        FileNotFoundError: si le fichier CSV n'existe pas.
        ValueError: si des colonnes attendues sont absentes du CSV.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier de données introuvable : {csv_path}")

    # keep_default_na=False : le texte littéral "None" (test non réalisé)
    # ne doit pas être transformé en NaN.
    df = pd.read_csv(csv_path, keep_default_na=False)

    absentes = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if absentes:
        raise ValueError(f"Colonnes manquantes dans {csv_path} : {absentes}")

    # --- Cible : 4 classes -> 3 classes (None, Norm, Anormal)
    df["A1C_target"] = df["A1Cresult"].apply(merge_target)

    # --- 3.1 Doublons patients : premier épisode de chaque patient
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first").copy()

    # --- 3.2 Colonnes trop incomplètes ou non pertinentes
    df = df.drop(columns=["weight", "payer_code", "encounter_id"])

    # --- 3.3 Valeurs manquantes restantes
    for col in ["medical_specialty", "diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].replace("?", "Missing")
    imputer = SimpleImputer(missing_values="?", strategy="most_frequent")
    df["race"] = imputer.fit_transform(df[["race"]]).ravel()

    # --- 3.4 Colonnes constantes
    near_constant = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=near_constant)

    # --- 4. Feature selection : séparation cible / variables explicatives
    y = df["A1C_target"]
    X = df.drop(columns=["A1Cresult", "A1C_target", "patient_nbr"])

    # --- 5.1 Regroupement des diagnostics
    for col in ["diag_1", "diag_2", "diag_3"]:
        X[col + "_group"] = X[col].apply(group_diag)
    X = X.drop(columns=["diag_1", "diag_2", "diag_3"])

    # --- 5.2 Encodage ordinal de age
    enc_age = OrdinalEncoder(categories=[AGE_ORDER])
    X["age_encoded"] = enc_age.fit_transform(X[["age"]])
    X = X.drop(columns=["age"])

    # --- 5.3 Encodage one-hot des variables catégorielles
    categorical_nominal = list(CATEGORICAL_NOMINAL)
    non_numeric_cols = X.select_dtypes(exclude="number").columns.tolist()
    med_cols = [c for c in non_numeric_cols if c not in categorical_nominal]
    categorical_nominal += med_cols
    X = pd.get_dummies(X, columns=categorical_nominal, drop_first=True)

    # --- 5.4 Mise à l'échelle des variables numériques
    scaler = StandardScaler()
    X[NUM_COLS] = scaler.fit_transform(X[NUM_COLS])

    # --- 6. Feature engineering
    df["total_visits_prior"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["nb_meds_changed"] = (df[MED_COLS_RAW] != "No").sum(axis=1)
    X["total_visits_prior"] = StandardScaler().fit_transform(df[["total_visits_prior"]]).ravel()
    X["nb_meds_changed"] = StandardScaler().fit_transform(df[["nb_meds_changed"]]).ravel()

    # --- 6.1 Feature engineering avancé (12 variables, sans utiliser la cible)
    d = df.loc[X.index]
    days = d["time_in_hospital"].clip(lower=1)
    grp = d[["diag_1", "diag_2", "diag_3"]].apply(lambda col: col.map(group_diag))

    X["meds_per_day"] = d["num_medications"] / days
    X["labs_per_day"] = d["num_lab_procedures"] / days
    X["diag_per_day"] = d["number_diagnoses"] / days
    X["nb_meds_dose_changed"] = d[MED_COLS_RAW].isin(["Up", "Down"]).sum(axis=1)
    X["insulin_dose_changed"] = d["insulin"].isin(["Up", "Down"]).astype(int)
    X["nb_diag_diabetes"] = (grp == "Diabete").sum(axis=1)
    X["nb_diag_groups"] = grp.nunique(axis=1)
    X["has_prior_inpatient"] = (d["number_inpatient"] > 0).astype(int)
    X["high_utilizer"] = (d["total_visits_prior"] >= 3).astype(int)
    X["emergency_admission"] = (d["admission_type_id"] == 1).astype(int)
    X["age_senior"] = (X["age_encoded"] >= 7).astype(int)
    X["polypharmacy"] = (d["num_medications"] >= 15).astype(int)

    # --- Normalisation des noms de colonnes (caractères spéciaux)
    X.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in X.columns]
    assert X.columns.is_unique

    # --- 8. Découpage train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- 17. Variables catégorielles brutes (traitées nativement par CatBoost)
    raw = df.loc[X.index]
    raw_cat = pd.DataFrame({
        "diag_1_3": raw["diag_1"].astype(str).str[:3],
        "diag_2_3": raw["diag_2"].astype(str).str[:3],
        "diag_3_3": raw["diag_3"].astype(str).str[:3],
        "specialty_cat": raw["medical_specialty"].astype(str),
        "discharge_cat": raw["discharge_disposition_id"].astype(str),
    }, index=X.index)

    X_train = X_train.join(raw_cat)
    X_test = X_test.join(raw_cat)

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# 2) train_model
# --------------------------------------------------------------------------
def train_model(X_train, y_train):
    """Entraîne le modèle final du notebook (deux étages CatBoost).

    Args:
        X_train (pd.DataFrame): variables explicatives (sortie de
            ``prepare_data``, avec les colonnes ``TE_COLS``).
        y_train (pd.Series): cible à 3 classes (Anormal, None, Norm).

    Returns:
        TwoStageClassifier: modèle entraîné.

    Raises:
        ValueError: si ``X_train`` et ``y_train`` n'ont pas la même longueur
            ou si une colonne de ``TE_COLS`` manque.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train doivent avoir le même nombre de lignes.")
    manquantes = [c for c in TE_COLS if c not in X_train.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans X_train : {manquantes}")

    model = build_model()
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------
# 3) evaluate_model
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, verbose=True):
    """Évalue un modèle entraîné sur le jeu de test avec la batterie de
    métriques du notebook (Lab 2, section 8 et Lab 3, section 22).

    Args:
        model: modèle entraîné (``predict`` et ``predict_proba``).
        X_test (pd.DataFrame): variables explicatives de test.
        y_test (pd.Series): cible réelle de test.
        verbose (bool): affiche le rapport de classification et la matrice
            de confusion.

    Returns:
        dict: métriques (Accuracy, Balanced accuracy, Precision/Recall macro,
        F1 macro, F1 pondéré, MCC, Kappa, F1 par classe, ROC-AUC (OvR),
        PR-AUC (macro), Log-loss).
    """
    if len(X_test) != len(y_test):
        raise ValueError("X_test et y_test doivent avoir le même nombre de lignes.")

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    y_bin = label_binarize(y_test, classes=CLASSES)
    m = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 pondéré": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Kappa": cohen_kappa_score(y_test, y_pred),
    }
    for c, f in zip(CLASSES, f1_score(y_test, y_pred, labels=CLASSES, average=None, zero_division=0)):
        m["F1 " + c] = f
    m["ROC-AUC (OvR)"] = roc_auc_score(y_bin, proba, average="macro")
    m["PR-AUC (macro)"] = average_precision_score(y_bin, proba, average="macro")
    m["Log-loss"] = log_loss(y_test, proba, labels=CLASSES)

    if verbose:
        print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0, digits=3))
        cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
        print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
        print(pd.DataFrame(cm, index=[f"réel {c}" for c in CLASSES],
                           columns=[f"prédit {c}" for c in CLASSES]))
        print()
        for k, v in m.items():
            print(f"{k:<20s} {v:.3f}")
    return m


# --------------------------------------------------------------------------
# 4) save_model / 5) load_model
# --------------------------------------------------------------------------
def save_model(model, path="model.joblib"):
    """Sauvegarde un modèle entraîné avec joblib.

    Args:
        model: modèle entraîné.
        path (str): chemin du fichier de sortie.

    Returns:
        str: chemin du fichier créé.
    """
    dossier = os.path.dirname(os.path.abspath(path))
    os.makedirs(dossier, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib.

    Le module ``model_pipeline`` doit être importable (les classes
    ``TwoStageClassifier`` et ``CatWrap`` y sont définies).

    Args:
        path (str): chemin du fichier du modèle.

    Returns:
        Le modèle chargé.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)
