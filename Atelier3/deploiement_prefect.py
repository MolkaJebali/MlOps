"""
deploiement_prefect.py — Planification des flows (Atelier 3, étape 5)

Prérequis : le serveur Prefect tourne (Terminal 1 : prefect server start)
et l'URL de l'API est configurée :
    prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

Lancement (Terminal 2, reste ouvert : c'est le "worker") :
    python deploiement_prefect.py

Déclenchement manuel (Terminal 3) :
    prefect deployment ls
    prefect deployment run 'all/ml-pipeline-all'
    prefect deployment run 'train/ml-pipeline-train'
"""
from prefect import serve

from pipeline_prefect import flow_all, flow_evaluate, flow_train

if __name__ == "__main__":
    deploiement_all = flow_all.to_deployment(
        name="ml-pipeline-all",
        cron="0 8 * * *",  # tous les jours à 08h00
    )
    deploiement_train = flow_train.to_deployment(
        name="ml-pipeline-train",
        cron="0 9 * * *",  # tous les jours à 09h00
    )
    deploiement_evaluate = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",  # sans planification : lancement manuel
    )
    serve(deploiement_all, deploiement_train, deploiement_evaluate)
