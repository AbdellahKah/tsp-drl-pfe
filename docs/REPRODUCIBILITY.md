# Reproductibilité et exécution bornée

## Contrôles locaux autorisés

Installer Python et PyTorch, exécuter `python -m pip install -r requirements.txt` puis `python -m pytest -q`. Tester en CPU : `python scripts/quick_inference.py --n 20 --decode greedy`, de même pour 50 et 100. Un smoke test de sampling peut utiliser `--decode best-of-512 --sample-count 3`; ce budget réduit n'a aucune valeur de résultat expérimental.

La réalisation d'une nouvelle expérimentation nécessite **les mêmes coordonnées, les mêmes versions, les checkpoint binaries de chaque seed et les notebooks originaux**. Les CSV archivés sont des observations de référence, non des données simulées par le dépôt.

## Environnement des expériences archivées

PyTorch 2.11.0+cu130, CUDA 13.0 et GPU Tesla T4 pour les évaluations finales; Python 3.13.15 pour seed 20261013, 3.13.16 pour 20261014/15. Cette livraison ne verrouille pas un conteneur identique, et aucun résultat local CPU ne doit être présenté comme re-réplication de l'étude.

## Étapes interdites par défaut

Ne pas lancer automatiquement les entraînements de 13 000 updates, les sweeps Best-of-512 à grande échelle, l'agrégation contre de nouveaux dossiers Drive ni de nouveaux benchmarks complets. Aucun stockage distant ou accès à l'encadrant ne doit être annoncé sans vérification préalable.

## Comment reconstituer les deux poids manquants

Récupérer uniquement les fichiers **originaux** nommés dans `artifacts/aggregation/multiseed_selected_checkpoints.csv`, puis contrôler leurs SHA-256 avant de les importer. Aucune substitution d'un modèle voisin ou de nouvelles poids aléatoires n'est valide.
