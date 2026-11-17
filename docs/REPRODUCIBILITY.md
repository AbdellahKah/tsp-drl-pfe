# Reproductibilité et exécution bornée

## Contrôles locaux autorisés

Installer Python et PyTorch, exécuter `python -m pip install -r requirements.txt`, puis `python scripts/verify_checkpoints.py` et `python -m pytest -q`. Tester en CPU : `python scripts/quick_inference.py --n 20 --decode greedy`, de même pour 50 et 100. Un smoke test de sampling peut utiliser `--decode best-of-512 --sample-count 3`; ce budget réduit n'a aucune valeur de résultat expérimental.

La réalisation d'une nouvelle expérimentation nécessite **les mêmes coordonnées, les mêmes versions, les notebooks originaux et les checkpoints de chaque seed (les trois sélectionnés sont maintenant inclus)**. Les CSV archivés sont des observations de référence, non des données simulées par le dépôt.

## Environnement des expériences archivées

PyTorch 2.11.0+cu130, CUDA 13.0 et GPU Tesla T4 pour les évaluations finales; Python 3.13.15 pour seed 20261013, 3.13.16 pour 20261014/15. Cette livraison ne verrouille pas un conteneur identique, et aucun résultat local CPU ne doit être présenté comme re-réplication de l'étude.

## Étapes interdites par défaut

Ne pas lancer automatiquement les entraînements de 13 000 updates, les sweeps Best-of-512 à grande échelle, l'agrégation contre de nouveaux dossiers Drive ni de nouveaux benchmarks complets. Aucun stockage distant ou accès à l'encadrant ne doit être annoncé sans vérification préalable.

## Contrôle des trois poids acteurs sélectionnés

Les fichiers d'origine seed-13/14/15 sont inclus dans `models/`, organisés par seed et listés dans `artifacts/aggregation/multiseed_selected_checkpoints.csv`. Utiliser `python scripts/verify_checkpoints.py` pour comparer toutes les empreintes SHA-256 avec les métadonnées d’évaluation finale. Ne pas remplacer ces fichiers par des checkpoints voisins ou de nouveaux poids.

La commande `quick_inference.py` accepte un `--checkpoint` explicite, par exemple :

```bash
python scripts/quick_inference.py --n 50 --decode greedy --checkpoint models/seed_20261014/train20_1025_04_27_step12999_act.pt
python scripts/quick_inference.py --n 100 --decode greedy --checkpoint models/seed_20261015/train20_1025_04_56_step12999_act.pt
```

Ces contrôles utilisent une seule instance générée localement (et ne constituent **ni les 100 tests du mémoire, ni une nouvelle estimation des performances**). Les chemins Drive originaux et les historiques complets d'entraînement ne sont pas reproduits.
