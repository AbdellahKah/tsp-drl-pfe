# Architecture du code et correspondance scientifique

| Composant | Fichier (préservé) | Responsabilité | Trajet expérimental |
|---|---|---|---|
| Acteur | `code_snapshot/actor.py`, classe `PtrNet1` | Embedding linéaire des coordonnées (2D), encodeur LSTM, décodeur LSTM, glimpse et scores de pointeurs, masque des villes visitées, permutation et log-vraisemblance | Entraînement + évaluations greedy/sampling |
| Baseline apprise | `code_snapshot/critic.py`, classe `PtrNet2` | Embedding, encodeur LSTM, attention itérative, sorties linéaires de régression d'un coût scalaire par instance | Entraînement uniquement |
| Environnement | `code_snapshot/env.py`, `Env_tsp.stack_l_fast` | Longueur euclidienne du cycle fermé, y compris arc dernière→première ville | Entraînement et contrôles de coût |
| Générateur | `code_snapshot/data.py`, `Generator` | Coordonnées aléatoires / DataLoader des instances | Entraînement |
| Config | `code_snapshot/config.py` | Options, sérialisation des paramètres d'exécution | Notebook de lancement de l'entraînement |
| Mise à jour | `code_snapshot/train.py` | Critic MSE `(pred_l, real_l.detach())`; avantage détaché `real_l - pred_l`; perte acteur `(adv * ll).mean()`; Adam | Entraînement + patch conservé dans `experiment_patch/` |
| Recherche historique | `code_snapshot/search.py`, `test.py` | Sampling et active search de la base de code | **Non** utilisées comme protocole final Greedy/Best-of-512 |
| Décodage et validation | `notebooks/seed_20261014_evaluation_v1_2.ipynb`, `notebooks/seed_20261015_evaluation_corrected_torch2110_v1_2.ipynb`, `notebooks/seed_20261013_evaluation_v1_1.ipynb` | Chargement de `PtrNet1`, greedy à 1 tournée, échantillonnage à 512, sélection de la plus courte, validations | Évaluation finale |
| Heuristiques | `src/tsp_drl/algorithms/reference_baselines.py` | NN (départ ville 0), amélioration 2-opt, ACO fixe | Extraction verbatim du notebook seed-15 corrigé, cellule 2 |
| Agrégation | `scripts/aggregate_multiseed_results_TORCH2110_FIXED.py` | Résultats inter-seeds / vérifications et bootstrap | Script final corrigé |

## Calcul différentiel réellement implémenté

Le chemin de calcul construit une tournée discrète `pi` et évalue sa longueur euclidienne `real_l`. Le critic prédit `pred_l` et est optimisé sur une **erreur quadratique** contre la longueur réelle détachée. Le gradient de l'acteur provient de la log-vraisemblance de la tournée (gradient de politique REINFORCE) pondérée par l'avantage `(real_l - pred_l).detach()`. La sélection discrète des villes elle-même n'est pas différentiable.

Le `search.py` initial contient une procédure « active_search » distincte; ne pas la présenter comme le protocole de résultats du mémoire. Les arguments du chemin de décodage final se trouvent dans les notebooks d'évaluation, non dans `test.py`.

Le `train.py` préservé a déjà des lignes du patch : périodicité des checkpoints, suppression d'early stopping et seeds. Aucun changement silencieux n'a été introduit dans ces modules lors de l'assemblage.
