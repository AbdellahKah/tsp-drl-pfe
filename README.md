# TSP–DRL — dépôt du projet de fin d'études

**Mémoire :** *Apprentissage par Renforcement Profond pour l’Optimisation Combinatoire : Étude Algorithmique du Problème du Voyageur de Commerce*  
**Auteur du mémoire :** Abdellah Kahlaoui  
**Établissement :** Faculté des Sciences et Techniques de Settat, Université Hassan Ier.  

## Méthode étudiée

Problème du voyageur de commerce euclidien symétrique en 2D. Un réseau de pointeurs (`PtrNet1`) à encodeur-décodeur LSTM choisit successivement les villes avec un masque des villes déjà parcourues. Son apprentissage utilise **REINFORCE avec baseline apprise** (`PtrNet2`, régression du coût total de tournée par MSE), et non un Actor–Critic TD pas-à-pas. Aucune architecture GNN n'est utilisée.

Trois entraînements indépendants à 20 villes : seeds **20261013 / 20261014 / 20261015**, 13 000 mises à jour par modèle, batch 512, dimension 128, Adam (taux initial 0,001). Checkpoints choisis uniquement sur la validation à 20 villes. Tests sur 100 instances à **n = 20, 50 et 100**; les deux dernières tailles mesurent le transfert hors distribution de taille d'entraînement. Décodages comparés : **Greedy** et **Best-of-512**. Heuristiques de référence : NN (PPV), NN+2-opt, ACO fixe.

## Organisation du dépôt

| Chemin | Contenu et provenance |
|---|---|
| `code_snapshot/` | Huit modules Python du workspace fourni; `train.py` y contient déjà le patch expérimental. Imports originaux conservés. |
| `experiment_patch/` | Diff du script d'entraînement et SHA amont déclaré par les manifestes. |
| `notebooks/` | Notebooks de recherche Colab, évaluations finales corrigées et historique clairement séparé. |
| `src/tsp_drl/algorithms/reference_baselines.py` | Fonctions NN, 2-opt, ACO **extraites textuellement** de l'évaluation v1.2 corrigée. |
| `scripts/quick_inference.py` | Adaptateur local de vérification avec le véritable acteur et le checkpoint inclus; ce n'est pas une reproduction de toutes les expériences. |
| `scripts/aggregate_multiseed_results_TORCH2110_FIXED.py` | Script final d'agrégation, conservé à l'identique. |
| `artifacts/` | CSV et manifestes finaux v1.2, sans inventer ni recalculer les observations. |
| `models/` | Les **trois** checkpoints acteurs sélectionnés (seeds 20261013, 20261014, 20261015), noms et SHA-256 vérifiés contre les manifestes. |
| `thesis/` | PDF envoyé à l'encadrant et manuscrit LaTeX v1.2 distinct. |
| `tests/` | Tests CPU bornés, avec source et checkpoint réels. |
| `docs/` | Correspondance code–méthode, protocole, provenance, limites et exécution. |

## Vérification rapide, sans réentraîner

Dans un environnement Python avec PyTorch CPU :

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_checkpoints.py
python scripts/quick_inference.py --n 20 --decode greedy --device cpu
python scripts/quick_inference.py --n 50 --decode greedy --device cpu
python scripts/quick_inference.py --n 100 --decode greedy --device cpu
```

Pour un **test limité du chemin d'échantillonnage** :

```bash
python scripts/quick_inference.py --n 20 --decode best-of-512 --sample-count 3
```

Le budget officiel de l'expérience est `--sample-count 512`. Une exécution avec 3 échantillons ne reproduit **aucun résultat chiffré du mémoire**; elle valide uniquement le chemin de code.

## Disponibilité des modèles — trois seeds vérifiées

Les trois poids acteurs sélectionnés sont inclus sans modification. Les empreintes SHA-256 ont été recalculées sur les fichiers binaires et comparées à `artifacts/aggregation/multiseed_selected_checkpoints.csv` :

| Seed | Fichier dans `models/seed_<seed>/` | SHA-256 vérifié |
|---|---|---|
| 20261013 | `train20_1013_17_14_step12999_act.pt` | `87610c76c1239c32d2bdba584196a00af85a45d008d8ba3fed181a31891ee5b4` |
| 20261014 | `train20_1025_04_27_step12999_act.pt` | `ed20ed5d06b3dbaaea666cd96b64ba43fb4f9cfa543927c6999061297cc47e4e` |
| 20261015 | `train20_1025_04_56_step12999_act.pt` | `0dbaaecc56baf013b411aae1bb3e69982682a49c3d314c9a7f0dedbb4b11284d` |

Contrôle indépendant sans PyTorch : `python scripts/verify_checkpoints.py`.

Pour inspecter **une instance** avec une autre seed, fournir explicitement `--checkpoint` :

```bash
python scripts/quick_inference.py --n 20 --decode greedy --checkpoint models/seed_20261014/train20_1025_04_27_step12999_act.pt
python scripts/quick_inference.py --n 100 --decode greedy --checkpoint models/seed_20261015/train20_1025_04_56_step12999_act.pt
```

La CLI utilise par défaut les poids de la seed 20261013. Les trois checkpoints se chargent avec les 20 mêmes clés et dimensions de tenseurs; ils possèdent des poids appris distincts. Les tests CPU ont confirmé des permutations valides pour n=20/50/100 et le chemin de sampling limité (3 candidats). **Cela ne reproduit pas les moyennes des 100 instances ni le benchmark Best-of-512** du mémoire.

Les répertoires Colab complets des deux entraînements récents, leurs checkpoints intermédiaires et certains manifestes de travail restent hors du dépôt; ne pas les confondre avec les trois checkpoints sélectionnés, désormais présents.

## Résultats : limites d'interprétation

À n=20, ACO donne la meilleure longueur moyenne; à n=50, la DRL dépasse NN simple mais non NN+2-opt ni ACO; à n=100, les deux décodages DRL sont moins performants que les trois baselines classiques. Les « écarts » sont relatifs au meilleur coût **observé** parmi neuf configurations, **pas** à l'optimum exact. L'écart-type inter-seeds ne doit pas être confondu avec celui calculé sur les instances. Le profil de temps v1.1 n'a pas été re-mesuré pour les deux nouvelles graines.


