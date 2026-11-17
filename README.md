# TSP–DRL — dépôt d'inspection du projet de fin d'études

**Mémoire :** *Apprentissage par Renforcement Profond pour l’Optimisation Combinatoire : Étude Algorithmique du Problème du Voyageur de Commerce*  
**Auteur du mémoire :** Abdellah Kahlaoui  
**Établissement :** Faculté des Sciences et Techniques de Settat, Université Hassan Ier.  
**Objet de ce dépôt :** permettre à l'encadrant d'inspecter le code, les protocoles et les preuves expérimentales. Il ne s'agit **pas** de la future application de démonstration.

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
| `models/` | Checkpoint sélectionné de la seed 20261013, SHA-256 vérifié. |
| `thesis/` | PDF envoyé à l'encadrant et manuscrit LaTeX v1.2 distinct. |
| `tests/` | Tests CPU bornés, avec source et checkpoint réels. |
| `docs/` | Correspondance code–méthode, protocole, provenance, limites et exécution. |

## Vérification rapide, sans réentraîner

Dans un environnement Python avec PyTorch CPU :

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/quick_inference.py --n 20 --decode greedy --device cpu
python scripts/quick_inference.py --n 50 --decode greedy --device cpu
python scripts/quick_inference.py --n 100 --decode greedy --device cpu
```

Pour un **test limité du chemin d'échantillonnage** :

```bash
python scripts/quick_inference.py --n 20 --decode best-of-512 --sample-count 3
```

Le budget officiel de l'expérience est `--sample-count 512`. Une exécution avec 3 échantillons ne reproduit **aucun résultat chiffré du mémoire**; elle valide uniquement le chemin de code.

## Disponibilité des modèles

Le fichier `models/seed_20261013/train20_1013_17_14_step12999_act.pt` a le SHA-256 :

`87610c76c1239c32d2bdba584196a00af85a45d008d8ba3fed181a31891ee5b4`.

Les poids sélectionnés des seeds **20261014** et **20261015** sont **absents** de la livraison. Leurs noms, hashes et métriques sont documentés dans les manifestes; il serait faux de dire que les trois modèles sont exécutables localement. Il faut récupérer les poids d'origine pour refaire leurs inférences. 

## Résultats : limites d'interprétation

À n=20, ACO donne la meilleure longueur moyenne; à n=50, la DRL dépasse NN simple mais non NN+2-opt ni ACO; à n=100, les deux décodages DRL sont moins performants que les trois baselines classiques. Les « écarts » sont relatifs au meilleur coût **observé** parmi neuf configurations, **pas** à l'optimum exact. L'écart-type inter-seeds ne doit pas être confondu avec celui calculé sur les instances. Le profil de temps v1.1 n'a pas été re-mesuré pour les deux nouvelles graines.

**Attribution :** la base du modèle provient de [Rintarooo/TSP_DRL_PtrNet](https://github.com/Rintarooo/TSP_DRL_PtrNet), commit déclaré `33636039c37da190cadb5bc136162c1e3f4daf46`. Licence amont MIT; lire `NOTICE.md` et `THIRD_PARTY_LICENSES/Rintarooo-MIT.txt` avant publication. **Dépôt distant non créé; diffusion privée préconisée initialement.**
