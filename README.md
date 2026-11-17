# TSP-DRL — Pointer Network pour le voyageur de commerce

Ce dépôt regroupe le code et les résultats de mon projet de fin d'études sur la résolution du problème du voyageur de commerce (TSP) euclidien par apprentissage par renforcement profond.

L'approche utilise un **Pointer Network** avec encodeur-décodeur LSTM, entraîné par **REINFORCE** avec une baseline apprise pour estimer le coût des tournées. Les expériences portent sur des instances de 20, 50 et 100 villes, avec une comparaison aux heuristiques Nearest Neighbour, 2-opt et Ant Colony Optimization (ACO).

Projet réalisé dans le cadre du Master Mathématiques Appliquées à la Faculté des Sciences et Techniques de Settat — Université Hassan Ier.

## Installation

~~~bash
git clone https://github.com/AbdellahKah/tsp-drl-pfe.git
cd tsp-drl-pfe

python -m venv .venv
~~~

Activation de l'environnement :

- Linux / macOS : `source .venv/bin/activate`
- Windows (PowerShell) : `.\.venv\Scripts\Activate.ps1`

~~~bash
python -m pip install -r requirements.txt
~~~

Les dépendances sont prévues pour les vérifications locales. Les expériences originales ont été effectuées sur Google Colab avec un GPU Tesla T4.

## Utilisation

Les checkpoints des trois entraînements sont disponibles dans `models/`. Le script ci-dessous permet de charger un modèle, de générer une instance aléatoire et d'afficher la tournée obtenue ainsi que sa longueur.

~~~bash
# TSP20, modèle entraîné avec la graine 20261013
python scripts/quick_inference.py --n 20 --decode greedy --device cpu

# TSP50, modèle entraîné avec la graine 20261014
python scripts/quick_inference.py --n 50 --decode greedy --device cpu --checkpoint models/seed_20261014/train20_1025_04_27_step12999_act.pt

# TSP100, modèle entraîné avec la graine 20261015
python scripts/quick_inference.py --n 100 --decode greedy --device cpu --checkpoint models/seed_20261015/train20_1025_04_56_step12999_act.pt
~~~

Deux modes de décodage sont disponibles : `greedy` (une tournée construite de façon déterministe par le modèle) et `best-of-512` (512 tournées échantillonnées, dont on retient la plus courte).

~~~bash
python scripts/quick_inference.py --n 20 --decode best-of-512 --sample-count 512 --device cpu
~~~

Pour vérifier les empreintes des checkpoints et lancer les tests :

~~~bash
python scripts/verify_checkpoints.py
python -m pytest -q
~~~

Ces commandes effectuent des contrôles locaux sur les modèles et les résultats sauvegardés. Elles ne relancent pas l'entraînement complet.

## Expériences

Les trois modèles ont été entraînés **indépendamment** à partir de graines différentes, uniquement sur des instances de **20 villes** :

| Paramètre | Valeur |
|---|---|
| Graines d'entraînement | 20261013, 20261014, 20261015 |
| Taille d'entraînement | 20 villes |
| Batch size | 512 |
| Dimensions embedding / hidden | 128 / 128 |
| Optimiseur | Adam, learning rate initial 0,001 |
| Budget d'entraînement | 13 000 mises à jour par modèle |
| Sélection du checkpoint | Validation Greedy sur 100 instances TSP20 |
| Tests | 100 instances pour chacune des tailles 20, 50 et 100 |
| Décodage | Greedy et Best-of-512 |

Les tailles 50 et 100 servent à évaluer la généralisation des modèles entraînés à 20 villes.

### Reproduire les entraînements et les évaluations

Les notebooks utilisés lors des expériences sont conservés dans `notebooks/`. Pour relancer l'expérience sur Colab, activer un runtime GPU, monter Google Drive et les exécuter dans l'ordre suivant :

1. [Entraînement 20261013](notebooks/seed_20261013_training_and_pilot.ipynb)
2. [Évaluation 20261013](notebooks/seed_20261013_evaluation_v1_1.ipynb)
3. [Entraînements 20261014 et 20261015](notebooks/multiseed_training_v1_2.ipynb)
4. [Évaluation 20261014](notebooks/seed_20261014_evaluation_v1_2.ipynb)
5. [Évaluation 20261015 — version corrigée](notebooks/seed_20261015_evaluation_corrected_torch2110_v1_2.ipynb)

Les notebooks utilisent initialement `/content/drive/MyDrive/PFE_TSP_DRL`. Prévoir un dossier Drive distinct pour une nouvelle exécution et adapter `PROJECT_ROOT` si nécessaire. Pour garder la même version du code, utiliser le commit amont `33636039c37da190cadb5bc136162c1e3f4daf46` : le premier notebook doit être épinglé sur ce commit après le clonage. Le notebook multiseed contient également une vérification d'un ancien nom horodaté de checkpoint 20261013, à adapter au fichier produit par un nouvel entraînement.

La sélection des modèles repose sur les checkpoints `step1000`, `step5000`, `step10000` et `step12999`, évalués sur 100 instances de validation TSP20 (graine `7302026`). Les tests finaux utilisent des instances communes aux trois modèles et aux méthodes de comparaison (graine de base `8302026`, avec `base + n` pour chaque taille).

Une fois les évaluations terminées, l'agrégation se lance avec le dossier Drive contenant les sorties complètes :

~~~bash
python scripts/aggregate_multiseed_results_TORCH2110_FIXED.py --project-root "/chemin/vers/PFE_TSP_DRL"
~~~

Les évaluations finales archivées ont été réalisées avec **PyTorch 2.11.0+cu130, CUDA 13.0, Tesla T4**. Les checkpoints intermédiaires et tous les historiques Colab des runs originaux ne sont pas inclus : pour refaire la sélection depuis le début, il faut réentraîner les modèles ou retrouver ces fichiers.

Le protocole détaillé, les paramètres ACO, les graines de test et le bootstrap sont décrits dans [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) et [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Résultats

Longueur moyenne des tournées obtenues sur les 100 instances de chaque taille (plus faible = meilleur) :

| Méthode | TSP20 | TSP50 | TSP100 |
|---|---:|---:|---:|
| DRL Greedy | 4,0355 | 6,6403 | 10,4373 |
| DRL Best-of-512 | 3,9584 | 6,3294 | 9,8946 |
| Nearest Neighbour | 4,4961 | 6,9443 | 9,7388 |
| NN + 2-opt | 3,9424 | **5,9712** | **8,2705** |
| ACO | **3,8780** | 6,1568 | 9,1606 |

Les valeurs DRL correspondent à la moyenne des trois entraînements indépendants. Les fichiers complets, y compris les résultats par graine et les comparaisons bootstrap appariées, se trouvent dans [artifacts/aggregation/](artifacts/aggregation/).

## Organisation du dépôt

~~~text
code_snapshot/       Modèle original, environnement et entraînement
experiment_patch/    Modifications du script d'entraînement
notebooks/           Entraînements et évaluations Colab
models/              Checkpoints acteurs sélectionnés
src/tsp_drl/         Heuristiques de comparaison
scripts/             Inférence, vérification et agrégation
artifacts/           Résultats CSV et manifestes
configs/             Paramètres expérimentaux
tests/               Tests de cohérence et d'inférence
docs/                Détails de l'implémentation et du protocole
thesis/              Mémoire PDF et source LaTeX
~~~

Pour la correspondance entre les équations et le code : [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md). Pour la provenance des sources et des modifications : [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Références

Le modèle est basé sur [Rintarooo/TSP_DRL_PtrNet](https://github.com/Rintarooo/TSP_DRL_PtrNet), commit `33636039c37da190cadb5bc136162c1e3f4daf46`. Le code amont est distribué sous licence MIT, reproduite dans [THIRD_PARTY_LICENSES/Rintarooo-MIT.txt](THIRD_PARTY_LICENSES/Rintarooo-MIT.txt).

Le mémoire complet est disponible dans [thesis/submitted_main.pdf](thesis/submitted_main.pdf).
