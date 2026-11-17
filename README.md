# Guide de reproduction du PFE — à l'attention de l'encadrant

**Mémoire :** *Apprentissage par Renforcement Profond pour l’Optimisation Combinatoire : Étude Algorithmique du Problème du Voyageur de Commerce*  
**Auteur :** Abdellah Kahlaoui — Faculté des Sciences et Techniques de Settat, Université Hassan Ier.

Ce dépôt fournit le **code réellement utilisé**, les **trois checkpoints acteurs sélectionnés**, les notebooks et les **résultats archivés**. Ce document explique comment contrôler les livrables, exécuter les modèles et, avec un GPU, reprendre le protocole d'apprentissage. L'application interactive pour le jury n'est pas encore développée.

**Parcours recommandé pour une première revue : étapes 1 à 3 (CPU, sans réentraînement).** La répétition des expériences complètes (étape 4) requiert un environnement Colab/GPU adapté, de nouveaux fichiers d'entraînement et du temps de calcul.

## 1. Cloner et préparer le dépôt

**Prérequis :** autorisation d'accès au dépôt privé, Git, Python compatible avec PyTorch, terminal.

~~~bash
git clone https://github.com/AbdellahKah/tsp-drl-pfe.git
cd tsp-drl-pfe
python -m venv .venv
~~~

Activer l'environnement virtuel :
- **Windows PowerShell :** `.\.venv\Scripts\Activate.ps1`
- **Linux/macOS :** `source .venv/bin/activate`

Puis lancer les commandes suivantes **à la racine du dépôt** :

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_checkpoints.py
python -m pytest -q
~~~

**Résultat attendu :** le script de vérification affiche `OK` pour les graines 20261013, 20261014 et 20261015. Les tests doivent réussir (23 tests avaient réussi lors de l'audit local initial). Le nombre de tests peut évoluer. Si nécessaire, remplacer `python` par `python3` ou `py` selon le système. Le fichier `requirements.txt` ne verrouille **pas** l'environnement exact des évaluations GPU de 2026.

## 2. Exécuter les modèles déjà entraînés (CPU)

Les poids d'origine sont sous `models/seed_<graine>/`. Le script utilise le vrai réseau acteur `PtrNet1` (`code_snapshot/actor.py`). Il doit retourner une **permutation valide** des villes et calculer un **circuit fermé**, avec retour au point initial.

~~~bash
# Graine 20261013 : TSP20, Greedy
python scripts/quick_inference.py --n 20 --decode greedy --device cpu

# Graine 20261014 : TSP50, Greedy
python scripts/quick_inference.py --n 50 --decode greedy --device cpu --checkpoint models/seed_20261014/train20_1025_04_27_step12999_act.pt

# Graine 20261015 : TSP100, Greedy
python scripts/quick_inference.py --n 100 --decode greedy --device cpu --checkpoint models/seed_20261015/train20_1025_04_56_step12999_act.pt

# Contrôle léger de l'échantillonnage : seulement 3 candidats
python scripts/quick_inference.py --n 20 --decode best-of-512 --sample-count 3 --device cpu
~~~

**À vérifier :** chaque ville est visitée une seule fois (indices de 0 à n−1), et la longueur calculée est positive et finie. Le script affiche la tournée et son coût.

**Attention :** le test à 3 candidats vérifie seulement l'exécution du chemin de sampling. Le mode expérimental **Best-of-512** échantillonne **512** tournées et conserve la plus courte. Pour une seule instance :

~~~bash
python scripts/quick_inference.py --n 20 --decode best-of-512 --sample-count 512 --device cpu
~~~

Cette invocation peut être plus lente et ne reproduit **pas** le benchmark final à 100 instances par taille.

## 3. Contrôler les résultats archivés sans relancer les réseaux

~~~bash
python -m pytest -q tests/test_archived_results.py
~~~

Ce test reconstitue les coordonnées des instances communes selon les graines archivées, vérifie **2 700 permutations** et leurs longueurs euclidiennes, et contrôle les statistiques enregistrées. Il ne refait ni l'entraînement ni les inférences du benchmark officiel.

| Fichier | Rôle |
|---|---|
| `artifacts/aggregation/multiseed_combined_per_instance.csv` | Tournées et coûts par instance |
| `artifacts/aggregation/multiseed_per_seed_summary.csv` | Résultats séparés par graine d'entraînement |
| `artifacts/aggregation/multiseed_across_seed_summary.csv` | Moyennes des trois graines et dispersion inter-graines |
| `artifacts/aggregation/multiseed_paired_bootstrap_by_seed.csv` | Intervalles bootstrap appariés à 95 % |
| `artifacts/aggregation/multiseed_selected_checkpoints.csv` | Fichiers sélectionnés et empreintes SHA-256 |
| `artifacts/aggregation/multiseed_aggregation_manifest.json` | Provenance de l'agrégation |

**Repères issus des résultats finaux v1.2** (longueur moyenne, plus faible = meilleur) :

| Nombre de villes | DRL Greedy | DRL Best-of-512 | NN+2-opt | ACO |
|---:|---:|---:|---:|---:|
| 20 | 4,035502 | 3,958365 | 3,942381 | **3,878018** |
| 50 | 6,640337 | 6,329445 | **5,971182** | 6,156831 |
| 100 | 10,437335 | 9,894557 | **8,270525** | 9,160622 |

Les « écarts » archivés prennent pour référence le **meilleur coût observé** parmi neuf configurations (trois modèles × deux décodages + trois baselines) et **non** un optimum exact du TSP. L'écart-type *entre graines d'entraînement* n'est pas l'écart-type *entre instances de test*. À 100 villes, les trois baselines classiques dépassent les deux modes DRL ; aucune supériorité générale n'est revendiquée.

## 4. Répéter le protocole expérimental complet sur GPU

### 4.1 Préparation

Le protocole original comprend trois runs **indépendants** : graines **20261013, 20261014 et 20261015** ; chacun est entraîné uniquement sur **20 villes** avec batch **512**, dimensions embedding/hidden **128/128**, Adam au taux initial **0,001** et **13 000 mises à jour**. Les tailles 50 et 100 sont des tests de **généralisation**, sans réentraînement à ces tailles.

Prévoir Google Colab avec GPU, Google Drive et suffisamment de ressources pour trois entraînements, les validations, les décodages Best-of-512 et les statistiques. Les évaluations finales retenues utilisaient **PyTorch 2.11.0+cu130, CUDA 13.0 et Tesla T4** ; Python 3.13.15 pour le premier entraînement et 3.13.16 pour les deux suivants. La dernière évaluation de 20261015 s'arrête volontairement si PyTorch/CUDA/GPU diffèrent : ne pas supprimer cette vérification et qualifier l'essai d'identique.

Utiliser **une nouvelle racine de projet**, car les notebooks écrivent initialement dans `/content/drive/MyDrive/PFE_TSP_DRL`. Si le nom est changé, modifier `PROJECT_ROOT` dans chacun des notebooks. Ne jamais effacer ni écraser les données historiques.

### 4.2 Ordre d'exécution

Dans Colab, sélectionner un runtime GPU, monter Drive et exécuter les cellules **de haut en bas** :

| Ordre | Notebook ou script | Fonction |
|---:|---|---|
| 1 | `notebooks/seed_20261013_training_and_pilot.ipynb` | Entraînement 20261013 |
| 2 | `notebooks/seed_20261013_evaluation_v1_1.ipynb` | Sélection, tests finaux et profil de temps initial |
| 3 | `notebooks/multiseed_training_v1_2.ipynb` | Nouveaux entraînements indépendants 20261014 et 20261015 |
| 4 | `notebooks/seed_20261014_evaluation_v1_2.ipynb` | Validation et évaluation 20261014 |
| 5 | `notebooks/seed_20261015_evaluation_corrected_torch2110_v1_2.ipynb` | Validation et évaluation **corrigée** 20261015 |
| 6 | `scripts/aggregate_multiseed_results_TORCH2110_FIXED.py` | Agrégation et vérification croisée |

**Ne pas exécuter** les anciennes versions présentes dans `notebooks/superseded/` et `scripts/superseded/` pour reproduire les chiffres finaux.

### 4.3 Deux ajustements de provenance pour un nouveau run

**(a) Forcer le commit amont du premier notebook.** `seed_20261013_training_and_pilot.ipynb` clone par défaut la branche courante du dépôt amont, ce qui peut changer le code. Dans **une copie** de ce notebook, insérer après le clonage et **avant** la lecture de `COMMIT_SHA` et l'écriture du manifest :

~~~python
PINNED_COMMIT = "33636039c37da190cadb5bc136162c1e3f4daf46"
subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", PINNED_COMMIT], check=True)
~~~

Vérifier ensuite `COMMIT_SHA == PINNED_COMMIT`. Conserver le notebook d'archive intact et documenter la modification dans le nouveau run.

**(b) Vérifier le checkpoint 20261013 référencé par le notebook multiseed.** Une cellule de `multiseed_training_v1_2.ipynb` attend le nom horodaté historique `train20_1013_17_14_step12999_act.pt`. Un nouvel entraînement peut produire un autre nom. Adapter **dans une copie** uniquement la vérification du chemin pour cibler l'**unique** checkpoint final `*_step12999_act.pt` dans le dossier `runs/clean_restart_seed20261013_v1/full_train_n20/Pt/`, après contrôle de son manifeste. Ne **pas** renommer un modèle nouveau pour le présenter comme l'original.

Ces ajustements sont d'orchestration : ne pas modifier le modèle, les objectifs de perte ni les paramètres expérimentaux.

### 4.4 Contrôles du protocole

1. **Entraînement :** vérifier `manifest_training.json`, les logs et **13 000 mises à jour par graine**, sans transfert de poids entre runs. Le suffixe `step12999` correspond à la 13 000e mise à jour (index démarrant à zéro).
2. **Sélection :** comparer indépendamment les snapshots `step1000`, `step5000`, `step10000` et `step12999` sur **100 instances de validation n=20**, seed **7302026**, en **Greedy**. Conserver celui qui minimise la longueur moyenne et enregistrer `checkpoint_selection.json`. Dans l'étude historique, `step12999` a été retenu pour les trois graines ; un nouveau run peut avoir un autre résultat.
3. **Tests finaux :** 100 instances **communes** à chaque taille `n=20,50,100`, graine de base **8302026**, donc graines effectives **8302046**, **8302076** et **8302126**. Méthodes : DRL Greedy, DRL Best-of-512, PPV/NN départ ville 0, PPV + 2-opt (maximum 100 passes) et ACO (15 fourmis, 30 itérations, α=1, β=3, ρ=0,1, Q=1).
4. **Statistiques :** **10 000** tirages bootstrap **appariés**, graine **8802026**, IC percentile **95 %**, sans correction des comparaisons multiples. Moyenne finale de la DRL : moyenne des **trois moyennes par graine** ; dispersion : écart-type empirique **entre graines**. Ne pas traiter les 300 couples graine-instance comme 300 instances indépendantes.
5. **Profil de temps :** la v1.2 **réutilise** les temps v1.1 ; les temps n'ont pas été remesurés pour les graines 20261014/15. Ils ne constituent pas un benchmark strictement comparable de bout en bout.

### 4.5 Agrégation des nouveaux résultats

Après la réussite des cinq notebooks, avec les **dossiers Drive complets** (notamment les `.npz` de coordonnées, les `final_test_per_instance.csv` et les manifestes des trois évaluations), lancer depuis la racine du dépôt sur une machine pouvant accéder à ces dossiers :

~~~bash
python scripts/aggregate_multiseed_results_TORCH2110_FIXED.py --project-root "/chemin/vers/PFE_TSP_DRL"
~~~

Ou, après avoir téléversé le script dans `/content/` sur Colab :

~~~bash
python /content/aggregate_multiseed_results_TORCH2110_FIXED.py --project-root /content/drive/MyDrive/PFE_TSP_DRL
~~~

Le script attend les répertoires d'évaluation `final_evaluation_v1_1`, `final_evaluation_seed20261014_v1_2` et `final_evaluation_seed20261015_torch2110_v1_2`. Il contrôle les coordonnées communes, les baselines et l'environnement. Il **refuse par défaut d'écraser** une sortie existante : utiliser une nouvelle racine ou le paramètre `--output-folder` avec un nom inédit.

Inspecter ensuite les CSV générés, notamment `multiseed_across_seed_summary.csv`, `multiseed_paired_bootstrap_by_seed.csv` et `multiseed_aggregation_manifest.json` ; documenter les différences par rapport aux résultats archivés sans remplacer l'archive historique.

## 5. Ce qui est, et n'est pas, reproductible directement

**Présent :** les modules d'origine, le patch d'entraînement, les notebooks, les trois **checkpoints acteurs finaux** vérifiés par SHA-256, les CSV et manifestes des résultats, le mémoire et les documents d'audit.

**Partiellement absent :** journaux Colab complets et checkpoints **intermédiaires** originaux (`step1000/5000/10000`), certains dossiers Drive et tableaux `.npz`. Les trois modèles finaux ne suffisent pas à **refaire rétrospectivement la sélection historique** ; il faut les fichiers intermédiaires d'origine ou de **nouveaux** entraînements. Les contrôles CPU ne remplacent pas une évaluation GPU officielle, et de nouvelles graines fixées ne garantissent pas des poids binaires identiques sur un autre environnement.

Le modèle est une **Pointer Network entraînée par REINFORCE avec baseline de coût apprise** : ce n'est **pas** un Actor–Critic à mises à jour TD par étape ; aucun GNN n'est employé.

## 6. Où lire chaque élément ?

| Question | Chemin |
|---|---|
| Construction de la tournée par le réseau | `code_snapshot/actor.py` (`PtrNet1`) |
| Baseline prédite | `code_snapshot/critic.py` (`PtrNet2`) |
| Coût euclidien fermé | `code_snapshot/env.py` |
| Perte REINFORCE et mise à jour MSE | `code_snapshot/train.py` |
| Diff avec le code amont | `experiment_patch/training_patch.diff` ; `docs/PROVENANCE.md` |
| NN, 2-opt, ACO | `src/tsp_drl/algorithms/reference_baselines.py` |
| Protocoles et chiffres | `configs/experiment_v1_2.json` ; `docs/EXPERIMENTS.md` |
| Résultats archivés | `artifacts/aggregation/` |
| Audit des limites et vérifications | `docs/AUDIT_REPORT.md` ; `docs/REPRODUCIBILITY.md` |
| Mémoire PDF transmis à l'encadrant | `thesis/submitted_main.pdf` |
| Version éditable LaTeX distincte | `thesis/Memoire_PFE_multiseed_v1_2.tex` |
| Présentation générale du projet (ancien README) | **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** |

**Code amont :** [Rintarooo/TSP_DRL_PtrNet](https://github.com/Rintarooo/TSP_DRL_PtrNet), révision `33636039c37da190cadb5bc136162c1e3f4daf46`, licence MIT conservée dans `THIRD_PARTY_LICENSES/Rintarooo-MIT.txt`.

**Pour toute reproduction :** conserver versions logicielles et GPU, paramètres, graines, logs, manifestes, commandes, fichiers de résultats et empreintes SHA-256. Documenter toute modification ou divergence ; ne pas modifier silencieusement les observations historiques.
