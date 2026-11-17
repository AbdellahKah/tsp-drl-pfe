# Audit de transfert et périmètre de preuve

## Matrice des sources faisant autorité

| Domaine | Source de référence présente | Ce qu'elle établit | Réserve |
|---|---|---|---|
| Mémoire vu par l'encadrant | `thesis/submitted_main.pdf` | Version PDF de 106 pages effectivement communiquée selon le handoff | Son contenu ne doit pas être remplacé par la source TeX |
| Manuscrit modifiable | `thesis/Memoire_PFE_multiseed_v1_2.tex` | Source multiseed v1.2 | Ne prouve pas à elle seule ce qui a été envoyé |
| Architecture initiale | `code_snapshot/{actor,critic,env,data,config,train,test,search}.py` | Fichiers du workspace historique disponibles dans la livraison | `train.py` déjà modifié; identité parfaite avec Git à commit fixe non indépendante |
| Écart expérimental | `experiment_patch/training_patch.diff` | Changements déclarés pour le budget fixe, checkpoints et seeds | Patch inversable; pas de diff direct Git amont faute d'accès remote |
| Exécution seed 20261013 | `notebooks/seed_20261013_training_and_pilot.ipynb`, manifests et `models/` | Protocole, fichier acteur sélectionné, hash vérifié | Entraînement intégral non reproduit |
| Exécution seeds 20261014/15 | `notebooks/multiseed_training_v1_2.ipynb`, `models/seed_20261014/`, `models/seed_20261015/` | Orchestration Colab, paramètres et deux poids d’origine ajoutés ensuite, SHA-256 vérifiés | Journaux Drive complets non fournis |
| Validation et comparaison | `notebooks/seed_20261013_evaluation_v1_1.ipynb`, `notebooks/seed_20261014_evaluation_v1_2.ipynb`, `notebooks/seed_20261015_evaluation_corrected_torch2110_v1_2.ipynb` | Décodage, baseline, tests et bootstrap prévus | Dossiers Colab d'origine non disponibles intégralement |
| Résultats rapportés | `artifacts/aggregation/`, `artifacts/seed_*/final/` | Données multiseed finalisées, journaux de sélection, métadonnées de l'évaluation | Présence des résultats ≠ présence des trois poids |
| Statistiques finales | `scripts/aggregate_multiseed_results_TORCH2110_FIXED.py` | Dernière logique d'agrégation fournie | Ancien script sous `scripts/superseded` non retenu |

## Origine des modules et logique d'usage

`actor.PtrNet1` → coût `env.Env_tsp.stack_l_fast` → `train.py` : REINFORCE (gradient de la log-vraisemblance de la tournée, ajusté par avantage). `critic.PtrNet2` prédit le coût total (MSE contre le coût observé). `data.Generator` produit les instances d'entraînement; `config.Config` enregistre les paramètres. Les notebooks d'évaluation rechargent `actor.PtrNet1` et comparent Greedy, Best-of-512, NN, NN+2-opt et ACO. `test.py` / `search.py` du code initial n'implémentent pas directement cette dernière comparaison.

## Constats et limites techniques

1. **Confirmé** : la copie historique de `train.py` possède déjà le patch. Le diff historique peut être renversé sans conflit (`patch --dry-run -R`).
2. **Confirmé** : le `config.py` conservé écrit `CUDA_VISIBLE_DEVICE` au lieu de la variable conventionnelle `CUDA_VISIBLE_DEVICES`. Ne pas modifier cette copie avant une reproduction contrôlée.
3. **Confirmé** : les deux décodages du manuscrit sont décrits par les notebooks; `active_search` du dépôt initial n'est pas cette évaluation.
4. **Confirmé** : les dimensions 128/128 sont compatibles avec les paramètres d'attention; des dimensions différentes peuvent ne pas fonctionner sans adaptation.
5. **Non contrôlé en totalité** : un audit exhaustif, ligne par ligne, des 106 pages du PDF par rapport au code. Les passages vérifiés concernant REINFORCE, le LSTM, les trois seeds, les comparateurs, les tailles 20/50/100 et la distinction « meilleur observé » / « optimum exact » sont cohérents avec le protocole.
6. **À récupérer** : binaires sélectionnés des seeds 20261014/15, manifests d'entraînement complets et éventuels fichiers d'instances `.npz` originaux si une relecture binaire des artefacts est nécessaire.
7. **Non vérifié** : identité des huit modules avec les objets du commit Git upstream, car le clonage direct de GitHub est indisponible dans l'environnement d'assemblage.

## Bilan de contrôle local (2026-11-17)

- Intégrité du bundle : 111/111 empreintes enregistrées dans `MANIFEST_SHA256.txt` vérifiées.
- Sept notebooks source : JSON valide, toutes les cellules de code compilables, aucune sortie d'exécution incorporée.
- SHA-256 des trois checkpoints sélectionnés (20261013/14/15) : conformes aux enregistrements du manifeste. Les deux nouveaux fichiers ont été ajoutés sans modification binaire. Structures des `state_dict` vérifiées (20 tenseurs et mêmes formes).
- Vérification des 2 700 tournées archivées : permutations valides et longueurs euclidiennes recalculées depuis les graines de génération exactes, tolérance `1e-6`, sans discordance.
- Moyennes et écarts-types DRL inter-seeds : accord aux valeurs des CSV agrégés.
- Modules sources copiés octet pour octet depuis le bundle; fonctions NN, 2-opt, ACO extraites textuellement de la cellule 2 du notebook d'évaluation corrigé et testées contre l'exécution des fonctions de la cellule d'origine.
- Tests CPU et smoke inference : **23 tests réussis** (PyTorch 2.10.0+cpu local), Greedy n=20/50/100 pour chaque seed, sampling limité à 3 tirages. Aucun entraînement de 13 000 pas ni sweep officiel de 512 échantillons exécuté.

## Plans / autorisations

Le dépôt local d'inspection est prêt pour contrôle humain. Avant de créer un dépôt GitHub **privé**, contrôler l'identité Git amont, la politique de licence du code nouveau, les données privées, et décider de la diffusion des checkpoints. La publication publique et l'invitation de collaborateurs demandent une validation explicite de l'auteur. L'application graphique reste hors du périmètre.
