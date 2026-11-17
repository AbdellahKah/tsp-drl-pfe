# Provenance et état d'audit

## Chaîne de sources

1. `thesis/submitted_main.pdf` : mémoire de **106 pages** transmis à l'encadrant, référence primaire des affirmations qu'il a reçues.
2. `thesis/Memoire_PFE_multiseed_v1_2.tex` : source LaTeX v1.2 fournie distinctement, qui ne remplace pas automatiquement le PDF transmis.
3. `code_snapshot/` : huit modules du workspace historique; la copie de `train.py` est **déjà patchée**. La provenance de l'amont est déclarée par les manifestes et un fichier `git-commit.txt`.
4. `experiment_patch/training_patch.diff` : différence historique (ajout seed, checkpoint périodique, désactivation arrêt anticipé) ; `patch --dry-run -R` passe sur le `train.py` fourni.
5. `notebooks/` : cellules source des protocoles seed-13, entraînement multiseed et évaluation. Version PyTorch 2.11 corrigée de seed-15 privilégiée.
6. `scripts/aggregate_multiseed_results_TORCH2110_FIXED.py` : agrégation finale v1.2, conservée sans retouche.
7. `artifacts/aggregation/` : CSV et manifestes finaux; les données agrégées ont été fournies, non recalculées.

## Contrôles documentés lors de l'assemblage

- Bundle : 111/111 éléments décrits dans la liste SHA-256 vérifiés.
- Les **trois** checkpoints acteurs sélectionnés : SHA-256 vérifiés contre `artifacts/aggregation/multiseed_selected_checkpoints.csv` (seed-13 : `87610c76c1239c32d2bdba584196a00af85a45d008d8ba3fed181a31891ee5b4` ; seed-14 : `ed20ed5d06b3dbaaea666cd96b64ba43fb4f9cfa543927c6999061297cc47e4e` ; seed-15 : `0dbaaecc56baf013b411aae1bb3e69982682a49c3d314c9a7f0dedbb4b11284d`). Les deux derniers fichiers binaires proviennent de l'archive ajoutée `PFE_Missing_Selected_Checkpoints.zip` ; les originaux ont été conservés octet pour octet.
- Les trois fichiers contiennent des state_dicts PyTorch chargés avec `weights_only=True` sur CPU : 20 tenseurs chacun, formes et clés identiques, poids distincts. Greedy validé à 20/50/100 villes pour les trois; sampling contrôlé avec seulement 3 tirages par instance, sans réexécuter Best-of-512 officiel.
- 7 notebooks top-level extraits : JSON et compilation des cellules Python vérifiés; zéro sortie exécutée enregistrée. Une copie identique seed-14 était répétée dans deux emplacements.
- Licence du dépôt amont sur GitHub (`master`) : MIT, copyright 2020 Rintarooo. **Vérification Git locale du commit exact non effectuée**, la résolution de `github.com` étant indisponible dans ce runtime.

## Fichiers / éléments toujours manquants

- **Aucun** des trois checkpoints acteurs **sélectionnés** n'est manquant. Les checkpoints intermédiaires non sélectionnés ne sont pas tous conservés.
- Dossiers originaux de travail de ces deux runs : manifests d'entraînement complets et checkpoints intermédiaires non intégrés au bundle.
- Accès à un remote GitHub : aucun dépôt créé/poussé; ne pas conclure que l'encadrant y a accès.

## Problèmes techniques à traiter sans altérer les résultats

- `code_snapshot/config.py` définit `CUDA_VISIBLE_DEVICE` (sans S), au lieu de `CUDA_VISIBLE_DEVICES`. Son rôle effectif sur le GPU est donc suspect; aucune correction dans le code historique.
- Dans `code_snapshot/actor.py`, le tenseur d'attention est paramétré avec `cfg.embed` tandis que les états LSTM ont `cfg.hidden`; le modèle suppose en pratique leurs tailles compatibles (128/128 ici).
- Dans `code_snapshot/train.py`, `torch.backends.cudnn.benchmark = True` est exécuté à l'import; la section de lancement force ensuite `benchmark = False` et `deterministic = True`. Cette séquence mérite un contrôle lors d'une future reproduction GPU.
- `search.py` comporte un chemin « active_search » qui ne correspond pas aux décodages publiés; ne pas le confondre avec l'évaluation finale.
- Les notebooks Colab présupposent des chemins Google Drive et versions CUDA définis : ce sont des records d'exécution, non des scripts portables.
- Les chemins des CSV finaux pointent vers des résultats antérieurs; leur présence ne suffit pas à exécuter à l'identique tous les notebooks sans les poids manquants et les dossiers Drive originaux.

## Vérifications supplémentaires recommandées

Comparer les huit sources à l'objet Git upstream exact lorsque l'accès réseau Git sera rétabli; vérifier les transitions notebook→scripts et les figures/chiffres du mémoire section par section avant toute diffusion publique. Ne pas attribuer une licence aux ajouts personnels sans décision de l'auteur.
