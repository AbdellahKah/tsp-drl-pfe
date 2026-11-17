# Protocole expérimental v1.2

- **Apprentissage** : n=20, graines 20261013, 20261014, 20261015, initialisations indépendantes, budget identique 13 000 mises à jour, batch 512, dimensions embed=hidden=128, Adam à 0,001.
- **Sélection** : checkpoints `step1000`, `step5000`, `step10000`, `step12999`; coût moyen greedy sur 100 instances de validation n=20, seed 7302026. Dans les trois cas `step12999` est retenu. Indice 0–12999 signifie 13 000 updates.
- **Test** : 100 instances par taille n=20,50,100, toutes partagées entre les méthodes; seed base 8302026, effective `base+n`. Aucun réentraînement n=50/100.
- **Décodage** : Greedy et Best-of-512; uniquement ce dernier retient le coût minimal parmi 512 tournées échantillonnées.
- **Comparateurs** : NN démarrant à la ville 0; NN+2-opt avec maximum 100 passes; ACO 15 fourmis, 30 itérations, alpha=1, beta=3, rho=0.1, Q=1.
- **Statistique** : 3 seeds pour la DRL (moyenne des moyennes et SD inter-seeds), 100 instances pour les baselines (SD inter-instances). Bootstrap apparié 10 000 répétitions, IC percentile 95 %, seed 8802026, sans correction pour tests multiples; 54 comparaisons conditionnelles aux modèles.
- **Référence d'écart** : meilleur coût observé pour chaque instance parmi 3 seeds × 2 modes DRL + 3 baselines = 9 configurations; ce n'est pas l'optimum du TSP.
- **Temps** : chronométrage v1.1 réutilisé; 10 instances par taille, 1 warm-up, 5 répétitions; protocole d'exécution asymétrique, non comparaison stricte end-to-end.

## Longueur moyenne observée v1.2 (source CSV agrégé)

Voir `artifacts/aggregation/multiseed_across_seed_summary.csv` pour toutes les valeurs, colonnes exactes et définitions. Conclusions : ACO meilleur à n=20; DRL inférieure à 2-opt et ACO n=50; DRL inférieure aux trois méthodes n=100. Aucun résultat exact d'optimalité.

## Versions / environnements

La dernière évaluation seed 20261015 exécutée sous PyTorch 2.11.0+cu130 est **la version retenue**, celle sous PyTorch 2.14.1 est supplantée. Les évaluations rapportées sont sous Tesla T4/CUDA 13.0. Les tests locaux CPU ne reproduisent pas l'environnement d'origine.
