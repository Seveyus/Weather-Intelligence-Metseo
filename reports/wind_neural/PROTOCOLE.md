# Protocole neuronal enregistré avant entraînement

## Objectif et référence

Le modèle principal doit être un réseau de neurones récurrent. La baseline à arbres reste une référence figée : commit `e2376b6`, métriques et prédictions sous `reports/wind_advanced/`. Aucun score des arbres ni aucune de leurs prédictions ne sera utilisé comme entrée ou cible du réseau neuronal.

Objectif expérimental : améliorer conjointement MAE et RMSE aux échéances +24/+48/+72 h. Une amélioration n'est pas garantie. Les expériences négatives seront conservées.

## Architectures et entraînement

- BiGRU : une couche, 48 unités par direction, résumé par les états finaux.
- BiLSTM : même dimension et même décodeur, pour comparer la cellule récurrente.
- BiGRU avec attention temporelle conditionnée par l'échéance et les prévisions météo.
- Contrôle d'ablation : BiGRU avec fenêtre temporelle neutralisée, pour mesurer l'apport de l'encodeur séquentiel au-delà des variables historiques déjà calculées dans le contexte.

Chaque candidat utilise les graines **11 et 23**. Son résultat de validation est la moyenne des deux réseaux. Le contrôle d'ablation est rapporté séparément et n'est pas éligible au choix du réseau récurrent principal. Aucun assemblage avec les arbres n'est utilisé.

Historique de 72 h ; sorties directes et non négatives à +24/+48/+72 h ; correction neuronale de la moyenne des bulletins météo ; couches denses partagées et codage de l'échéance. Les transformations sont apprises exclusivement sur l'apprentissage. Dropout explicite 0,1, AdamW, taux initial 0,001, pénalisation 0,001, gradient borné à 1, lots de 256, maximum **32 époques**, patience **6** sur validation. Perte : erreur quadratique normalisée par 5 km/h et faible composante absolue. Les meilleurs poids sont conservés selon le critère de validation.

Si une contrainte matérielle impose une adaptation du lot ou du calcul, elle sera documentée avant de lire les scores 2026. Les modifications motivées par un bug seront distinguées d'un réglage expérimental.

## Informations disponibles

Même archive météo et même marge de disponibilité de 24 h que la baseline : offsets d'archive 48/72/96 h pour les prévisions à 24/48/72 h. Les variables tabulaires de la baseline sont accessibles au décodeur. Le réseau reçoit aussi les observations horaires brutes sous forme de séquence, avec remplissage causal borné et indicateurs de valeurs manquantes.

Une entrée supplémentaire fournit les erreurs des bulletins météo dont l'observation de vérification est déjà connue : `observation(s) − prévision_archivée_valide(s)` uniquement pour `s ≤ heure d'émission`. Cela enrichit l'information disponible par rapport à la baseline ; une éventuelle amélioration ne pourra pas être attribuée uniquement au type de cellule RNN.

## Découpage temporel et sélection

1. Apprentissage initial : toutes les cibles de chaque exemple, y compris +72 h, précèdent le 1er janvier 2025.
2. Validation : émissions à partir du 1er janvier 2025 et toutes les cibles avant le 1er octobre 2025. Moyenne sur les échéances du critère 50 % MAE normalisée + 50 % RMSE normalisée ; référence climatologique fixée avant l'optimisation.
3. Choisir la meilleure architecture récurrente sur les prédictions moyennes des deux graines, puis figer les nombres d'époques par graine.
4. Réentraîner ces réseaux avec toutes les cibles antérieures au 1er octobre 2025, pendant ces nombres d'époques fixes. Aucune décision d'arrêt sur calibration ou test.
5. Calibration : émissions à partir du 1er octobre 2025 et cibles avant le 1er janvier 2026. Calibration des intervalles sur les erreurs des réseaux réentraînés uniquement.
6. Comparaison 2026 : exactement les **748 dates cibles par échéance** de la baseline, en alignant séparément les heures d'émission de chaque horizon. La disponibilité des autres sorties ne doit pas supprimer une prévision évaluable.

Le test 2026 a déjà été consulté pendant les expériences précédentes. Il s'agit d'une **comparaison rétrospective sur un test réutilisé**, pas d'une validation indépendante. Aucun changement du réseau n'est autorisé en fonction de ces nouveaux scores. Une future évaluation indépendante devra utiliser de nouvelles dates et couvrir plusieurs saisons.

## Mesures et livrables

MAE, RMSE, biais, erreurs sur les vents dépassant le 90e percentile d'apprentissage, incertitude et couverture ; résultats par graine, architecture et échéance ; courbes d'apprentissage ; comparaison datée avec observations et arbres ; paramètres, versions, durées, protocole et limites. Les fichiers de poids et de normalisation restent locaux, tandis que les scripts, tests et comptes rendus sont commités par étape.
