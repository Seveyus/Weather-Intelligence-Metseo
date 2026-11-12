# Protocole — benchmark multi-sources du vent à Saint-Étienne (+24 / +48 / +72 h)

Rédigé le 22 octobre 2026, **avant tout entraînement et avant tout calcul d'erreur sur la période de test**. Seules la disponibilité et la complétude des données ont été inspectées (taux de valeurs présentes par station et par mois), pas leurs relations avec les prévisions.

## Question

Des données plus riches centrées sur Saint-Étienne améliorent-elles la prévision du vent moyen horaire (FF, km/h, station `42218011` SAINT-ETIENNE) à +24, +48 et +72 h ?

1. **Plus de modèles météo et de variables** : 6 modèles archivés au lieu de 4, 10 variables au lieu de 4.
2. **Stations voisines** : vent, pression et températures du réseau Météo-France autour de Saint-Étienne.

## Données

- **Observations** : exports horaires publics Météo-France (data.gouv.fr), départements 42, 69, 43 et 63, 2020 → 21 octobre 2026. Fichiers compressés tels que publiés et empreintes dans `data/meteofrance/provenance.json` ([fetch_meteofrance.py](../../analysis/fetch_meteofrance.py)).
- **Stations de vent** (règle : ≤ 60 km, FF présent ≥ 99 % des heures depuis 2024) : SAINT-ETIENNE, ST ETIENNE-BOUTHEON, ST-CHAMOND-P, PILAT GRAIX, MONISTROL-SUR-LOIRE, BRINDAS, ST-ANTHEME, MAZET-VOLAMONT, AMBERT, CHALMAZEL_RA, LYON-BRON, LES SAUVAGES.
- **Stations de pression mer** (PMER ≥ 90 %) : ST ETIENNE-BOUTHEON, LYON-BRON, LYON-ST EXUPERY, ROANNE-RIORGES_AERO, CLERMONT-FD.
- **Prévisions archivées** : Open-Meteo Previous Runs, au point de la station, 2024-01-01 → 2026-10-21. Modèles `ecmwf_ifs025`, `icon_global`, `icon_eu`, `gfs_global`, `gem_global`, `jma_seamless`. Variables : vent 10 m (vitesse, direction), rafales 10 m, vent 100 m, température 2 m, pression de surface et mer, humidité, nébulosité, précipitations. Un modèle qui ne fournit pas une variable la laisse vide ; rien n'est substitué.
- **Âge des prévisions** : comme dans la baseline figée, les échéances +24/+48/+72 h utilisent les décalages d'archive 48/72/96 h (24 h de marge de publication). Les valeurs NWP à ±3 h de l'heure cible proviennent du même décalage et sont donc au moins 21 h plus anciennes que l'émission.
- Observations supposées disponibles à leur horodatage (latence de transmission non mesurée, comme avant).

Exclus : AROME, ARPEGE, ICON-D2, HARMONIE (archives vides au point), UKMO et AIFS (archives seulement depuis 2025, absentes de l'essentiel de l'apprentissage).

## Découpage chronologique

| Rôle | Heures cibles |
| --- | --- |
| Développement | 2024-01-05 → 2026-02-04 23 h |
| Validation glissante (5 plis) | 2025 T1, T2, T3, T4, puis 2026-01-01 → 2026-02-04 |
| **Test final, jamais utilisé** | **2026-02-08 00 h → fin des observations (≈ 21 octobre 2026)** |

Chaque pli entraîne sur les cibles antérieures à son début moins 72 h, puis valide sur sa période. Le test exige des émissions postérieures ou égales au 2026-02-05 00 h pour toutes les échéances : une prévision à +72 h vise donc au plus tôt le 8 février. Les mêmes heures cibles sont comparées pour toutes les méthodes et échéances. Une heure n'est retenue que si FF observé, la persistance et les six vitesses NWP sont présents.

Le mois de janvier 2026 a déjà été regardé par les études précédentes. Il ne sert ici qu'en développement.

## Ensembles de variables (ablation)

- **A — v1** : historique de Saint-Étienne (variables et retards de la baseline figée) et 4 modèles × 4 variables, comme la baseline de février 2026.
- **B — +NWP** : A et les 6 modèles × 10 variables, la moyenne et la dispersion multi-modèles, le cisaillement 100 m / 10 m et la moyenne NWP à ±3 h.
- **C — +stations** : B et les stations voisines (vitesse et composantes u/v du vent, rafale max, retards 0/3/6/24 h, moyennes 24 h ; pression mer, tendances 3 h et 24 h, différences de pression entre stations ; écarts de température Pilat − Saint-Étienne et Bouthéon − Saint-Étienne).

## Méthodes candidates

- **Références** : persistance, moyenne saisonnière (mois × heure), chaque modèle NWP brut, moyenne NWP.
- **Pour chaque ensemble A, B, C** : Ridge sur le résidu de la moyenne NWP (α = 100), gradient boosting direct et gradient boosting sur le résidu (HistGradientBoosting, 7 ou 15 feuilles, 75/150/300 itérations, mêmes réglages que la baseline figée).

Le nombre d'itérations est choisi sur la moyenne des plis. Aucun réglage ne se fait sur le test.

## Règle de sélection

Score de pli = ½ (MAE / MAE saisonnière + RMSE / RMSE saisonnière), moyenné sur les 5 plis. Pour chaque ensemble et chaque échéance, le meilleur candidat est **le finaliste** de cet ensemble.

Le **modèle retenu** est le meilleur finaliste toutes familles confondues. Les finalistes et le choix sont écrits dans `selection.json` avant toute prédiction sur le test, puis réentraînés sur toutes les cibles de développement.

## Évaluation finale

- **Critère principal** : MAE et RMSE en km/h sur le test, par échéance, pour le modèle retenu, les finalistes A/B/C, la meilleure prévision NWP brute (choisie en validation) et la persistance.
- **Gain** de C sur A et du modèle retenu sur la NWP brute : bootstrap apparié par blocs de 7 jours (indicatif ; séries dépendantes).
- **Saisons** : scores par mois et par saison (février–mars, avril–juin, juillet–août, septembre–octobre). Ce sont des analyses descriptives.
- **Vents forts** : MAE et biais au-dessus du 90e percentile de FF du développement.
- **Intervalles 80 %** : rayon conforme calculé sur les résidus hors pli des 5 plis de validation. La couverture est mesurée sur le test, sans garantie pour des séries dépendantes.

Les résultats seront publiés quels qu'ils soient, y compris si les données supplémentaires n'apportent rien.

## Hors périmètre de ce benchmark

Le BiLSTM n'est pas réentraîné ici : le disque Linux ne permet pas d'installer PyTorch actuellement. Son ajout sur le même découpage et les mêmes heures cibles fera l'objet d'une extension explicitement datée.
