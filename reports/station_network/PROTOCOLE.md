# Protocole — modèle « capteurs seuls » appris sur le réseau Météo-France

Rédigé le 22 octobre 2026, avant tout entraînement. Seules la disponibilité des fichiers et la présence des variables dans un département (l'Ain) ont été inspectées, pas leurs relations avec les cibles.

## But

La station météo de Télécom Saint-Étienne mesurera en temps réel :
- **SHT30** : température `T` et humidité `U` ;
- **BMP280** : pression `PSTAT` ;
- **anémomètre et girouette** : vent `FF`, direction `DD` et rafale `FXI` ;
- **pluviomètre à augets** : pluie `RR1`.

Le modèle doit prévoir toutes ces variables **à partir de ces seuls capteurs**, avec un apprentissage sur le plus grand historique disponible des mêmes mesures.

## Données

- **Source** : exports horaires publics Météo-France, 95 départements métropolitains, 2000 → 21 octobre 2026 ([fetch_meteofrance.py](../../analysis/fetch_meteofrance.py) puis [build_station_dataset.py](../../analysis/build_station_dataset.py)). Seules les 7 variables ci-dessus sont lues. Les valeurs physiquement impossibles sont mises à vide et rien n'est interpolé.
- **Stations retenues** : `T`, `U`, `FF`, `DD` et `RR1` présents chacun au moins 2 ans (17 532 h). `PSTAT` et `FXI` sont facultatifs : en entrée, une valeur absente est signalée par un masque ; en cible, elle n'est pas notée. Cette règle s'applique avant tout entraînement, sur le seul décompte des valeurs présentes.

## Découpage

Le découpage est à la fois **temporel et spatial**, pour simuler une station nouvelle installée à un endroit jamais vu :

| Ensemble | Stations | Heures cibles |
| --- | --- | --- |
| Apprentissage | 75 % des stations | 2000 → 2021 |
| Validation (choix du modèle) | 10 % d'autres stations | 2022 → 2023 |
| **Test principal** | **15 % d'autres stations + SAINT-ETIENNE (42218011)** | **2024-01-04 → 2026-10-21** |
| Test secondaire | stations d'apprentissage | 2024-01-04 → 2026-10-21 |

- Le tirage des stations est aléatoire (graine 42), fait une seule fois et enregistré dans `split.json` avant tout entraînement.
- SAINT-ETIENNE est forcée dans le test. Ses voisines peuvent rester en apprentissage, comme le seront demain les stations proches de Télécom.
- Une heure cible n'est notée que si la variable visée est observée. Toutes les méthodes sont comparées sur les mêmes heures, variable par variable.

## Entrées et cibles

- **Entrées** :
  - les 72 dernières heures des 7 capteurs, avec un masque de présence pour chacun ;
  - le vent décomposé en u/v ;
  - l'heure et le jour de l'année de l'émission ;
  - la latitude, la longitude et l'altitude de la station.

  Aucune prévision météo externe.
- **Cibles** :
  - `T` (°C), `U` (%), `PSTAT` (hPa), `FF` (km/h), `FXI` (km/h) ;
  - la **pluie dans l'heure** (`RR1` ≥ 0,2 mm, sous forme de probabilité) ;
  - la **direction** du vent, via les composantes u/v.
- **Échéances** : +1, +2, +3, +6, +12, +24, +48 et +72 h. L'objectif principal est +1 à +12 h. Les échéances au-delà sont rapportées pour savoir jusqu'où le modèle reste utile.

## Méthodes

1. **Références** :
   - la persistance (dernière valeur mesurée) ;
   - le cycle diurne persistant (valeur observée 24 h avant la cible, pour les échéances ≤ 24 h) ;
   - pour SAINT-ETIENNE seulement, les arbres mono-station de [sensor_horizons](../sensor_horizons/metrics.json), réévalués sur les mêmes heures. Ils servent de point de comparaison avec un modèle qui a vu l'historique du site.
2. **Arbres groupés** : un gradient boosting par variable et par échéance, appris sur un échantillon de 3 millions d'heures tirées des stations d'apprentissage. Les variables sont celles de l'étude mono-station, avec en plus la position et l'altitude.
3. **Réseau multi-têtes (GPU)**, en trois architectures candidates :
   - un MLP sur les variables construites ;
   - un GRU sur la fenêtre de 72 h ;
   - un réseau convolutif temporel.

   Chacune a une sortie par variable et par échéance. La perte est Huber sur des cibles normalisées, plus une entropie croisée pour la pluie, à poids égaux. Deux graines par architecture. L'arrêt anticipé se fait sur la validation.

## Règle de choix et évaluation

- **Score de validation** : moyenne, sur les variables et les échéances, du rapport MAE modèle / MAE persistance (Brier pour la pluie), calculée sur les stations et années de validation. Les deux graines de la meilleure architecture sont moyennées. Le choix est écrit dans `selection.json` avant toute prédiction sur le test.
- **Rapport final** :
  - MAE, RMSE et biais par variable et par échéance ;
  - score de Brier pour la pluie ;
  - erreur angulaire de direction quand `FF` ≥ 7,2 km/h ;
  - résultats séparés pour SAINT-ETIENNE, l'ensemble des stations de test et les stations d'apprentissage.
- **Intervalles** : bootstrap par blocs (station × mois), à titre indicatif.
- Tous les résultats sont publiés, y compris si le réseau ne bat pas les arbres.

## Limites connues d'avance

- Les capteurs et l'emplacement sont ceux de Météo-France. La station de Télécom demandera une période de comparaison pour corriger ses propres écarts (abri, hauteur de l'anémomètre, pluviomètre imprimé en 3D).
- Une partie de 2025–2026 a déjà été regardée pour SAINT-ETIENNE dans les études du vent. Le test ne compte comme jamais vu que pour les autres stations de test.
- Sans information extérieure, les échéances de 24 à 72 h resteront proches de la moyenne saisonnière ([étude mono-station](../sensor_horizons/metrics.json)). Ce protocole ne prétend pas lever cette limite.

## Amendements

**2026-11-12, 15 h environ — échantillon des arbres réduit de 3 à 1 million d'heures.** Avec 3 millions d'heures, un seul des 64 modèles d'arbres prenait 546 s (près de 10 h au total), et la mémoire vive de la machine (15 Go) tombait à 2 Go disponibles pendant que les réseaux s'entraînaient. Ce changement est décidé avant toute prédiction sur le test. Le seul score vu jusque-là est la validation de la température à +1 h des arbres (MAE 0,55 °C). Les 1 million d'heures sont tirées de la même façon, uniformément parmi les heures d'émission des stations d'apprentissage (graine 42). Rien d'autre ne change pour les arbres ni pour les réseaux.

**2026-11-12, 16 h 30 — évaluation des réseaux par groupes de stations.** La première évaluation des réseaux a manqué de mémoire GPU sur les 660 stations d'apprentissage, après avoir traité les stations de test et SAINT-ETIENNE sans écrire ni afficher aucun score. Le code d'évaluation note désormais les stations par groupes de 60 et additionne les erreurs, ce qui donne exactement les mêmes MAE, RMSE, biais et Brier. Les modèles retenus, leurs poids et le code qui les a entraînés ne changent pas : l'évaluation vérifie que le script sélectionné est une version enregistrée dans git (commit 386557a) et que ce protocole n'a changé que par des amendements ajoutés à la fin.
