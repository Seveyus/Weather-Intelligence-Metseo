# Format des données attendu de la station météo

Fiche pour l'équipe Arduino et serveur de la station de Télécom Saint-Étienne.

Le modèle de prévision a appris sur les données horaires de Météo-France. Pour qu'il fonctionne avec la station, le serveur doit produire **chaque heure une ligne au même format, avec les mêmes définitions** que Météo-France. Si une mesure est calculée autrement (unité, moyenne, durée), le modèle reçoit des valeurs qu'il n'a jamais vues et ses prévisions se dégradent sans prévenir.

## Une ligne par heure

Fichier CSV, séparateur `;`, une ligne ajoutée chaque heure quelques minutes après l'heure pile :

```text
AAAAMMJJHH;T;U;PSTAT;FF;DD;FXI;RR1
2026111214;11.4;78;958.6;3.2;230;7.9;0.4
2026111215;11.9;74;958.3;3.8;240;8.6;0.0
2026111216;11.6;76;958.1;;;;0.2
```

La dernière ligne montre un anémomètre en panne : les cases restent **vides**.

## Les colonnes

| Colonne | Capteur | Unité | Ce qu'il faut envoyer pour l'heure `HH` |
| --- | --- | --- | --- |
| `AAAAMMJJHH` | horloge | — | Année, mois, jour, heure **en UTC** (pas l'heure française). `2026111214` = 12 novembre 2026, 14 h UTC. |
| `T` | SHT30 | °C | Température **à l'heure pile** (moyenne de la dernière minute), au dixième. |
| `U` | SHT30 | % | Humidité relative **à l'heure pile** (moyenne de la dernière minute), entier de 0 à 100. |
| `PSTAT` | BMP280 | hPa | Pression **au niveau de la station**, à l'heure pile, au dixième. Le BMP280 donne des pascals : diviser par 100. **Ne pas** ramener au niveau de la mer. |
| `FF` | anémomètre | **m/s** | Vitesse **moyenne sur les 10 minutes** avant l'heure pile (`HH-0:10` → `HH:00`), au dixième. |
| `DD` | girouette | degrés | Direction **d'où vient** le vent, moyenne sur les mêmes 10 minutes. 90 = est, 180 = sud, 270 = ouest, 360 = nord. Vent nul : `FF` = 0 et `DD` vide. |
| `FXI` | anémomètre | **m/s** | **Rafale maximale de l'heure** (`HH-1:00` → `HH:00`) : plus forte moyenne glissante sur 3 secondes. L'anémomètre doit donc être lu **au moins une fois par seconde**. |
| `RR1` | pluviomètre | mm | **Cumul de pluie de l'heure** (`HH-1:00` → `HH:00`) : nombre de basculements × hauteur d'eau d'un auget. 0 s'il n'a pas plu. |

Ces unités et définitions sont celles des exports Météo-France (vent en m/s, comme dans leurs fichiers). Le serveur applique ensuite exactement la même préparation que pour l'entraînement : passage en km/h, contrôle des valeurs ([build_station_dataset.py](../analysis/build_station_dataset.py)). Les définitions sont à vérifier une fois dans le descriptif officiel des données horaires, disponible sur la [page du jeu de données](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires/).

## Règles

1. **Une valeur manquante reste vide.** Jamais `0`, `-999` ou la valeur de l'heure précédente. Le modèle sait qu'une case vide signifie « pas de mesure ». Un faux 0 serait pris pour une vraie mesure : 0 °C, 0 mm, vent nul.
2. **Rien n'est interpolé ni lissé** avant l'envoi. Le modèle fait son propre traitement.
3. **L'horloge est synchronisée** (NTP sur le Raspberry Pi ou l'UNO Q) et tout est en **UTC**, sans changement d'heure été/hiver.
4. **Les valeurs hors limites sont laissées vides** et signalées dans le journal. Le modèle les écarte de toute façon :

   | Colonne | Limites acceptées |
   | --- | --- |
   | `T` | −40 à 50 °C |
   | `U` | 0 à 100 % |
   | `PSTAT` | 600 à 1100 hPa |
   | `FF` | 0 à 70 m/s |
   | `DD` | 0 à 360° |
   | `FXI` | 0 à 97 m/s |
   | `RR1` | 0 à 150 mm |

5. **72 heures d'historique** sont nécessaires avant que les prévisions soient fiables : le modèle lit toujours les 72 dernières heures.

## Données brutes à conserver en plus

Pour corriger plus tard les écarts entre la station et Météo-France, gardez aussi les mesures **à la minute** : température, humidité, pression, vitesse et direction du vent, basculements du pluviomètre horodatés. Elles ne servent pas au modèle, mais elles permettent de recalculer les agrégats horaires si une définition change.

## Informations à transmettre une fois

Le modèle utilise la position de la station, et la correction des écarts a besoin de connaître l'installation :

| Information | Exemple | Pourquoi |
| --- | --- | --- |
| Latitude, longitude | 45.45, 4.39 | entrée du modèle |
| Altitude du sol | m | entrée du modèle |
| Hauteur de l'anémomètre | m au-dessus du sol (Météo-France : 10 m) | le vent augmente avec la hauteur |
| Nombre de directions de la girouette | 8 ou 16 | le modèle sera réentraîné avec la même résolution |
| Hauteur d'eau par basculement | mm (Météo-France : 0,2 mm) | conversion du nombre de basculements en mm |
| Emplacement de l'abri du SHT30 | ombre, ventilé, à environ 1,5 m du sol | le soleil direct fausse la température de plusieurs degrés |
| Dates des changements de capteur ou d'emplacement | — | les données d'avant et d'après ne sont plus comparables |

## Période de comparaison

Pendant les premières semaines, comparez les lignes horaires avec celles de la station Météo-France SAINT-ETIENNE (`42218011`, publiées chaque jour sur data.gouv.fr). Un écart régulier, par exemple un vent toujours plus faible sur le toit, se corrige facilement une fois mesuré.
