# Historique des modèles : prix par nuit et revenu annuel

Comment on est arrivé au modèle utilisé par l'appli web (`web/start.py`), étape par étape, avec les
chiffres de chaque test. Données : 26 920 logements entiers Airbnb actifs à Paris (instantané de juin
2026, devis de 7 nuits au plus, 1 % des prix extrêmes retirés de chaque côté).

✔ = ce qui a été retenu à chaque étape.

**Comment lire les colonnes**

- **R²** : part des écarts de prix que le modèle explique, de 0 à 1 (plus haut = mieux).
- **Erreur moyenne / médiane** : écart entre prix réel et prix prévu ; la moitié des annonces ont
  une erreur inférieure à l'erreur médiane.
- **RMSE** : erreur qui pèse davantage les grosses erreurs.
- **Biais** : réel − prévu, en moyenne ; positif = le modèle sous-estime.
- **Validation** : « 5 plis » = chaque annonce est prévue par un modèle qui ne l'a jamais vue ;
  « 5 plis par hôte » = idem, en gardant toutes les annonces d'un même hôte dans le même pli (plus strict).

## 1. Prix par nuit

| # | Étape (script) | Modèle | Variables | Validation | R² | Erreur moyenne | Erreur médiane | RMSE | Biais | À 50 € près |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Premier modèle (`modelisation/regression.py`) | Linéaire | taille en nombres (capacité, chambres, salles de bain), type de bien, arrondissement | 5 plis | 0,614 | 82 € | 49 € | 141 € | +16 € | 13 649 / 26 920 |
| 2a | Monuments (`modelisation/monuments.py`) | Linéaire | taille + type, **sans lieu** | 5 plis | 0,474 | 96 € | 60 € | — | — | — |
| 2b | | | + distance au centre | | 0,508 | 93 € | 57 € | — | — | — |
| 2c | | | + score d'influence des monuments | | 0,555 | 89 € | 53 € | — | — | — |
| 2d | | | + score pondéré par la fréquentation | | 0,506 | 93 € | 57 € | — | — | — |
| 2e | | | + arrondissement (= modèle 1) | | 0,614 | 82 € | 49 € | — | — | — |
| 2f | | | + arrondissement + distance au centre | | 0,623 | 81 € | 48 € | — | — | — |
| 2g ✔ | | | **+ arrondissement + score monuments** | | 0,629 | 81 € | 48 € | — | — | — |
| 3 | Analyse des erreurs (`modelisation/residus.py`) | — | diagnostic du modèle 1 : grands logements surestimés (8 pers. et plus : −92 €, 4 chambres et plus : −153 €), T2 sous-estimés (+24 €) | 5 plis | — | — | — | — | — | — |
| 4a ✔ | Comparaison (`modelisation/comparaison.py`) | **Linéaire enrichi** | **capacité et chambres en catégories** + salles de bain + type + arrondissement + score monuments | 5 plis | 0,648 | 77 € | 47 € | 124 € | +17 € | 14 124 / 26 920 |
| 4b | | Linéaire enrichi + Duan | idem, prix moyen au lieu du prix typique | | 0,637 | 79 € | 51 € | 122 € | +1 € | 13 249 |
| 4c | | Arbre de décision | mêmes informations | | 0,616 | 80 € | 49 € | 128 € | +16 € | 13 596 |
| 4d | | Forêt aléatoire | mêmes informations | | 0,633 | 78 € | 48 € | 124 € | +16 € | 13 790 |
| 4e | | Gradient boosting | mêmes informations | | 0,655 | 76 € | 47 € | 121 € | +16 € | 14 205 |
| 5a | Position exacte (`modelisation/precision.py`) | Linéaire enrichi | rappel (4a) | 5 plis par hôte | 0,647 | 77 € | 47 € | 124 € | +17 € | 14 129 / 26 920 |
| 5b | | Linéaire enrichi | + prix des 20 voisins | | 0,655 | 76 € | 46 € | 122 € | +16 € | 14 200 |
| 5c | | Gradient boosting | rappel (4e) | | 0,649 | 76 € | 47 € | 122 € | +17 € | 14 149 |
| 5d | | Gradient boosting | + voisins + latitude/longitude | | 0,662 | 75 € | 46 € | 120 € | +16 € | 14 375 |
| 5e | Surface (`modelisation/precision.py`, annonces dont la surface est connue) | Linéaire enrichi | rappel | 5 plis par hôte | 0,664 | 81 € | 49 € | 129 € | +17 € | 4 081 / 8 094 |
| 5f | | Linéaire enrichi | + surface | | 0,696 | 78 € | 46 € | 125 € | +16 € | 4 283 |
| 5g | | Linéaire enrichi | + surface + voisins | | 0,700 | 77 € | 46 € | 124 € | +16 € | 4 299 |
| 5h | | Gradient boosting | rappel | | 0,648 | 82 € | 50 € | 131 € | +17 € | 4 067 |
| 5i | | Gradient boosting | + surface + position | | 0,698 | 77 € | 46 € | 124 € | +16 € | 4 321 |
| **App** | `modele_lineaire_enrichi/` → `web/model.pkl` | **Linéaire enrichi (4a)** | + fourchette par arrondissement (la moitié des logements comparables, de ×0,76–0,83 à ×1,17–1,28 le prix prévu) | 5 plis | 0,648 | 77 € | 47 € | 124 € | — | 14 124 / 26 920 |

Les lignes 5e à 5i portent sur un autre ensemble d'annonces (8 094 au lieu de 26 920) : elles se
comparent entre elles, pas avec les autres lignes.

## 2. Revenu annuel

| # | Étape (script) | Calcul du revenu | Validation | R² | Erreur moyenne | Erreur médiane | Biais |
|---|---|---|---|---|---|---|---|
| R1 | `modelisation/monuments.py` | régression directe du revenu, avec l'arrondissement | 5 plis | 0,180 | 22,0 k€ | 13,4 k€ | — |
| R2 | `modelisation/monuments.py` | idem + score monuments | 5 plis | 0,189 | 21,9 k€ | 13,4 k€ | — |
| R3 | dossier final, 1re version | prix prévu (linéaire enrichi) × nuits **médianes** de l'arrondissement | 5 plis | 0,173 | 22,6 k€ | 14,7 k€ | +7,9 k€ |
| R4 | idem | prix prévu (boosting) × nuits médianes | 5 plis | 0,176 | 22,6 k€ | 14,5 k€ | +8,0 k€ |
| R5 | `modelisation/occupation.py` | prix prévu × nuits **moyennes** de l'arrondissement | 5 plis | 0,083 | 22,9 k€ | 16,4 k€ | +1,6 k€ |
| R6 | `modelisation/occupation.py` | prix prévu × **modèle de nuits** (logement seul) | 5 plis | 0,083 | 22,4 k€ | 16,1 k€ | −0,4 k€ |
| R7 ✔ | `modelisation/occupation.py` | prix prévu × **modèle de nuits (logement + gestion)** | 5 plis | 0,167 | 20,8 k€ | 14,4 k€ | −0,4 k€ |
| R8 | `modelisation/occupation.py` | idem + jours libres (fuite, écarté) | 5 plis | 0,187 | 20,3 k€ | 13,9 k€ | −0,5 k€ |
| R9 | `modele_lineaire_enrichi/` | linéaire enrichi × modèle de nuits (R7) | 5 plis | 0,169 | 20,7 k€ | 14,3 k€ | −0,4 k€ |
| R10 | `modele_gradient_boosting/` | boosting × modèle de nuits | 5 plis | 0,172 | 20,7 k€ | 14,3 k€ | −0,3 k€ |
| R11 | `modelisation/precision.py` | R9, avec la validation plus stricte | 5 plis par hôte | 0,148 | 21,2 k€ | 14,7 k€ | −0,3 k€ |
| Réf. | dossiers finaux | **vrai** prix × nuits prévues (le plafond atteignable) | 5 plis | 0,249 | 19,8 k€ | 14,3 k€ | −0,7 k€ |

Ordre de grandeur : le revenu annuel réel est de 35,5 k€ en moyenne et de 23,1 k€ en médiane.

À l'échelle d'un arrondissement (test 3 de `modelisation/occupation.py`), l'écart moyen entre revenu
prévu et revenu réel passe de 8,0 k€ (nuits médianes) à **0,7 k€** (modèle de nuits), et le classement
des arrondissements est identique.

## À retenir

- **Le score d'influence des monuments est dans le modèle depuis l'étape 2g.** Il fait gagner 1 € ;
  le gros gain vient de l'étape 4a (la taille en catégories) : de 81 à 77 €.
- **Le modèle de l'appli (4a) est le plus précis des modèles linéaires.** Seuls le boosting et l'ajout
  de la position ou de la surface font mieux, de 1 à 4 € seulement.
- **Pour le revenu, le levier a été le modèle de nuits (R7)**, pas le modèle de prix : même un prix
  parfait (ligne « Réf. ») laisserait environ 20 k€ d'erreur, à cause des nuits louées.

## Refaire les calculs

Depuis la racine du projet, avec le Python du `.venv` (`.venv/bin/python` sous Linux) :

```sh
.venv/Scripts/python modelisation/monuments.py               # étape 2, lignes R1-R2
.venv/Scripts/python modelisation/residus.py                 # étape 3
.venv/Scripts/python modelisation/comparaison.py             # étape 4
.venv/Scripts/python modelisation/precision.py               # étape 5, ligne R11
.venv/Scripts/python modelisation/occupation.py              # lignes R5-R8
.venv/Scripts/python modele_lineaire_enrichi/entrainement.py # ligne R9, modèle de l'appli
.venv/Scripts/python modele_gradient_boosting/entrainement.py # ligne R10
```

Les chiffres peuvent varier très légèrement d'une version de scikit-learn ou de pandas à l'autre.
