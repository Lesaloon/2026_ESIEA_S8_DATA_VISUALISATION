# Simple price correlations

Run from the project root:

```sh
.venv/bin/python visualisation/correlations.py
```

Uses the existing SQLite database plus pandas, matplotlib and seaborn. No arguments.
Creates two images in this folder:

- `housing_correlations.png`: apartment/house sale price versus area, rooms, lots,
  and property type.
- `airbnb_correlations.png`: EUR nightly quote price versus capacity, bedrooms,
  beds, bathrooms, room type, minimum stay, availability, reviews, rating and superhost status.

Each image includes a correlation matrix and a ranking of price correlations.
The terminal prints the coefficients and the number of non-missing pairs used.

Spearman correlation measures rank association from -1 to +1 and is less sensitive
to extreme prices than Pearson correlation. Missing values are excluded per pair,
with at least 30 pairs required. There is no outlier trimming or zero imputation.

These are exploratory associations, not causal importance or a predictive model.
DVF prices can cover multiple properties and repeat across rows, so the housing
chart is at source-row level, not unique-sale level. Airbnb uses only available
positive EUR quotes, which may not represent listings with no quoted price.
IDs, price-derived revenue, and duplicate price fields are excluded. Arrondissement
is not treated as a numeric feature: district numbers have no meaningful order.

## Graphiques par quartier (arrondissement)

```sh
.venv/bin/python visualisation/quartiers.py
```

Le champ « quartier » de cette source Airbnb correspond aux 20 arrondissements,
pas aux 80 quartiers administratifs. Le script utilise ce découpage commun au DVF.

Il produit cinq images :

- `housing_correlations_quartiers.png` et `airbnb_correlations_quartiers.png` :
  corrélations de chaque variable avec le prix, une ligne par arrondissement.
- `housing_surface_quartiers.png` : prix médian selon la surface bâtie.
- `airbnb_capacite_quartiers.png` : prix médian selon le nombre de voyageurs.
- `airbnb_sejour_quartiers.png` : prix médian selon la durée minimum de séjour.

Les trois derniers graphiques contiennent chacun 20 petits panneaux. Les segments
rouges indiquent une baisse du prix médian entre deux groupes successifs : cela
permet d'explorer des changements de tendance qu'une corrélation globale masque.
Les chiffres près des points sont les effectifs. Un minimum de 30 observations est
requis par point ; les groupes trop petits sont masqués, sans relier leurs voisins.
Pour la surface et le séjour minimum, les points représentent des tranches de
quantiles calculées dans chaque arrondissement (x et y sont les médianes de tranche).
Pour la capacité d'accueil, chaque point représente une capacité exacte.
Les axes sont ajustés par panneau pour rendre les tendances locales lisibles :
il faut lire leurs graduations avant de comparer les quartiers. L'axe des durées
minimum de séjour est logarithmique pour distinguer les courts et longs séjours.

Une baisse de médiane n'établit ni un seuil précis ni une corrélation négative
significative au-delà de ce seuil. Le regroupement, la composition des logements
et les effectifs influencent les courbes. Les limites DVF et Airbnb décrites plus
haut s'appliquent aussi à ces graphiques. Les noms des variables restent identiques
à ceux du premier script pour faciliter la comparaison.
