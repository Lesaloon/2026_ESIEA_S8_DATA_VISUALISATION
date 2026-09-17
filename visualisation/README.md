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
