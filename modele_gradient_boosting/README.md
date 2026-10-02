# Final model: gradient boosting

Run from the project root, after building `ingestion/paris.sqlite`:

```sh
.venv/bin/python modele_gradient_boosting/entrainement.py      # Linux: train and save
.venv/bin/python modele_gradient_boosting/resultats.py         # Linux: results and charts
.venv/Scripts/python modele_gradient_boosting/entrainement.py  # Windows
.venv/Scripts/python modele_gradient_boosting/resultats.py     # Windows
```

No arguments. Same outputs as `modele_lineaire_enrichi/`, so the two can be compared
file by file; both use the shared pipeline `modelisation/revenu.py`.

**Price model** (`entrainement.py`): scikit-learn's `HistGradientBoostingRegressor`
(default settings) on the log nightly price, from the same information as the linear
model: bedrooms, capacity and bathrooms as numbers (the trees find the thresholds),
monument score, and arrondissement and property type as native categories. Slightly more
precise on the price, but its effects have no single value in euros.

**Nights model** (shared with the other folder): gradient boosting of the nights booked
per year from the flat and its management (number of flats run by the host, minimum stay).
Tested in `modelisation/occupation.py`.

**Annual revenue** = mean predicted nightly price (Duan's correction) × predicted nights.

**Ranges**: every estimate comes with the range where half of the comparable listings fall,
measured on the out-of-fold errors of each arrondissement and saved with the model (see
`RANGE` and `predict_range()` in `modelisation/revenu.py`).

`entrainement.py` prints the price, nights and revenue errors by 5-fold cross-validation,
next to the former calculation (median nights of the arrondissement) and to a reference
that uses the real price, then trains both models on every listing and saves them in
`modeles/modele.joblib` (ignored by git: each machine trains its own).
Outputs: `resultats/erreurs_prix.csv`, `resultats/erreurs_revenu.csv`,
`graphiques/qualite_modele.png`.

`resultats.py` loads the saved models and estimates a standard studio, T2 and T3 in each
arrondissement under two management scenarios (**Gestion seul**: one flat, 2 nights
minimum; **Conciergerie**: as many flats as the median multi-flat host, 2 nights minimum,
20 % commission on the revenue). It gives the typical nightly price, nights booked, annual
revenue, years of net gain to pay back the purchase, each with its range, and the T2's revenue
at a low and a high monument score of its arrondissement.
Outputs: `resultats/logements_types.csv` and, in `graphiques/`: `prix_par_nuit.png`,
`fourchette_prix_t2.png`, `fourchette_revenu_t2_gestion_seul.png`,
`revenu_annuel_gestion_seul.png`, `revenu_annuel_conciergerie.png`,
`annees_remboursement_gestion_seul.png`, `annees_remboursement_conciergerie.png`,
`effet_monuments.png`.

Limits: same as the linear model (revenue error about 21 k€ a year for a single listing,
under 1 k€ averaged by arrondissement; mean revenues; high-season quotes; Paris rental rules
not applied).
