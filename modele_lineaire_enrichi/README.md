# Final model: enriched linear regression

Run from the project root, after building `ingestion/paris.sqlite`:

```sh
.venv/bin/python modele_lineaire_enrichi/entrainement.py      # Linux: train and save
.venv/bin/python modele_lineaire_enrichi/resultats.py         # Linux: results and charts
.venv/Scripts/python modele_lineaire_enrichi/entrainement.py  # Windows
.venv/Scripts/python modele_lineaire_enrichi/resultats.py     # Windows
```

No arguments. Same outputs as `modele_gradient_boosting/`, so the two can be compared
file by file; both use the shared pipeline `modelisation/revenu.py`.

**Price model** (`entrainement.py`): linear regression of the log nightly price on capacity
(1 to 8+) and bedrooms (studio to 4+) as categories, bathrooms, monument score, and
one-hot arrondissement and property type. Every effect reads in euros.

**Nights model** (shared with the other folder): gradient boosting of the nights booked
per year from the flat (bedrooms, capacity, bathrooms, monument score, arrondissement,
type) and its management (number of flats run by the host, minimum stay). Tested in
`modelisation/occupation.py`.

**Annual revenue** = mean predicted nightly price (Duan's correction) × predicted nights.

**Ranges**: every estimate comes with the range where half of the comparable listings fall.
It is measured on the out-of-fold errors of each arrondissement (central ones are wider) and
saved with the model; `RANGE` at the top of `modelisation/revenu.py` sets the share (quartiles
by default; `(0.1, 0.9)` would hold 8 listings in 10). For any flat,
`revenu.predict_range(bundle, frame, features)` returns the low, typical and high nightly price.

`entrainement.py` prints the price, nights and revenue errors by 5-fold cross-validation,
next to the former calculation (median nights of the arrondissement) and to a reference
that uses the real price. It then trains both models on every listing and saves them in
`modeles/modele.joblib` (ignored by git: each machine trains its own, as loading needs the
same scikit-learn version). Outputs: `resultats/erreurs_prix.csv`,
`resultats/erreurs_revenu.csv`, `graphiques/qualite_modele.png`.

**Web app**: `entrainement.py` also writes the price model for `web/start.py` at `MODEL_PATH`
(`web/model.pkl` by default, ignored by git), in the format of `web/README.md`, plus
`price_range` (low and high multipliers by arrondissement) and `range_label`: the app shows
the typical price and its range. Train with the same Python as the app (the project `.venv`):
a pickled model needs the same scikit-learn version to load.

`resultats.py` loads the saved models and estimates a standard studio, T2 and T3 in each
arrondissement, under two management scenarios (assumptions at the top of
`modelisation/revenu.py`):

- **Gestion seul**: a host with one flat, 2 nights minimum;
- **Conciergerie**: a host running as many flats as the median multi-flat host, 2 nights
  minimum, and a 20 % commission on the revenue taken from the net gain.

It gives the typical nightly price, nights booked, annual revenue, years of net gain to pay
back the purchase (DVF prices and cost assumptions of `modelisation/regression.py`), each with
its range, and the T2's revenue at a low and a high monument score of its own arrondissement.
Outputs: `resultats/logements_types.csv` and, in `graphiques/`: `prix_par_nuit.png`,
`fourchette_prix_t2.png`, `fourchette_revenu_t2_gestion_seul.png`,
`revenu_annuel_gestion_seul.png`, `revenu_annuel_conciergerie.png`,
`annees_remboursement_gestion_seul.png`, `annees_remboursement_conciergerie.png`,
`effet_monuments.png`.

Limits: for a single listing, revenue error is still about 21 k€ a year (nights booked are
estimated by Inside Airbnb from reviews and vary a lot); averaged by arrondissement it is
under 1 k€. Revenues are mean (expected) values, above the typical listing's. Prices are
high-season quotes; Paris short-term rental rules are not applied.
