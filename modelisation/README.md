# Price regression and payback time by arrondissement

Run from the project root, after building `ingestion/paris.sqlite`:

```sh
.venv/bin/python modelisation/regression.py      # Linux
.venv/Scripts/python modelisation/regression.py  # Windows
```

No arguments. The database path comes from `DATABASE` in the project-root `.env`.
It answers: *what nightly price can an Airbnb investor charge, and which
arrondissements pay back the purchase fastest?*

1. Load the SQLite tables into pandas.
2. Keep active, short-term, entire-home listings with a usable EUR quote.
3. Drop the columns not known before buying (including latitude and longitude).
4. One-hot encode arrondissement and property type (baselines: 11th, *Entire rental unit*).
5. Fit two linear regressions, on the log of nightly price and of annual revenue.
6. Predict the price of a standard studio, T2 and T3 in each arrondissement.
7. Divide the purchase cost (median DVF price of single-flat sales + notary +
   furnishing) by the yearly net gain of a self-managed flat: years to pay back.
   Revenue is taken two ways (predicted price × median occupancy, or predicted
   revenue) to check the ranking holds.

The terminal prints model quality, variable effects in euros and the detailed
tables. Three charts are saved in this folder:

- `prix_par_nuit.png`: predicted nightly price per arrondissement and flat type.
- `annees_remboursement.png`: years of net revenue needed to pay back the purchase.
- `modeles.png`: predicted against real values on test listings, for both regressions.

Cost assumptions are constants at the top of the script. Results are optimistic:
high-season quotes, review-based occupancy estimates, and Paris short-term rental
rules are not applied.

`regression.ipynb` is the first, notebook version of the same analysis (returns in %).
