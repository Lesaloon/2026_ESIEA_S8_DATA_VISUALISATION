# Dash web interface

From the project root:

```sh
.venv/bin/python web/start.py
```

Then open `http://127.0.0.1:8050`. The app reads the SQLite path from `DATABASE`
in `.env` and uses `ingestion/paris.sqlite` by default.

## Connect the price model

Put the trusted pickle at `web/model.pkl`, or set `MODEL_PATH` in `.env`. Pickle
files can execute code when loaded, so never use a model from an untrusted source.

The recommended format is a dictionary containing the fitted estimator and its
metadata:

```python
bundle = {
    "model": fitted_model,
    "feature_columns": list(training_columns),
    "target_transform": "log",  # "none", "log", or "log1p"
}
```

The available raw inputs are:

- `arrondissement`, `latitude`, `longitude`
- `accommodates`, `bedrooms`, `bathrooms`, `minimum_nights`
- `property_type`, `room_type`
- `nearest_monument_distance_km`, `tourist_proximity_score`

`feature_columns` may contain these raw names or one-hot names such as
`arrondissement_7`, `property_type_Entire rental unit`, and
`room_type_Entire home/apt`. A plain fitted pipeline can also be pickled directly;
the app uses its `feature_names_in_` when available.

The estimator computes `tourist_proximity_score` from the selected map position
with the same three-nearest-monuments formula used during database ingestion. It
also builds the `capacity_*` and `rooms_*` dummy columns expected by the enriched
linear model. A bundle may provide `duan_factor` when that correction was used.
