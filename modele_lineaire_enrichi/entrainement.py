"""Enriched linear model of the nightly price: train it and save it. No arguments.

Capacity (1 to 8+) and bedrooms (studio to 4+) as categories, bathrooms, monument score,
one-hot arrondissement and property type; learns the log of the price.
The training, saving and charts are shared with modele_gradient_boosting/: modelisation/revenu.py.
Also writes the price model with its range for the web app (MODEL_PATH, web/model.pkl by default).
"""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

FOLDER = Path(__file__).resolve().parent
sys.path.append(str(FOLDER.parent / 'modelisation'))  # Shared pipeline and data preparation.
from revenu import RANGE, train


NAME = 'Linéaire enrichi'
# Same setting as web/start.py; the .env file is already loaded by the shared pipeline.
WEB_MODEL = Path(os.getenv('MODEL_PATH', 'web/model.pkl'))


def features(frame, reference):
    """Training drops the baselines (2 guests, 1 bedroom, 11th, classic flat); prediction reuses its columns.

    Category names (capacity_8+, rooms_4+) match the dummy columns built by web/start.py.
    """
    frame = frame.assign(capacity=frame['accommodates'].clip(upper=8).astype(int).astype(str).replace('8', '8+'),
                         rooms=frame['bedrooms'].clip(upper=4).astype(int).astype(str).replace('4', '4+'))
    X = pd.get_dummies(frame[['bathrooms', 'tourist_proximity_score', 'capacity', 'rooms', 'arrondissement', 'property_type']],
                       columns=['capacity', 'rooms', 'arrondissement', 'property_type'], dtype=int)
    if 'columns' not in reference:
        return X.drop(columns=['capacity_2', 'rooms_1', 'arrondissement_11', 'property_type_Entire rental unit'])
    return X.reindex(columns=reference['columns'], fill_value=0)


def export_for_web(bundle):
    """The price model in the format web/start.py reads (see web/README.md), with its range by arrondissement."""
    path = WEB_MODEL if WEB_MODEL.is_absolute() else FOLDER.parent / WEB_MODEL
    low, high = RANGE
    web = {
        'name': bundle['name'],
        'model': bundle['model'],
        'feature_columns': bundle['columns'],
        'target_transform': 'log',  # Typical price, the one to charge: no Duan factor.
        'price_range': {int(area): (float(row[low]), float(row[high])) for area, row in bundle['price_range'].iterrows()},
        'range_label': ('la moitié des logements comparables' if RANGE == (0.25, 0.75)
                        else f'{high - low:.0%} des logements comparables'),
        'sklearn': bundle['sklearn'],
    }
    with path.open('wb') as file:
        pickle.dump(web, file)
    print(f'Modèle pour l’appli web : {path}')


if __name__ == '__main__':
    export_for_web(train(NAME, LinearRegression, features, FOLDER))
