"""Enriched linear model of the nightly price: train it and save it. No arguments.

Capacity (1 to 8+) and bedrooms (studio to 4+) as categories, bathrooms, monument score,
one-hot arrondissement and property type; learns the log of the price.
The training, saving and charts are shared with modele_gradient_boosting/: modelisation/revenu.py.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

FOLDER = Path(__file__).resolve().parent
sys.path.append(str(FOLDER.parent / 'modelisation'))  # Shared pipeline and data preparation.
from revenu import train


NAME = 'Linéaire enrichi'


def features(frame, reference):
    """Training drops the baselines (2 guests, 1 bedroom, 11th, classic flat); prediction reuses its columns."""
    frame = frame.assign(capacity=frame['accommodates'].clip(upper=8).astype(int),
                         rooms=frame['bedrooms'].clip(upper=4).astype(int))
    X = pd.get_dummies(frame[['bathrooms', 'tourist_proximity_score', 'capacity', 'rooms', 'arrondissement', 'property_type']],
                       columns=['capacity', 'rooms', 'arrondissement', 'property_type'], dtype=int)
    if 'columns' not in reference:
        return X.drop(columns=['capacity_2', 'rooms_1', 'arrondissement_11', 'property_type_Entire rental unit'])
    return X.reindex(columns=reference['columns'], fill_value=0)


if __name__ == '__main__':
    train(NAME, LinearRegression, features, FOLDER)
