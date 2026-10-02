"""Gradient boosting model of the nightly price: train it and save it. No arguments.

Bedrooms, capacity and bathrooms stay numbers (the trees find the thresholds themselves),
monument score, arrondissement and property type as native categories; learns the log of the price.
The training, saving and charts are shared with modele_lineaire_enrichi/: modelisation/revenu.py.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FOLDER = Path(__file__).resolve().parent
sys.path.append(str(FOLDER.parent / 'modelisation'))  # Shared pipeline and data preparation.
from revenu import train


NAME = 'Gradient boosting'


def model():
    return HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=42)


def features(frame, reference):
    """Categories keep the training codes, so a prediction grid is encoded like the training data."""
    X = frame[['bedrooms', 'accommodates', 'bathrooms', 'tourist_proximity_score']].astype(float)
    X['arrondissement'] = pd.Categorical(frame['arrondissement'], categories=range(1, 21))
    X['property_type'] = pd.Categorical(frame['property_type'], categories=reference['types'])
    return X


if __name__ == '__main__':
    train(NAME, model, features, FOLDER)
