"""Results of the saved enriched linear model: price, revenue, payback, monuments. No arguments.

Run modele_lineaire_enrichi/entrainement.py first: it trains the model and saves it in modeles/.
"""

from entrainement import FOLDER, features
from revenu import report


if __name__ == '__main__':
    report(features, FOLDER)
