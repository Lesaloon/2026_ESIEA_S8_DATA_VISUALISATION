"""Does the minimum stay make the nightly price model more precise? No arguments.

Adds the minimum stay (1 to 29 nights, the web form's range) to the final price models, in three
forms: a number (the only one web/start.py can feed without change), its log, and categories.
Same listings and same 5 folds as modele_lineaire_enrichi/ (revenu.out_of_fold).
"""

import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from monuments import BLUE, ORANGE, finish, plain
from precision import BOOSTING, LINEAR, boosting
from regression import INK, load_data
from residus import metrics as price_metrics
from revenu import PRICE, load_listings, out_of_fold, show


STAYS = [0, 1, 2, 3, 6, 29]
STAY_LABELS = ['1', '2', '3', '4-6', '7-29']


def with_stay(base, form):
    """A final model's inputs plus the minimum stay in the given form."""
    def build(frame, reference):
        X = base(frame, reference)
        nights = frame['minimum_nights'].astype(float)
        if form == 'nombre':
            X['minimum_nights'] = nights
        elif form == 'log':
            X['log_minimum_nights'] = np.log(nights)
        elif form == 'catégories':
            stay = pd.cut(nights, STAYS, labels=STAY_LABELS)
            X = X.join(pd.get_dummies(stay, prefix='sejour', dtype=int).drop(columns='sejour_2'))
        return X
    return build


VARIANTS = {
    'Linéaire enrichi (actuel)': (LinearRegression, LINEAR),
    'Linéaire enrichi + séjour minimum (nombre)': (LinearRegression, with_stay(LINEAR, 'nombre')),
    'Linéaire enrichi + séjour minimum (log)': (LinearRegression, with_stay(LINEAR, 'log')),
    'Linéaire enrichi + séjour minimum (catégories)': (LinearRegression, with_stay(LINEAR, 'catégories')),
    'Gradient boosting (actuel)': (boosting, BOOSTING),
    'Gradient boosting + séjour minimum (nombre)': (boosting, with_stay(BOOSTING, 'nombre')),
}


def quotes(data):
    """How often the quote length equals the minimum stay, and the median price by minimum stay."""
    airbnb, _, _ = load_data()
    quote = (pd.to_datetime(airbnb.loc[data.index, 'price_quote_checkout_date'])
             - pd.to_datetime(airbnb.loc[data.index, 'price_quote_checkin_date'])).dt.days
    same = int((quote == data['minimum_nights']).sum())
    print(f'\nDevis dont la durée est égale au séjour minimum : {same:,} sur {len(data):,}'.replace(',', ' '))
    stay = pd.cut(data['minimum_nights'], STAYS, labels=STAY_LABELS)
    table = data.groupby(stay, observed=True)[PRICE].agg(annonces='size', **{'prix médian (€)': 'median'})
    print('\nAnnonces et prix médian par séjour minimum\n' + table.round(0).to_string())


def effects(data, types):
    """Price of the reference T2 (2 guests, 1 bedroom, 11th, median monument score) by minimum stay."""
    reference = pd.DataFrame({'minimum_nights': np.arange(1, 30), 'accommodates': 2, 'bedrooms': 1, 'bathrooms': 1,
                              'arrondissement': 11, 'property_type': 'Entire rental unit',
                              'tourist_proximity_score': data['tourist_proximity_score'].median()})
    both = pd.concat([data, reference], ignore_index=True)  # One design matrix: same columns for both parts.
    curves = {}
    for form in ['nombre', 'log', 'catégories']:
        X = with_stay(LINEAR, form)(both, {'types': types})
        model = LinearRegression().fit(X.iloc[:len(data)], np.log(data[PRICE].to_numpy()))
        curves[form] = np.exp(model.predict(X.iloc[len(data):]))
    curves = pd.DataFrame(curves, index=reference['minimum_nights'])
    print('\nPrix typique du T2 de référence selon le séjour minimum (€)\n'
          + curves.loc[[1, 2, 3, 5, 7, 14, 29]].round(0).to_string())
    return curves


def plot(curves, data):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    longest = int(data['minimum_nights'].max())
    ax.axvspan(longest, 29, color='#f0efec', zorder=0)  # No listing there: the curves are extrapolated.
    ax.text((longest + 29) / 2, curves.to_numpy().max(), 'aucune annonce :\nextrapolation', ha='center', va='top',
            color=INK, fontsize=9)
    for (form, color) in zip(curves, [BLUE, ORANGE, '#1baf7a']):
        ax.plot(curves.index, curves[form], color=color, linewidth=2, label=f'Modèle, séjour minimum en {form}')
    ax.set_xlabel('Séjour minimum (nuits)', color=INK)
    ax.set_ylabel('Prix par nuit prévu, T2 de référence (€)', color=INK)
    ax.set_xticks([1, 2, 3, 5, 7, 10, 14, 21, 29])
    ax.legend(loc='lower left', frameon=False)
    plain(ax, 'y')
    finish(fig, 'Séjour minimum et prix par nuit, selon la forme donnée au modèle',
           'T2 de référence : 2 personnes, 1 chambre, 1 salle de bain, 11e, score monuments médian.\n'
           f'Les données ne vont que jusqu’à {longest} nuits (devis de 7 nuits au plus) ; au-delà, jusqu’aux 29 nuits '
           'du formulaire, le modèle extrapole.',
           'sejour_minimum.png')


def main():
    data, _, _, _ = load_listings()
    types = sorted(data['property_type'].unique())
    quotes(data)
    rows = {}
    for name, (make_model, build) in VARIANTS.items():
        typical, _ = out_of_fold(make_model, build, data, types)
        rows[name] = price_metrics(data[PRICE], typical)
    show('Prix par nuit typique (€, mêmes 5 plis que le modèle de l’appli)', rows)
    plot(effects(data, types), data)


if __name__ == '__main__':
    main()
