"""Can the annual revenue error shrink? Two tests around the final enriched linear model. No arguments.

1. Nights booked: instead of the arrondissement's median, a gradient boosting model predicts each
   listing's nights from the flat, then from the flat and its management.
2. Error where the decision is made: predicted against real revenue, averaged by arrondissement.
Revenue = mean predicted nightly price (same 5 folds, Duan) x nights booked.
"""

import sys

import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from monuments import BLUE, ORANGE, finish, plain
from regression import GRID, INK, ROOT, load_data, prepare_listings
from revenu import FOLDS, PRICE, REVENUE, out_of_fold, revenue_metrics, show

sys.path.append(str(ROOT / 'modele_lineaire_enrichi'))
from entrainement import features


def linear_features(frame, reference):
    """The enriched linear price model as tested here (step 4a, before the minimum stay was added)."""
    return features(frame, reference).drop(columns='minimum_nights', errors='ignore')


NIGHTS = 'estimated_occupancy_l365d'
FLAT = ['bedrooms', 'accommodates', 'bathrooms', 'tourist_proximity_score']
MANAGEMENT = ['calculated_host_listings_count', 'minimum_nights']
# Days still free over the coming year: bookings lower it, so it partly contains the answer.
LEAK = ['availability_365']
CURRENT, MEAN = 'Nuits médianes de l’arrondissement (actuel)', 'Nuits moyennes de l’arrondissement'
FLAT_MODEL, MANAGED_MODEL = 'Modèle de nuits : logement', 'Modèle de nuits : logement + gestion'
NIGHT_MODELS = {FLAT_MODEL: FLAT, MANAGED_MODEL: FLAT + MANAGEMENT,
                'Modèle de nuits : + jours libres (fuite)': FLAT + MANAGEMENT + LEAK}
COLORS = [BLUE, ORANGE, '#1baf7a', '#eda100', '#e87ba4']  # Fixed categorical order.


def load():
    airbnb, _, names = load_data()
    data, median_nights = prepare_listings(airbnb)
    data = data[data[PRICE].between(*data[PRICE].quantile([0.01, 0.99]))]  # Same trimming as the final models.
    data = data.join(airbnb[['tourist_proximity_score', NIGHTS] + MANAGEMENT + LEAK])
    # Same population as prepare_listings' median: active short-term entire homes, priced or not.
    market = airbnb[(airbnb['room_type'] == 'Entire home/apt') & (airbnb['number_of_reviews_ltm'] > 0)
                    & (airbnb['minimum_nights'] < 30)]
    return data, median_nights, market.groupby('arrondissement')[NIGHTS].mean(), names


def predict_nights(data, columns, types):
    """Nights booked of every listing, from the folds that exclude it."""
    X = data[columns].astype(float)
    X['arrondissement'] = pd.Categorical(data['arrondissement'], categories=range(1, 21))
    X['property_type'] = pd.Categorical(data['property_type'], categories=types)
    predicted = pd.Series(np.nan, index=data.index)
    for train_rows, test_rows in FOLDS.split(X):
        model = HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=42)
        predicted.iloc[test_rows] = model.fit(X.iloc[train_rows], data[NIGHTS].iloc[train_rows]).predict(X.iloc[test_rows])
    return predicted.clip(lower=0)


def by_arrondissement(data, revenues, names):
    """Mean real and predicted revenue of each arrondissement, and how well they match."""
    table = pd.DataFrame({'réel': data.groupby('arrondissement')[REVENUE].mean()})
    for name in [CURRENT, MEAN, MANAGED_MODEL]:
        table[name] = revenues[name].groupby(data['arrondissement']).mean()
    table.index = [f'{n:02} · {names[n]}' for n in table.index]
    print('\nTest 3 · revenu annuel moyen par arrondissement, réel et prévu (€/an)\n' + table.round(0).to_string())
    for name in table.columns[1:]:
        gap = table[name] - table['réel']
        print(f'  {name} : écart moyen {gap.abs().mean():,.0f} €/an (biais {gap.mean():+,.0f}), '
              f'plus grand écart {gap.abs().max():,.0f} €/an, classement des arrondissements identique '
              f'à {table[name].corr(table["réel"], method="spearman"):.2f} près (1 = identique)')
    cells = data.assign(rooms=data['bedrooms'].clip(upper=3))
    for name in [CURRENT, MANAGED_MODEL]:
        mean = cells.assign(predicted=revenues[name]).groupby(['arrondissement', 'rooms'])[[REVENUE, 'predicted']].mean()
        print(f'  Par arrondissement et nombre de chambres (80 cases) · {name} : écart moyen '
              f'{(mean["predicted"] - mean[REVENUE]).abs().mean():,.0f} €/an, classement à '
              f'{mean["predicted"].corr(mean[REVENUE], method="spearman"):.2f} près')
    return table


def plot_errors(table):
    columns = ['erreur moyenne (€/an)', 'erreur médiane (€/an)', 'biais moyen (€/an)']
    rows = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    step = 0.84 / len(table)
    for i, (name, color) in enumerate(zip(table.index, COLORS)):
        values = table.loc[name, columns].astype(float) / 1000
        offset = (i - (len(table) - 1) / 2) * step
        ax.barh(rows + offset, values, height=step * 0.9, color=color, label=name)
        for row, value in zip(rows, values):
            ax.text(value, row + offset, f' {value:.1f} k€'.replace('.', ','), va='center', fontsize=8.5, color=INK)
    ax.set_yticks(rows, [c.replace(' (€/an)', '') for c in columns])
    ax.invert_yaxis()
    ax.set_xlabel('k€ par an (moins = mieux ; biais : proche de 0 = mieux)', color=INK)
    ax.margins(x=0.15)
    plain(ax, 'x')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.01, 0.88), ncols=2, frameon=False)
    finish(fig, 'Test 1 · Prédire les nuits louées réduit-il l’erreur sur le revenu ?',
           'Revenu = prix moyen prévu (linéaire enrichi) × nuits louées, en validation croisée sur 5 plis.\n'
           '« Jours libres » : en partie calculés à partir des réservations, ils donnent la réponse au modèle ; à ne pas utiliser.',
           'occupation_revenu.png', top=0.78)


def plot_by_arrondissement(table):
    table = table.sort_values('réel') / 1000
    rows = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.hlines(rows, table.min(axis=1), table.max(axis=1), color=GRID, linewidth=2, zorder=1)
    ax.scatter(table['réel'], rows, s=90, facecolor='white', edgecolor='#0b0b0b', linewidth=1.5, zorder=3,
               label='Revenu moyen réel')
    for name, color in [(CURRENT, BLUE), (MANAGED_MODEL, '#eda100')]:
        ax.scatter(table[name], rows, s=60, color=color, edgecolor='white', linewidth=1.2, zorder=2,
                   label=f'Prévu · {name}')
    ax.set_yticks(rows, table.index)
    ax.set_xlabel('Revenu annuel moyen par annonce (k€)', color=INK)
    ax.legend(loc='lower right', frameon=False)
    plain(ax, 'x')
    finish(fig, 'Test 3 · À l’échelle de l’arrondissement, le revenu moyen est-il bien prévu ?',
           'Moyenne, sur les annonces de chaque arrondissement, du revenu réel et du revenu prévu en validation croisée.\n'
           'Une annonce isolée reste incertaine ; c’est la moyenne par arrondissement qui sert à choisir où investir.',
           'occupation_par_arrondissement.png')


def main():
    data, median_nights, mean_nights, names = load()
    types = sorted(data['property_type'].unique())
    _, price = out_of_fold(LinearRegression, linear_features, data, types)  # Mean nightly price (Duan).

    nights = {CURRENT: data['arrondissement'].map(median_nights), MEAN: data['arrondissement'].map(mean_nights)}
    nights |= {name: predict_nights(data, columns, types) for name, columns in NIGHT_MODELS.items()}
    quality = pd.DataFrame({name: {'R²': r2_score(data[NIGHTS], n), 'erreur moyenne (nuits)': mean_absolute_error(data[NIGHTS], n)}
                            for name, n in nights.items()}).T
    print('\nTest 1 · qualité de la prévision des nuits louées par an (validation croisée sur 5 plis)\n'
          + quality.round({'R²': 3, 'erreur moyenne (nuits)': 1}).to_string())
    revenues = {name: price * n for name, n in nights.items()}
    table = show('Test 1 · revenu annuel = prix moyen prévu × nuits louées (€/an)',
                 {name: revenue_metrics(data[REVENUE], revenue) for name, revenue in revenues.items()})

    plot_errors(table)
    plot_by_arrondissement(by_arrondissement(data, revenues, names))


if __name__ == '__main__':
    main()
