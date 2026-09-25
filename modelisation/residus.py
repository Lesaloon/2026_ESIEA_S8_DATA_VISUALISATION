"""Residual analysis of the nightly price model (size, type, arrondissement). No arguments.

Every listing is predicted by a model that never saw it (5-fold cross-validation). Prints the
error figures in euros, errors by segment, how monument proximity follows the errors and the
largest errors; saves three charts. metrics() is the yardstick for the next models.
"""

import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from monuments import BLUE, DIVERGING, finish, plain, sign_test
from regression import INK, TARGETS, encode, load_data, prepare_listings


TARGET = TARGETS['Prix par nuit']
PRICE_BANDS = [0, 130, 160, 200, 250, 320, 450, np.inf]
PRICE_LABELS = ['< 130 €', '130–160', '160–200', '200–250', '250–320', '320–450', '> 450 €']
ROOMS = ['Studio', '1 chambre', '2 chambres', '3 et +']
LIGHT_BLUE = '#b7d3f6'


def metrics(actual, predicted):
    """Error figures in euros per night: the same yardstick for every model."""
    error = actual - predicted
    return {
        'R² (log)': r2_score(np.log(actual), np.log(predicted)),
        'erreur moyenne (€)': error.abs().mean(),
        'erreur médiane (€)': error.abs().median(),
        'RMSE (€)': np.sqrt((error ** 2).mean()),
        'biais moyen (€)': error.mean(),
        'à 25 € près': int((error.abs() <= 25).sum()),
        'à 50 € près': int((error.abs() <= 50).sum()),
        'annonces': len(error),
    }


def out_of_fold_errors():
    """Current price model; each listing is predicted by the four fifths of the data without it."""
    airbnb, _, names = load_data()
    data, _ = prepare_listings(airbnb)
    data = data[data[TARGET].between(*data[TARGET].quantile([0.01, 0.99]))]  # Same trimming as regression.py.
    data = data.join(airbnb[['name', 'listing_url', 'nearest_monument_distance_km', 'tourist_proximity_score']])
    folds = KFold(5, shuffle=True, random_state=42)
    data['predicted'] = np.exp(cross_val_predict(LinearRegression(), encode(data), np.log(data[TARGET]), cv=folds))
    data['error'] = data[TARGET] - data['predicted']  # Above 0: dearer than predicted.
    return data, names


def print_metrics(data):
    current = metrics(data[TARGET], data['predicted'])
    naive = metrics(data[TARGET], pd.Series(data[TARGET].median(), index=data.index))
    table = pd.DataFrame([current, naive], index=['Modèle actuel', 'Sans modèle (prix médian pour tous)']).round(3)
    counts = table.columns.drop('R² (log)')
    table[counts] = table[counts].round().astype(int)
    print('\nChiffres d’erreur, € par nuit (validation croisée sur 5 plis)\n' + table.astype(object).T.to_string())
    # A model fitted on log prices predicts medians: Duan's factor rescales them to means.
    smearing = (data[TARGET] / data['predicted']).mean()
    print(f'\nLe modèle prédit des prix médians : il sous-estime de {current["biais moyen (€)"]:.0f} € en moyenne. '
          f'Multiplier ses prévisions par {smearing:.3f} (correction de Duan) ramènerait ce biais à '
          f'{(data[TARGET] - data["predicted"] * smearing).mean():.0f} €.')
    return current


def by_segment(data, key, label):
    """Median error, spread and precision for each value of a segment."""
    table = data.groupby(key, observed=True)['error'].agg(**{
        'annonces': 'size', 'médiane': 'median', 'p25': lambda e: e.quantile(0.25),
        'p75': lambda e: e.quantile(0.75), 'erreur absolue moyenne': lambda e: e.abs().mean()}).rename_axis(None)
    print(f'\nErreur selon {label} (réel − prévu, € par nuit)\n' + table.round(0).to_string())
    return table


def monument_leads(data):
    """Do the monument variables, absent from the model, follow its errors?"""
    print('\nPiste : les variables des monuments suivent-elles les erreurs ? (corrélation de rang avec l’erreur)')
    for column, label, sign in [('tourist_proximity_score', 'score monuments', 1),
                                ('nearest_monument_distance_km', 'distance au monument le plus proche', -1)]:
        within = data.groupby('arrondissement').apply(
            lambda group: group['error'].corr(group[column], method='spearman'), include_groups=False)
        right_way = int((sign * within > 0).sum())
        print(f'  {label} : {data["error"].corr(data[column], method="spearman"):+.3f} dans tout Paris, '
              f'{within.median():+.3f} en médiane dans un même arrondissement ; dans le sens attendu dans '
              f'{right_way} arrondissements sur 20 (probabilité si c’était le hasard : {sign_test(right_way, 20):.5f})')
    print('  Détail et graphiques : modelisation/monuments.py')


def largest_errors(data, names, count=10):
    shown = data.assign(annonce=data['name'].str.slice(0, 45),
                        arrondissement=[f'{n:02} · {names[n]}' for n in data['arrondissement']])
    shown = shown[['annonce', 'arrondissement', 'accommodates', 'bedrooms', TARGET, 'predicted', 'error', 'listing_url']].rename(
        columns={'accommodates': 'pers.', 'bedrooms': 'ch.', TARGET: 'réel (€)', 'predicted': 'prévu (€)',
                 'error': 'écart (€)', 'listing_url': 'lien'})
    for title, rows in [('plus chères que prévu', shown.nlargest(count, 'écart (€)')),
                        ('moins chères que prévu', shown.nsmallest(count, 'écart (€)'))]:
        print(f'\nLes {count} annonces les {title}\n' + rows.round(0).to_string(index=False))


def plot_distribution(data, figures):
    error = data['error']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(error.clip(-300, 300), bins=np.arange(-300, 310, 10), color=BLUE, edgecolor='white', linewidth=0.5)
    top = ax.get_ylim()[1]
    for value, label, align in [(error.median(), 'médiane', 'right'), (error.mean(), 'moyenne', 'left')]:
        ax.axvline(value, color=INK, linewidth=1)
        ax.text(value, top * 0.97, f' {label} {value:+.0f} € ', ha=align, va='top', color=INK, fontsize=9)
    ax.set_xlabel('Prix réel − prix prévu (€ par nuit) ; au-delà de ±300 €, regroupé aux bords', color=INK)
    ax.set_ylabel('Annonces', color=INK)
    plain(ax, 'y')
    count = {name: f'{figures[name]:,}'.replace(',', ' ') for name in ['annonces', 'à 25 € près', 'à 50 € près']}
    finish(fig, 'Distribution des erreurs du modèle de prix',
           f'{count["annonces"]} annonces, chacune prévue par un modèle qui ne l’a jamais vue : '
           f'{count["à 25 € près"]} à 25 € près, {count["à 50 € près"]} à 50 € près.\n'
           'À droite de 0 : logements plus chers que prévu.',
           'residus_distribution.png')


def plot_segments(by_price, by_capacity):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True,
                             gridspec_kw={'width_ratios': [len(by_price), len(by_capacity)]})
    for ax, table, xlabel in [(axes[0], by_price, 'Prix prévu (€ par nuit)'),
                              (axes[1], by_capacity, 'Capacité (personnes)')]:
        x = np.arange(len(table))
        ax.vlines(x, table['p25'], table['p75'], color=LIGHT_BLUE, linewidth=10,
                  label='La moitié des annonces (du 1er au 3e quartile)')
        ax.scatter(x, table['médiane'], s=60, color=BLUE, zorder=3, label='Erreur médiane')
        for xi, value in zip(x, table['médiane']):
            ax.text(xi + 0.18, value, f'{value:+.0f} €', va='center', fontsize=8.5, color=INK)
        ax.axhline(0, color='#c3c2b7', linewidth=1)
        ax.set_xticks(x, table.index.astype(str))
        ax.set_xlabel(xlabel, color=INK)
        plain(ax, 'y')
    axes[0].set_ylabel('Prix réel − prix prévu (€ par nuit)', color=INK)
    axes[0].legend(loc='upper left', frameon=False)
    finish(fig, 'Où le modèle se trompe-t-il, selon le prix et la taille ?',
           'Au-dessus de 0 : logements plus chers que prévu. Une erreur médiane qui dérive avec le prix ou la taille\n'
           'signale un effet que la régression linéaire ne capte pas (effet non linéaire ou variable manquante).',
           'residus_par_segment.png')


def plot_interactions(data, names):
    grid = data.assign(rooms=data['bedrooms'].clip(upper=3).map(dict(enumerate(ROOMS))))
    median = grid.pivot_table(index='arrondissement', columns='rooms', values='error', aggfunc='median')
    count = grid.pivot_table(index='arrondissement', columns='rooms', values='error', aggfunc='size')
    median = median.reindex(columns=ROOMS).where(count.reindex(columns=ROOMS) >= 30).rename_axis(columns=None)
    print('\nErreur médiane par arrondissement et nombre de chambres (€ ; vide = moins de 30 annonces)\n'
          + median.round(0).to_string())

    fig, ax = plt.subplots(figsize=(8.5, 9))
    image = ax.imshow(median, cmap=DIVERGING, norm=TwoSlopeNorm(0, -40, 40), aspect='auto')
    for (row, col), value in np.ndenumerate(median.to_numpy()):
        if not np.isnan(value):
            ax.text(col, row, f'{value:+.0f}', ha='center', va='center', fontsize=8.5,
                    color='white' if abs(value) > 25 else '#0b0b0b')
    ax.set_xticks(range(len(ROOMS)), ROOMS)
    ax.set_yticks(range(len(median)), [f'{n:02} · {names[n]}' for n in median.index])
    ax.tick_params(length=0, colors=INK)
    ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.6, label='Prix réel − prix prévu (€ par nuit, médiane)')
    finish(fig, 'Erreur selon l’arrondissement et le nombre de chambres',
           'Rouge : plus cher que prévu ; bleu : moins cher. Le modèle additionne l’effet de l’arrondissement\n'
           'et celui de la taille : une ligne ou une colonne qui change de couleur signale une interaction\n'
           'qu’il ignore. Cases vides : moins de 30 annonces.',
           'residus_arrondissement_chambres.png', top=0.87)


def main():
    data, names = out_of_fold_errors()
    figures = print_metrics(data)
    by_price = by_segment(data, pd.cut(data['predicted'], PRICE_BANDS, labels=PRICE_LABELS), 'le prix prévu')
    by_capacity = by_segment(data, data['accommodates'].clip(upper=8), 'la capacité (8 = 8 personnes et plus)')
    by_capacity.index = [str(n) if n < 8 else '8+' for n in by_capacity.index]
    by_segment(data, data['arrondissement'].map(lambda n: f'{n:02} · {names[n]}'), 'l’arrondissement')
    by_segment(data, data['property_type'], 'le type de bien')
    monument_leads(data)
    largest_errors(data, names)

    plot_distribution(data, figures)
    plot_segments(by_price, by_capacity)
    plot_interactions(data, names)


if __name__ == '__main__':
    main()
