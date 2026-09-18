"""Do monuments explain Airbnb prices that arrondissements miss? No arguments.

Checks the errors of the current price model (size, type, arrondissement) against the
monument proximity written by ingestion/build_database.py, and compares location variables.
"""

import math
import sqlite3

import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

from regression import DATABASE, GRID, INK, OUTPUT, SIZE, TARGETS, load_data, prepare_listings


CENTRE = (48.8566, 2.3522)  # Hôtel de Ville: separates "near a monument" from "central".
VARIANTS = {
    'Taille + type seulement': [],
    'Distance au centre': ['centre_km'],
    'Score monuments': ['tourist_proximity_score'],
    'Score pondéré par la fréquentation': ['weighted_score'],
    'Arrondissement (modèle actuel)': ['arrondissement'],
    'Arrondissement + distance au centre': ['arrondissement', 'centre_km'],
    'Arrondissement + score monuments': ['arrondissement', 'tourist_proximity_score'],
}
# Out-of-fold price errors kept for the charts: real minus predicted, € per night.
ERRORS = {'Arrondissement (modèle actuel)': 'error', 'Arrondissement + distance au centre': 'error_centre'}

BLUE, ORANGE, RED = '#2a78d6', '#eb6834', '#b8302f'
DIVERGING = LinearSegmentedColormap.from_list('errors', ['#184f95', '#6da7ec', '#f0efec', '#f08a89', RED])


def distance_km(lat, lon, lat0, lon0):
    """Great-circle distance, broadcast over arrays."""
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    h = np.sin((lat0 - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon0 - lon) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(h))


def load_listings():
    airbnb, _, names = load_data()
    if 'tourist_proximity_score' not in airbnb:
        raise SystemExit('Monument columns missing: rebuild the database with ingestion/build_database.py')
    data, _ = prepare_listings(airbnb)
    # prepare_listings keeps only the model columns: bring back position and monument proximity.
    data = data.join(airbnb[['latitude', 'longitude', 'nearest_monument_distance_km', 'tourist_proximity_score']])
    with sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True) as db:
        monuments = pd.read_sql_query('SELECT * FROM monuments', db)
    data['centre_km'] = distance_km(data['latitude'], data['longitude'], *CENTRE)
    # Variant: the most visited monument within reach, halved every km (visits are never summed).
    km = distance_km(data[['latitude']].to_numpy(), data[['longitude']].to_numpy(),
                     monuments['lat'].to_numpy(), monuments['long'].to_numpy())
    data['weighted_score'] = (monuments['score_importance_20'].to_numpy() * np.exp2(-km)).max(axis=1)
    return data, monuments, names


def design(frame, columns):
    categories = ['property_type'] + [c for c in columns if c == 'arrondissement']
    return pd.get_dummies(frame[SIZE + ['property_type'] + columns], columns=categories, drop_first=True, dtype=int)


def compare_models(data):
    """Quality of each set of location variables, by 5-fold cross-validation."""
    folds = KFold(5, shuffle=True, random_state=42)
    rows, scores = [], {}
    for label, target in TARGETS.items():
        kept = data[target].between(*data[target].quantile([0.01, 0.99]))  # Same trimming as regression.py.
        y = np.log(data.loc[kept, target])
        for name, columns in VARIANTS.items():
            X = design(data.loc[kept], columns)
            scores[label, name] = cross_val_score(LinearRegression(), X, y, cv=folds, scoring='r2')
            predicted = np.exp(cross_val_predict(LinearRegression(), X, y, cv=folds))
            error = (np.exp(y) - predicted).abs()
            rows.append({'cible': label, 'modèle': name, 'R²': scores[label, name].mean(),
                         'R² écart-type': scores[label, name].std(),
                         'erreur moyenne (€)': error.mean(), 'erreur médiane (€)': error.median()})
            if label == 'Prix par nuit' and name in ERRORS:
                data.loc[y.index, ERRORS[name]] = np.exp(y) - predicted
    results = pd.DataFrame(rows)
    print('\nComparaison des modèles (validation croisée, 5 plis)')
    print(results.round({'R²': 3, 'R² écart-type': 3, 'erreur moyenne (€)': 0, 'erreur médiane (€)': 0})
          .to_string(index=False))
    for label in TARGETS:
        gain = scores[label, 'Arrondissement + score monuments'] - scores[label, 'Arrondissement (modèle actuel)']
        print(f'{label} : gain de R² apporté par le score monuments, pli par pli : {np.round(gain, 3).tolist()}')
    return results


def score_effect(data):
    """Price difference between a low and a high monument score, at equal arrondissement."""
    target = TARGETS['Prix par nuit']
    kept = data[target].between(*data[target].quantile([0.01, 0.99]))
    X = design(data.loc[kept], ['arrondissement', 'tourist_proximity_score'])
    model = LinearRegression().fit(X, np.log(data.loc[kept, target]))
    low, high = data['tourist_proximity_score'].quantile([0.25, 0.75])
    base = np.median(np.exp(model.predict(X)))
    effect = base * (np.exp(model.coef_[X.columns.get_loc('tourist_proximity_score')] * (high - low)) - 1)
    print(f'\nScore monuments de {low:.0f} à {high:.0f} (1er -> 3e quartile), à arrondissement égal : '
          f'{effect:+.0f} € par nuit pour un logement prévu à {base:.0f} €')


def sign_test(successes, trials):
    """Chance of at least this many successes if each were a coin flip."""
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / 2 ** trials


def near_far(errors, column, label):
    """Median error of the nearest and farthest quarter from a monument, in each arrondissement."""
    quarter = errors.groupby('arrondissement')['nearest_monument_distance_km'].transform(
        lambda km: pd.qcut(km, 4, labels=False))
    near = errors[quarter == 0].groupby('arrondissement')[column].median()
    far = errors[quarter == 3].groupby('arrondissement')[column].median()
    gap = near - far
    dearer = int((gap > 0).sum())
    print(f'{label} : quart le plus proche - quart le plus éloigné = {gap.median():+.0f} € par nuit en médiane ; '
          f'plus cher près des monuments dans {dearer} arrondissements sur {len(gap)} '
          f'(probabilité si c’était le hasard : {sign_test(dearer, len(gap)):.5f})')
    return near, far, gap


def finish(fig, title, subtitle, filename, top=0.88):
    fig.suptitle(title, x=0.02, ha='left', fontsize=15)
    fig.text(0.02, 0.93, subtitle, ha='left', va='top', color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUTPUT / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Graphique : {OUTPUT / filename}')


def plain(ax, grid_axis):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK)
    ax.spines[['top', 'right']].set_visible(False)


def plot_map(errors, monuments):
    fig, ax = plt.subplots(figsize=(11, 7.2))
    cells = ax.hexbin(errors['longitude'], errors['latitude'], C=errors['error'], reduce_C_function=np.median,
                      gridsize=55, cmap=DIVERGING, norm=TwoSlopeNorm(0, -60, 60), mincnt=8, linewidths=0.2)
    ax.scatter(monuments['long'], monuments['lat'], s=10, color='#0b0b0b', zorder=3)
    top = monuments.nlargest(8, 'visiteurs_annuels')
    ax.scatter(top['long'], top['lat'], s=top['visiteurs_annuels'] / 3e4, facecolor='none',
               edgecolor='#0b0b0b', linewidth=1.2, zorder=3)
    for row in top[top['nom'] != 'Arc de Triomphe du Carrousel'].itertuples():  # Next to the Louvre label.
        short = row.nom.split(' – ')[0].replace('Cathédrale ', '').replace('Basilique du ', '').replace(' de Montmartre', '')
        dx, dy, ha = (-8, -14, 'right') if 'Notre-Dame' in row.nom else (7, 5, 'left')
        ax.annotate(short, (row.long, row.lat), xytext=(dx, dy), textcoords='offset points', fontsize=8.5, ha=ha,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
    ax.set_aspect(1 / np.cos(np.radians(CENTRE[0])))
    ax.set_axis_off()
    fig.colorbar(cells, ax=ax, shrink=0.7, label='Prix réel − prix prévu (€ par nuit)')
    finish(fig, 'Où le modèle « arrondissement » se trompe-t-il ?',
           'Rouge : logements plus chers que ce que prévoit le modèle actuel (taille, type, arrondissement). Bleu : moins chers.\n'
           'Points noirs : les monuments du catalogue ; cercles : les 8 plus visités (taille = fréquentation).',
           'carte_erreurs.png')


def plot_by_distance(errors):
    bins = pd.cut(errors['nearest_monument_distance_km'], [0, 0.25, 0.5, 1, 2, np.inf],
                  labels=['< 250 m', '250–500 m', '500 m–1 km', '1–2 km', '> 2 km'])
    by_distance = errors.groupby(bins, observed=True)['error'].agg(['median', 'size'])
    print('\nErreur médiane (réel − prévu, € par nuit) selon la distance au monument le plus proche\n'
          + by_distance.round(1).to_string())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(by_distance.index.astype(str), by_distance['median'], width=0.6,
           color=[RED if value > 0 else BLUE for value in by_distance['median']])
    ax.axhline(0, color='#c3c2b7', linewidth=1)
    for i, (value, count) in enumerate(zip(by_distance['median'], by_distance['size'])):
        ax.text(i, value + (1 if value >= 0 else -1), f'{value:+.0f} €\n{count:,} annonces'.replace(',', ' '),
                ha='center', va='bottom' if value >= 0 else 'top', fontsize=9, color=INK)
    ax.set_xlabel('Distance au monument le plus proche', color=INK)
    ax.set_ylabel('Prix réel − prix prévu (€ par nuit, médiane)', color=INK)
    ax.margins(y=0.25)
    plain(ax, 'y')
    finish(fig, 'Près des monuments, le modèle actuel sous-estime les prix',
           'Erreur du modèle « arrondissement » en validation croisée : au-dessus de 0, le logement est plus cher que prévu.',
           'erreur_selon_distance.png')


def plot_near_far(near, far, gap, names):
    order = gap.sort_values().index
    rows = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.hlines(rows, far[order], near[order], color=GRID, linewidth=2, zorder=1)
    for values, color, label in [(far, BLUE, 'Quart le plus éloigné d’un monument'),
                                 (near, ORANGE, 'Quart le plus proche d’un monument')]:
        ax.scatter(values[order], rows, s=70, color=color, edgecolor='white', linewidth=1.5, zorder=2, label=label)
    ax.axvline(0, color='#c3c2b7', linewidth=1)
    labels = ax.get_yaxis_transform()  # x in axes fraction, y in rows.
    ax.text(1.02, len(order) - 0.2, 'Écart', transform=labels, fontweight='bold', color=INK)
    for row, value in zip(rows, gap[order]):
        ax.text(1.02, row, f'{value:+.0f} €', transform=labels, va='center', color=INK)
    ax.set_yticks(rows, [f'{n:02} · {names[n]}' for n in order])
    ax.set_xlabel('Prix réel − prix prévu par le modèle actuel (€ par nuit, médiane)', color=INK)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=2, frameon=False)
    plain(ax, 'x')
    finish(fig, f'Même arrondissement, plus près d’un monument : plus cher que prévu dans {int((gap > 0).sum())} cas sur 20',
           'Dans chaque arrondissement, les annonces sont coupées en 4 selon leur distance au monument le plus proche.\n'
           'On compare l’erreur du modèle actuel pour le quart le plus proche et le quart le plus éloigné.',
           'proche_loin_monument.png')


def plot_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, (label, group) in zip(axes, results.groupby('cible', sort=False)):
        group = group.iloc[::-1]
        unit = '€' if label == 'Prix par nuit' else '€ / an'
        ax.barh(group['modèle'], group['erreur moyenne (€)'], color=BLUE, height=0.6)
        for row, (error, r2) in enumerate(zip(group['erreur moyenne (€)'], group['R²'])):
            ax.text(error, row, f'  {error:,.0f} {unit} · R² {r2:.2f}'.replace(',', ' '),
                    va='center', fontsize=9, color=INK)
        ax.set_title(label, loc='left', fontsize=11)
        ax.set_xlabel(f'Erreur moyenne ({unit}, moins = mieux)', color=INK)
        ax.margins(x=0.35)
        plain(ax, 'x')
    finish(fig, 'Monuments ou arrondissements : quel lieu explique le mieux ?',
           'Validation croisée sur 5 plis. Tous les modèles ont aussi la taille et le type de bien.',
           'comparaison_modeles.png', top=0.86)


def main():
    data, monuments, names = load_listings()
    results = compare_models(data)
    score_effect(data)

    errors = data.dropna(subset=['error'])
    print()
    near, far, gap = near_far(errors, 'error', 'Modèle actuel')
    near_far(errors, 'error_centre', 'Modèle actuel + distance au centre')  # Rules out a plain centrality effect.
    under = errors['error'] > 50
    print(f"Sous-estimés de plus de 50 € par le modèle actuel : {under.sum():,} annonces, à "
          f"{errors.loc[under, 'nearest_monument_distance_km'].median() * 1000:.0f} m d'un monument en médiane, "
          f"contre {errors.loc[~under, 'nearest_monument_distance_km'].median() * 1000:.0f} m pour les autres")

    plot_map(errors, monuments)
    plot_by_distance(errors)
    plot_near_far(near, far, gap, names)
    plot_comparison(results)


if __name__ == '__main__':
    main()
