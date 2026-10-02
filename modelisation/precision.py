"""Can a specific flat be priced more precisely? Tests exact position and surface. No arguments.

Price models of the final folders (enriched linear, gradient boosting), with:
- exact position: median price (and nights booked) of the 20 nearest listings run by other
  hosts, plus latitude and longitude for the boosting;
- surface in m², read from the description (about 3 listings in 10 give it).
Folds keep every host's listings together (GroupKFold): otherwise a host's near-identical flats
next door would hand the answer to the neighbour feature.
Revenue = mean predicted nightly price x predicted nights booked (revenu.py's nights model).
"""

import importlib.util
import re

import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree

from monuments import BLUE, ORANGE, finish, plain
from regression import INK, ROOT, load_data, prepare_listings
from residus import metrics as price_metrics
from revenu import NIGHTS, PRICE, REVENUE, night_features, nights_model, revenue_metrics, show


SURFACE = re.compile(r'(\d{2,3}(?:[.,]\d)?)\s?(?:m²|m2|sqm|sq\.?\s?m\b|square met(?:er|re)s?|mètres? carrés?)', re.I)
NEIGHBOURS = 20


def folder_features(folder):
    """The features() of a final model folder (both files are named entrainement.py)."""
    spec = importlib.util.spec_from_file_location(folder, ROOT / folder / 'entrainement.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.features


def step_4a(features):
    """The enriched linear inputs as compared at step 4a, before the minimum stay was added to the model."""
    return lambda frame, reference: features(frame, reference).drop(columns='minimum_nights', errors='ignore')


LINEAR, BOOSTING = step_4a(folder_features('modele_lineaire_enrichi')), folder_features('modele_gradient_boosting')


def load():
    airbnb, _, _ = load_data()
    data, _ = prepare_listings(airbnb)
    data = data[data[PRICE].between(*data[PRICE].quantile([0.01, 0.99]))]  # Same trimming as the final models.
    data = data.join(airbnb[['tourist_proximity_score', NIGHTS, 'calculated_host_listings_count', 'minimum_nights',
                             'latitude', 'longitude', 'host_id', 'description']])
    surface = data['description'].str.extract(SURFACE)[0].str.replace(',', '.').astype(float)
    data['surface'] = surface.where(surface.between(9, 300))
    data['log_price'] = np.log(data[PRICE])
    data['host_id'] = data['host_id'].fillna(-1).astype('int64')  # Plain integers for fast host comparisons.
    return data


def neighbours(train, target, column):
    """Median `column` of the nearest training listings run by another host than the target's."""
    tree = BallTree(np.radians(train[['latitude', 'longitude']].to_numpy()), metric='haversine')
    _, nearest = tree.query(np.radians(target[['latitude', 'longitude']].to_numpy()), k=min(5 * NEIGHBOURS, len(train)))
    other_host = train['host_id'].to_numpy()[nearest] != target['host_id'].to_numpy()[:, None]
    values = train[column].to_numpy()[nearest]
    return np.array([np.median(v[keep][:NEIGHBOURS]) if keep.any() else np.nan for v, keep in zip(values, other_host)])


def design(base, extra):
    """Train and test inputs of a final model, plus the extra information being tested."""
    def build(train, test, types):
        X_train = base(train, {'types': types})
        X_test = base(test, {'types': types, 'columns': list(X_train.columns)})
        if 'voisins' in extra:  # Neighbours of the training rows never include their own host.
            X_train['voisins'], X_test['voisins'] = neighbours(train, train, 'log_price'), neighbours(train, test, 'log_price')
            fill = X_train['voisins'].median()
            X_train['voisins'], X_test['voisins'] = X_train['voisins'].fillna(fill), X_test['voisins'].fillna(fill)
        if 'surface' in extra:
            X_train['log_surface'], X_test['log_surface'] = np.log(train['surface']), np.log(test['surface'])
        if 'coordonnées' in extra:
            for column in ['latitude', 'longitude']:
                X_train[column], X_test[column] = train[column], test[column]
        return X_train, X_test
    return build


def boosting():
    return HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=42)


def evaluate(data, build, make_model, folds, types):
    """Typical and mean nightly price of every listing, from folds that keep each host together."""
    typical, mean = pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)
    for train_rows, test_rows in folds:
        train, test = data.iloc[train_rows], data.iloc[test_rows]
        X_train, X_test = build(train, test, types)
        model = make_model().fit(X_train, train['log_price'])
        duan = np.exp(train['log_price'] - model.predict(X_train)).mean()
        typical.iloc[test_rows] = np.exp(model.predict(X_test))
        mean.iloc[test_rows] = typical.iloc[test_rows] * duan
    return typical, mean


def nights(data, folds, types, position):
    """Nights booked of every listing; with position, also its neighbours' nights and coordinates."""
    predicted = pd.Series(np.nan, index=data.index)
    for train_rows, test_rows in folds:
        train, test = data.iloc[train_rows], data.iloc[test_rows]
        X_train, X_test = night_features(train, types), night_features(test, types)
        if position:
            X_train['voisins'], X_test['voisins'] = neighbours(train, train, NIGHTS), neighbours(train, test, NIGHTS)
            for column in ['latitude', 'longitude']:
                X_train[column], X_test[column] = train[column], test[column]
        predicted.iloc[test_rows] = nights_model().fit(X_train, train[NIGHTS]).predict(X_test)
    return predicted.clip(0, 365)


def compare(title, data, variants, night_options, types):
    """Price and revenue errors of each variant, on one population of listings."""
    folds = list(GroupKFold(5).split(data, groups=data['host_id']))
    prices, revenues = {}, {}
    for name, (build, make_model, position) in variants.items():
        typical, mean = evaluate(data, build, make_model, folds, types)
        prices[name] = price_metrics(data[PRICE], typical)
        revenues[name] = revenue_metrics(data[REVENUE], mean * night_options[position].loc[data.index])
    prices = show(f'{title} · prix par nuit (€, 5 plis, chaque hôte dans un seul pli)', prices)
    revenues = show(f'{title} · revenu annuel = prix moyen prévu × nuits prévues (€/an)', revenues)
    return prices, revenues


def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), gridspec_kw={'height_ratios': [4, 5]})
    for row, (population, (prices, revenues)) in enumerate(results.items()):
        for col, (table, column, scale, unit) in enumerate([(prices, 'erreur moyenne (€)', 1, '€'),
                                                            (revenues, 'erreur moyenne (€/an)', 1000, 'k€')]):
            ax = axes[row, col]
            values = table[column].astype(float) / scale
            colors = [BLUE if '(actuel)' in name else ORANGE for name in table.index]
            ax.barh(table.index, values, color=colors, height=0.6)
            for y, value in enumerate(values):
                ax.text(value, y, f' {value:.0f} {unit}' if unit == '€' else f' {value:.1f} {unit}'.replace('.', ','),
                        va='center', fontsize=9, color=INK)
            ax.invert_yaxis()
            ax.margins(x=0.15)
            ax.set_title(f"{'Prix par nuit' if col == 0 else 'Revenu annuel'} · {population}", loc='left', fontsize=11)
            ax.set_xlabel(f"Erreur moyenne ({unit}{'' if col == 0 else ' par an'}, moins = mieux)", color=INK)
            plain(ax, 'x')
    finish(fig, 'Un logement précis : la position exacte et la surface réduisent-elles l’erreur ?',
           'Bleu : modèles actuels ; orange : avec les nouvelles informations. Validation croisée sur 5 plis, chaque hôte dans un seul pli.\n'
           'Voisins : prix (et nuits) médians des 20 annonces les plus proches d’autres hôtes. Surface lue dans les descriptions.',
           'precision_logement.png', top=0.88)


def main():
    data = load()
    types = sorted(data['property_type'].unique())
    with_surface = data.dropna(subset=['surface'])
    print(f"\nSurface trouvée dans {len(with_surface):,} descriptions sur {len(data):,} "
          f"(médiane {with_surface['surface'].median():.0f} m²)")

    folds = list(GroupKFold(5).split(data, groups=data['host_id']))
    night_options = {False: nights(data, folds, types, False), True: nights(data, folds, types, True)}
    print('\nNuits louées par an (5 plis, chaque hôte dans un seul pli)\n' + pd.DataFrame({
        label: {'R²': r2_score(data[NIGHTS], n), 'erreur moyenne (nuits)': mean_absolute_error(data[NIGHTS], n)}
        for label, n in [('Modèle de nuits (actuel)', night_options[False]),
                         ('Modèle de nuits + position', night_options[True])]}).T.round(2).to_string())

    linear, boost = LinearRegression, boosting
    results = {
        'toutes les annonces': compare('Toutes les annonces', data, {
            'Linéaire enrichi (actuel)': (design(LINEAR, []), linear, False),
            'Linéaire enrichi + voisins': (design(LINEAR, ['voisins']), linear, True),
            'Gradient boosting (actuel)': (design(BOOSTING, []), boost, False),
            'Gradient boosting + position': (design(BOOSTING, ['voisins', 'coordonnées']), boost, True),
        }, night_options, types),
        'annonces avec surface': compare('Annonces dont la surface est connue', with_surface, {
            'Linéaire enrichi (actuel)': (design(LINEAR, []), linear, False),
            'Linéaire enrichi + surface': (design(LINEAR, ['surface']), linear, False),
            'Linéaire enrichi + surface + voisins': (design(LINEAR, ['surface', 'voisins']), linear, True),
            'Gradient boosting (actuel)': (design(BOOSTING, []), boost, False),
            'Gradient boosting + surface + position': (design(BOOSTING, ['surface', 'voisins', 'coordonnées']), boost, True),
        }, night_options, types),
    }
    plot(results)


if __name__ == '__main__':
    main()
