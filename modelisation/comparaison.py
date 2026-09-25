"""Compare the current price model with an enriched linear model and tree models. No arguments.

Enriched: capacity and bedrooms as categories, monument score, Duan correction.
Tree models (decision tree, random forest, gradient boosting): the same information, left to
the trees (no size categories to build by hand). Same listings, same 5 folds and same error
figures (residus.metrics) for every model; also draws the first questions of the decision tree.
"""

import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree

from monuments import BLUE, ORANGE, finish, plain
from regression import BLUES, INK, encode, load_data, prepare_listings
from residus import TARGET, metrics


FOLDS = KFold(5, shuffle=True, random_state=42)  # Same folds as residus.py.
CAPACITY = ['1', '2', '3', '4', '5', '6', '7', '8+']
ROOMS = ['0', '1', '2', '3', '4+']
ROOM_LABELS = ['Studio', '1 ch.', '2 ch.', '3 ch.', '4 ch. et +']


def enriched(frame):
    """Capacity and bedrooms as categories, plus the monument score (baselines: 2 guests, 1 bedroom)."""
    X = pd.get_dummies(frame[['bathrooms', 'tourist_proximity_score', 'capacity', 'rooms', 'arrondissement', 'property_type']],
                       columns=['capacity', 'rooms', 'arrondissement', 'property_type'], dtype=int)
    return X.drop(columns=['capacity_2', 'rooms_1', 'arrondissement_11', 'property_type_Entire rental unit'])


def raw(frame):
    """Same information for the trees: sizes stay numbers, categories stay categories (no one-hot)."""
    X = frame[['accommodates', 'bedrooms', 'bathrooms', 'tourist_proximity_score', 'arrondissement', 'property_type']].copy()
    X[['arrondissement', 'property_type']] = X[['arrondissement', 'property_type']].astype('category')
    return X


def boosting():
    return HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=42)


def dummies(frame):
    """Same information for scikit-learn's single trees and forests, which need 0/1 columns."""
    return pd.get_dummies(raw(frame), columns=['arrondissement', 'property_type'], dtype=int)


def decision_tree():
    # Depth 12 and 20 listings per leaf: deeper trees stop improving in cross-validation.
    return DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42)


def forest():
    # At least 5 listings per leaf: fully grown trees learn the noise of individual prices.
    return RandomForestRegressor(min_samples_leaf=5, n_jobs=-1, random_state=42)


# Name: (features, model, Duan correction). All models learn the log of the price.
MODELS = {
    'Modèle actuel': (encode, LinearRegression, False),
    'Linéaire enrichi sans Duan': (enriched, LinearRegression, False),
    'Gradient boosting': (raw, boosting, False),
    'Linéaire enrichi + Duan': (enriched, LinearRegression, True),
    'Arbre de décision': (dummies, decision_tree, False),
    'Forêt aléatoire': (dummies, forest, False),
}
# Fixed categorical order; the size chart shows only the first three (all-pairs safe).
COLORS = dict(zip(MODELS, [BLUE, ORANGE, '#1baf7a', '#eda100', '#e87ba4', '#008300']))
TYPE_NAMES = {'Entire rental unit': 'un appartement classique', 'Entire condo': 'un appartement en copropriété',
              'Entire loft': 'un loft', 'Entire home': 'une maison', 'Entire serviced apartment': 'un appart-hôtel',
              'Entire townhouse': 'une maison de ville', 'Autre': 'un autre type de bien'}


def load():
    airbnb, _, _ = load_data()
    data, _ = prepare_listings(airbnb)
    data = data[data[TARGET].between(*data[TARGET].quantile([0.01, 0.99]))]  # Same trimming as regression.py.
    data = data.join(airbnb['tourist_proximity_score'])
    data['capacity'] = data['accommodates'].clip(upper=8).astype(int).astype(str).replace('8', '8+')
    data['rooms'] = data['bedrooms'].clip(upper=4).astype(int).astype(str).replace('4', '4+')
    return data


def out_of_fold(X, y, estimator, duan):
    """Price of every listing predicted by the folds that exclude it, in euros."""
    predicted = pd.Series(np.nan, index=y.index)
    for train, test in FOLDS.split(X):
        model = estimator().fit(X.iloc[train], y.iloc[train])
        # exp(log prediction) is a median; Duan's factor (mean training ratio) turns it into a mean.
        factor = np.exp(y.iloc[train] - model.predict(X.iloc[train])).mean() if duan else 1
        predicted.iloc[test] = np.exp(model.predict(X.iloc[test])) * factor
    return predicted


def compare(data):
    y = np.log(data[TARGET])
    predictions = pd.DataFrame({name: out_of_fold(features(data), y, estimator, duan)
                                for name, (features, estimator, duan) in MODELS.items()})
    table = pd.DataFrame({name: metrics(data[TARGET], predictions[name]) for name in MODELS}).T.round(3)
    counts = table.columns.drop('R² (log)')
    table[counts] = table[counts].round().astype(int)
    print('\nChiffres d’erreur, € par nuit (mêmes annonces, mêmes 5 plis)\n' + table.astype(object).T.to_string())

    fold = pd.Series(0, index=data.index)
    for number, (_, test) in enumerate(FOLDS.split(data), 1):
        fold.iloc[test] = number
    errors = predictions.rsub(data[TARGET], axis=0)  # Real minus predicted.
    by_fold = errors.abs().groupby(fold).mean()
    print('\nErreur moyenne par pli (€)\n' + by_fold.round(1).to_string())
    for better, worse in [('Linéaire enrichi sans Duan', 'Modèle actuel'),
                          ('Gradient boosting', 'Linéaire enrichi sans Duan'),
                          ('Forêt aléatoire', 'Arbre de décision'),
                          ('Gradient boosting', 'Forêt aléatoire'),
                          ('Linéaire enrichi sans Duan', 'Forêt aléatoire')]:
        wins = int((by_fold[better] < by_fold[worse]).sum())
        print(f'{better} fait mieux que {worse} dans {wins} plis sur 5')
    return table, errors


def by_category(data, errors, column, order, labels, title):
    table = errors.groupby(data[column]).median().reindex(order)
    table.index = labels
    table.insert(0, 'annonces', data[column].value_counts().reindex(order).to_numpy())
    print(f'\nErreur médiane selon {title} (réel − prévu, € par nuit)\n' + table.round(0).to_string())
    return table.drop(columns='annonces')


def print_effects(data):
    """Enriched model read in euros, around a reference flat."""
    X = enriched(data)
    model = LinearRegression().fit(X, np.log(data[TARGET]))
    coef = pd.Series(model.coef_, index=X.columns)
    low, median, high = data['tourist_proximity_score'].quantile([0.25, 0.5, 0.75])
    base = np.exp(model.intercept_ + coef['bathrooms'] + coef['tourist_proximity_score'] * median)
    effects = {f'capacité {c} (au lieu de 2)': coef[f'capacity_{c}'] for c in CAPACITY if c != '2'}
    effects |= {f'{label} (au lieu de 1 ch.)': coef[f'rooms_{r}'] for r, label in zip(ROOMS, ROOM_LABELS) if r != '1'}
    effects['une salle de bain de plus'] = coef['bathrooms']
    effects[f'score monuments {low:.0f} → {high:.0f}'] = coef['tourist_proximity_score'] * (high - low)
    euros = pd.Series({name: base * (np.exp(value) - 1) for name, value in effects.items()})
    print(f'\nLinéaire enrichi : un logement de référence (2 pers., 1 chambre, 1 salle de bain, 11e, score monuments '
          f'médian) vaut {base:.0f} € la nuit (prix médian). Écart pour :\n' + euros.round(0).to_string())


def plot_errors(table):
    panels = [(['erreur moyenne (€)', 'erreur médiane (€)', 'RMSE (€)', 'biais moyen (€)'], '€ par nuit', '{:.0f} €'),
              (['à 25 € près', 'à 50 € près'], 'annonces', '{:,.0f}')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), gridspec_kw={'width_ratios': [2, 1.2]})
    step = 0.84 / len(MODELS)  # One bar per model inside each group, with a small gap between bars.
    for ax, (columns, xlabel, fmt) in zip(axes, panels):
        rows = np.arange(len(columns))
        for i, name in enumerate(MODELS):
            values = table.loc[name, columns].astype(float)
            offset = (i - (len(MODELS) - 1) / 2) * step
            ax.barh(rows + offset, values, height=step * 0.9, color=COLORS[name], label=name)
            for row, value in zip(rows, values):
                ax.text(value, row + offset, ' ' + fmt.format(value).replace(',', ' '),
                        va='center', fontsize=8.5, color=INK)
        ax.set_yticks(rows, [c.replace(' (€)', '') for c in columns])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, color=INK)
        ax.margins(x=0.18)
        plain(ax, 'x')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.01, 0.89),
               ncols=3, frameon=False)
    finish(fig, 'Quel modèle se trompe le moins ? Les erreurs en chiffres',
           'Mêmes annonces, chacune prévue par un modèle qui ne l’a jamais vue (validation croisée sur 5 plis).\n'
           'À gauche, moins = mieux (biais : proche de 0 = mieux) ; à droite, plus = mieux.',
           'comparaison_erreurs.png', top=0.83)


def plot_by_size(by_capacity, by_rooms):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True,
                             gridspec_kw={'width_ratios': [len(by_capacity), len(by_rooms)]})
    for ax, table, xlabel in [(axes[0], by_capacity, 'Capacité (personnes)'), (axes[1], by_rooms, 'Chambres')]:
        x = np.arange(len(table))
        for offset, name in zip([-0.22, 0, 0.22], list(MODELS)[:3]):  # Models that predict the typical price.
            ax.scatter(x + offset, table[name], s=55, color=COLORS[name], edgecolor='white', linewidth=1.2,
                       zorder=3, label=name)
            if name != 'Modèle actuel':
                for xi, value in zip(x, table[name]):
                    ax.text(xi + offset, value + 4, f'{value:+.0f}', ha='center', va='bottom', fontsize=7.5, color=INK)
        ax.axhline(0, color='#c3c2b7', linewidth=1)
        ax.set_xticks(x, table.index)
        ax.set_xlabel(xlabel, color=INK)
        plain(ax, 'y')
    axes[0].set_ylabel('Prix réel − prix prévu (€ par nuit, médiane)', color=INK)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=3, frameon=False)
    finish(fig, 'Les nouveaux modèles corrigent-ils les erreurs sur les grands logements ?',
           'Erreur médiane (€) par capacité et par nombre de chambres : sur la ligne 0, le logement typique est bien prévu.\n'
           'Le linéaire enrichi + Duan n’est pas affiché : il relève toutes les prévisions d’environ 6 %.',
           'comparaison_par_taille.png', top=0.86)


def question(split):
    """Readable question from scikit-learn's 'feature <= threshold' split text."""
    feature, threshold = split.rsplit(' <= ', 1)
    threshold = float(threshold)
    if feature.startswith('arrondissement_'):  # 0/1 column: "<= 0.5" means "not this one".
        return f'pas dans le {feature.split("_")[1]}e ?'
    if feature.startswith('property_type_'):
        return f'pas {TYPE_NAMES[feature.split("_", 2)[2]]} ?'
    if feature == 'tourist_proximity_score':
        return f'score monuments ≤ {threshold:.0f} ?'
    if feature == 'bathrooms':
        return f'{np.floor(threshold * 2) / 2:g} salle(s) de bain ou moins ?'
    return f'{int(threshold)} {"personne(s)" if feature == "accommodates" else "chambre(s)"} ou moins ?'


def plot_decision_tree(data):
    """First questions of the decision tree fitted on every listing, with prices in euros."""
    X = dummies(data)
    tree = decision_tree().fit(X, np.log(data[TARGET]))
    fig, ax = plt.subplots(figsize=(17, 8))
    plot_tree(tree, max_depth=2, feature_names=list(X.columns), impurity=False, rounded=True,
              precision=4, fontsize=10, ax=ax)
    values = {}
    for node in ax.texts:
        text = node.get_text()
        if text.strip() in ('True', 'False'):
            node.set_text('oui' if text.strip() == 'True' else 'non')
        elif '(...)' in text:  # Levels not drawn: keep them quiet.
            node.get_bbox_patch().set_facecolor('#f0efec')
        else:
            lines = text.split('\n')
            values[node] = float(lines[-1].split('= ')[1])
            samples = int(lines[-2].split('= ')[1])
            node.set_text('\n'.join(([question(lines[0])] if ' <= ' in lines[0] else [])
                                    + [f'{np.exp(values[node]):.0f} € la nuit',
                                       f'{samples:,} annonces'.replace(',', ' ')]))
    shade = Normalize(min(values.values()), max(values.values()))
    for node, value in values.items():
        node.get_bbox_patch().set_facecolor(BLUES(shade(value)))
        node.set_color('white' if shade(value) > 0.55 else '#0b0b0b')
    finish(fig, 'Les premières questions de l’arbre de décision',
           'Arbre ajusté sur toutes les annonces (profondeur 12, au moins 20 annonces par feuille) ; seuls ses 3 premiers\n'
           'niveaux sont dessinés. « oui » part à gauche. Prix typique des annonces de chaque case : plus foncé = plus cher.',
           'arbre_decision.png')


def main():
    data = load()
    table, errors = compare(data)
    by_capacity = by_category(data, errors, 'capacity', CAPACITY, CAPACITY, 'la capacité')
    by_rooms = by_category(data, errors, 'rooms', ROOMS, ROOM_LABELS, 'le nombre de chambres')
    print_effects(data)
    plot_errors(table)
    plot_by_size(by_capacity, by_rooms)
    plot_decision_tree(data)


if __name__ == '__main__':
    main()
