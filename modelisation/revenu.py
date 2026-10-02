"""Shared pipeline of the final models: modele_lineaire_enrichi/ and modele_gradient_boosting/.

Each model folder only defines its nightly price features and its scikit-learn model. This module
adds the same nights-booked model to both (tested in occupation.py), trains them (5-fold error
figures, then every listing), saves them, and turns them into results and charts.
Annual revenue = mean predicted nightly price x predicted nights booked.
"""

import joblib
import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from monuments import BLUE, ORANGE, finish, plain
from regression import (AIRBNB_FEE, BLUES, CHARGES_EUR_M2, FIXED_EUR, FURNISHING_EUR_M2, GRID, INK, NOTARY_FEES,
                        SUPPLIES, TARGETS, TYPOLOGIES, dot_plot, load_data, prepare_listings, purchase_prices)
from residus import metrics as price_metrics


PRICE, REVENUE = TARGETS['Prix par nuit'], TARGETS['Revenu annuel']
NIGHTS = 'estimated_occupancy_l365d'
HOST_LISTINGS = 'calculated_host_listings_count'
NIGHT_FEATURES = ['bedrooms', 'accommodates', 'bathrooms', 'tourist_proximity_score', HOST_LISTINGS, 'minimum_nights']
FOLDS = KFold(5, shuffle=True, random_state=42)  # Same folds as the analyses.
MODEL_FILE = 'modele.joblib'

# Management scenarios (assumptions). Hosts running several flats book more nights; a conciergerie
# takes a share of the revenue, on top of the costs of modelisation/regression.py.
SCENARIOS = ['Gestion seul', 'Conciergerie']
MINIMUM_NIGHTS = 2
CONCIERGERIE_FEE = 0.20


def load_listings():
    airbnb, sales, names = load_data()
    data, occupancy = prepare_listings(airbnb)
    data = data[data[PRICE].between(*data[PRICE].quantile([0.01, 0.99]))]  # Same trimming as the analyses.
    return data.join(airbnb[['tourist_proximity_score', NIGHTS, HOST_LISTINGS, 'minimum_nights']]), occupancy, sales, names


def night_features(frame, types):
    """Flat and management; categories keep the training codes."""
    X = frame[NIGHT_FEATURES].astype(float)
    X['arrondissement'] = pd.Categorical(frame['arrondissement'], categories=range(1, 21))
    X['property_type'] = pd.Categorical(frame['property_type'], categories=types)
    return X


def nights_model():
    return HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=42)


def predict(bundle, frame, features):
    """Typical nightly price (to charge) and mean nightly price (Duan, to multiply into revenues)."""
    typical = np.exp(bundle['model'].predict(features(frame, bundle)))
    return typical, typical * bundle['duan']


def predict_nights(bundle, frame):
    return bundle['nights_model'].predict(night_features(frame, bundle['types'])).clip(0, 365)


def revenue_metrics(actual, predicted):
    error = actual - predicted
    return {
        'R² (log)': r2_score(np.log(actual), np.log(predicted)),
        'erreur moyenne (€/an)': error.abs().mean(),
        'erreur médiane (€/an)': error.abs().median(),
        'RMSE (€/an)': np.sqrt((error ** 2).mean()),
        'biais moyen (€/an)': error.mean(),
        'à 5 000 € près': int((error.abs() <= 5000).sum()),
        'à 10 000 € près': int((error.abs() <= 10000).sum()),
    }


def out_of_fold(make_model, features, data, types):
    """Typical and mean nightly price of every listing, from the folds that exclude it."""
    X, y = features(data, {'types': types}), np.log(data[PRICE])
    typical, mean = pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)
    for train_rows, test_rows in FOLDS.split(X):
        model = make_model().fit(X.iloc[train_rows], y.iloc[train_rows])
        duan = np.exp(y.iloc[train_rows] - model.predict(X.iloc[train_rows])).mean()
        typical.iloc[test_rows] = np.exp(model.predict(X.iloc[test_rows]))
        mean.iloc[test_rows] = typical.iloc[test_rows] * duan
    return typical, mean


def out_of_fold_nights(data, types):
    """Nights booked of every listing, from the folds that exclude it."""
    X, predicted = night_features(data, types), pd.Series(np.nan, index=data.index)
    for train_rows, test_rows in FOLDS.split(X):
        model = nights_model().fit(X.iloc[train_rows], data[NIGHTS].iloc[train_rows])
        predicted.iloc[test_rows] = model.predict(X.iloc[test_rows])
    return predicted.clip(0, 365)


def show(title, rows):
    table = pd.DataFrame(rows).T.round(3)
    counts = table.columns.drop('R² (log)')
    table[counts] = table[counts].round().astype(int)
    print(f'\n{title}\n' + table.astype(object).T.to_string())
    return table


def train(name, make_model, features, folder):
    """Cross-validated error figures, then both models trained on every listing, saved in folder/modeles/."""
    data, occupancy, _, _ = load_listings()
    types = sorted(data['property_type'].unique())
    typical, mean = out_of_fold(make_model, features, data, types)
    nights, median = out_of_fold_nights(data, types), data['arrondissement'].map(occupancy)
    prices = show(f'{name} · prix par nuit typique (€, validation croisée sur 5 plis)',
                  {name: price_metrics(data[PRICE], typical)})
    quality = pd.DataFrame({label: {'R²': r2_score(data[NIGHTS], n), 'erreur moyenne (nuits)': mean_absolute_error(data[NIGHTS], n)}
                            for label, n in [('Nuits médianes de l’arrondissement (ancien calcul)', median),
                                             ('Modèle de nuits (logement + gestion)', nights)]}).T
    print('\nNuits louées par an (validation croisée sur 5 plis)\n' + quality.round(1).to_string())
    main = f'{name} × nuits prévues'
    revenue = show('Revenu annuel = prix moyen prévu × nuits louées (€/an)',
                   {main: revenue_metrics(data[REVENUE], mean * nights),
                    f'{name} × nuits médianes (ancien calcul)': revenue_metrics(data[REVENUE], mean * median),
                    'Prix réel × nuits prévues': revenue_metrics(data[REVENUE], data[PRICE] * nights)})

    X, y = features(data, {'types': types}), np.log(data[PRICE])
    model = make_model().fit(X, y)
    bundle = {'name': name, 'model': model, 'duan': np.exp(y - model.predict(X)).mean(),
              'nights_model': nights_model().fit(night_features(data, types), data[NIGHTS]),
              'columns': list(X.columns), 'types': types,
              'scores': {'prix': prices.loc[name].to_dict(), 'revenu': revenue.loc[main].to_dict()},
              'sklearn': sklearn.__version__}
    for sub in ('modeles', 'resultats', 'graphiques'):
        (folder / sub).mkdir(exist_ok=True)
    joblib.dump(bundle, folder / 'modeles' / MODEL_FILE)
    print(f'\nModèles sauvegardés (prix et nuits) : {folder / "modeles" / MODEL_FILE}')
    prices.to_csv(folder / 'resultats' / 'erreurs_prix.csv')
    revenue.to_csv(folder / 'resultats' / 'erreurs_revenu.csv')
    plot_quality(data, name, typical, mean * nights, folder / 'graphiques' / 'qualite_modele.png')


def plot_quality(data, name, typical, revenue, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [('Prix par nuit', data[PRICE], typical), ('Revenu annuel (prix × nuits prévues)', data[REVENUE], revenue)]
    for ax, (label, actual, predicted) in zip(axes, panels):
        cells = ax.hexbin(actual, predicted, gridsize=35, xscale='log', yscale='log', cmap=BLUES, mincnt=1, linewidths=0.2)
        ends = [actual.min(), actual.max()]
        ax.plot(ends, ends, color='#898781', linewidth=1)
        for axis in [ax.xaxis, ax.yaxis]:  # Plain amounts (100 €, 300 €, 1 k€...) instead of powers of ten.
            axis.set_major_locator(LogLocator(subs=(1, 3)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g} k€' if v >= 1000 else f'{v:.0f} €'))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(f'{label} · R² {r2_score(np.log(actual), np.log(predicted)):.2f}', loc='left', fontsize=11)
        ax.set_xlabel('Valeur réelle', color=INK)
        ax.set_ylabel('Valeur prévue', color=INK)
        ax.tick_params(colors=INK)
        fig.colorbar(cells, ax=ax, label='annonces')
    finish(fig, f'Qualité du modèle : {name}',
           'Annonces prévues par des modèles qui ne les ont jamais vues. Sur la diagonale grise, la prévision est exacte.\n'
           'Le prix est bien suivi ; le revenu moins, car les nuits louées restent difficiles à prévoir pour une annonce isolée.',
           path, top=0.86)


def report(features, folder):
    """Price, nights, revenue, payback and monument effect of standard flats, from the saved models."""
    path = folder / 'modeles' / MODEL_FILE
    if not path.exists():
        raise SystemExit(f'Aucun modèle sauvegardé : lancer d’abord {folder.name}/entrainement.py')
    bundle = joblib.load(path)
    if 'nights_model' not in bundle:
        raise SystemExit(f'Modèle sauvegardé avant l’ajout du modèle de nuits : relancer {folder.name}/entrainement.py')
    if bundle['sklearn'] != sklearn.__version__:
        print(f"Attention : modèle entraîné avec scikit-learn {bundle['sklearn']}, installé {sklearn.__version__} : "
              f'relancer {folder.name}/entrainement.py.')
    per_year = f"{bundle['scores']['revenu']['erreur moyenne (€/an)']:,.0f}".replace(',', ' ')
    print(f"Modèle : {bundle['name']} (erreur moyenne en validation croisée : "
          f"{bundle['scores']['prix']['erreur moyenne (€)']:.0f} € par nuit, {per_year} € par an)")

    data, _, sales, names = load_listings()
    # Conciergerie: the median number of flats of hosts running more than one.
    listings = {'Gestion seul': 1, 'Conciergerie': int(data.loc[data[HOST_LISTINGS] > 1, HOST_LISTINGS].median())}
    flats = estimate(bundle, features, data, sales, names, listings)
    save_tables(flats, folder / 'resultats' / 'logements_types.csv')
    effect = monument_effect(bundle, features, data, flats)

    charts, name = folder / 'graphiques', bundle['name']
    alone = flats[flats['scenario'] == 'Gestion seul']
    guests = alone.groupby('typology')['accommodates'].first()
    flat_types = (f"studio {guests['Studio']:.0f} pers., T2 {guests['T2']:.0f} pers. et 1 chambre, "
                  f"T3 {guests['T3']:.0f} pers. et 2 chambres")
    descriptions = {
        'Gestion seul': f'Gestion seul : hôte avec 1 logement, {MINIMUM_NIGHTS} nuits minimum.',
        'Conciergerie': (f"Conciergerie : hôte gérant {listings['Conciergerie']} logements (médiane des hôtes multi-logements), "
                         f'{MINIMUM_NIGHTS} nuits minimum, commission de {CONCIERGERIE_FEE:.0%} du revenu.'),
    }
    # dot_plot saves next to modelisation/regression.py unless given an absolute path.
    dot_plot(by_area(alone, 'prix_nuit').sort_values('T2', ascending=False), '{:.0f} €', 'Prix par nuit (€)',
             f'Quel prix pratiquer ? Prix par nuit prévu · {name}',
             f'Prix typique pour un appartement type : {flat_types}, avec le score monuments médian\n'
             'de l’arrondissement. Devis de fin juin 2026 (haute saison).',
             str(charts / 'prix_par_nuit.png'))
    for scenario in SCENARIOS:
        rows, slug = flats[flats['scenario'] == scenario], scenario.lower().replace(' ', '_')
        dot_plot(by_area(rows, 'revenu').div(1000).sort_values('T2', ascending=False), '{:.0f} k€',
                 'Revenu annuel prévu (k€)', f'Combien rapporte un logement par an ? · {name} · {scenario}',
                 f'Prix moyen prévu × nuits louées prévues.\n{descriptions[scenario]}',
                 str(charts / f'revenu_annuel_{slug}.png'))
        dot_plot(by_area(rows, 'annees').sort_values('T2'), '{:.0f} ans', 'Années de gain net pour rembourser l’achat',
                 f'Où l’achat se rembourse-t-il le plus vite ? · {name} · {scenario}',
                 f'{descriptions[scenario]}\nAchat (prix DVF médian + notaire + ameublement) ÷ gain net annuel. '
                 'Moins d’années = meilleur retour. Réglementation non appliquée : c’est un minimum.',
                 str(charts / f'annees_remboursement_{slug}.png'))
    plot_monuments(effect, name, charts / 'effet_monuments.png')


def by_area(flats, column):
    return flats.pivot(index='label', columns='typology', values=column)


def estimate(bundle, features, data, sales, names, listings):
    """Studio, T2 and T3 in each arrondissement (median monument score there), in each management scenario."""
    profiles = pd.DataFrame([data.loc[data['bedrooms'] == b, ['accommodates', 'bathrooms']].median()
                             for b in TYPOLOGIES.values()], index=pd.Index(TYPOLOGIES, name='typology'))
    base = profiles.assign(bedrooms=list(TYPOLOGIES.values())).reset_index().merge(
        pd.DataFrame({'arrondissement': range(1, 21)}), how='cross')
    base['property_type'] = 'Entire rental unit'
    base['tourist_proximity_score'] = base['arrondissement'].map(
        data.groupby('arrondissement')['tourist_proximity_score'].median())
    base['prix_nuit'], base['prix_moyen'] = predict(bundle, base, features)
    # Payback: same DVF purchase prices and cost assumptions as modelisation/regression.py.
    base = base.merge(purchase_prices(sales), on=['arrondissement', 'typology'])
    base['achat'] = base['price'] * (1 + NOTARY_FEES) + base['surface'] * FURNISHING_EUR_M2
    base['label'] = [f'{n:02} · {names[n]}' for n in base['arrondissement']]
    scenarios = []
    for scenario in SCENARIOS:
        flats = base.assign(scenario=scenario, minimum_nights=MINIMUM_NIGHTS, **{HOST_LISTINGS: listings[scenario]})
        flats['nuits'] = predict_nights(bundle, flats)
        flats['revenu'] = flats['prix_moyen'] * flats['nuits']
        fee = CONCIERGERIE_FEE if scenario == 'Conciergerie' else 0
        flats['gain'] = (flats['revenu'] * (1 - AIRBNB_FEE - SUPPLIES - fee)
                         - flats['surface'] * CHARGES_EUR_M2 - FIXED_EUR)
        flats['annees'] = (flats['achat'] / flats['gain']).where(flats['gain'] > 0)
        scenarios.append(flats)
    return pd.concat(scenarios, ignore_index=True)


def monument_effect(bundle, features, data, flats):
    """Annual revenue of the standard T2 (managed alone) far from and close to monuments, in its arrondissement."""
    t2 = flats[(flats['typology'] == 'T2') & (flats['scenario'] == 'Gestion seul')].set_index('label')
    quarters = data.groupby('arrondissement')['tourist_proximity_score'].quantile([0.25, 0.75]).unstack()
    effect = pd.DataFrame(index=t2.index)
    for column, quarter in [('loin', 0.25), ('proche', 0.75)]:
        variant = t2.assign(tourist_proximity_score=t2['arrondissement'].map(quarters[quarter]))
        effect[column] = predict(bundle, variant, features)[1] * predict_nights(bundle, variant)
    effect['écart'] = effect['proche'] - effect['loin']
    print('\nT2 type, gestion seul : revenu annuel loin (1er quart du score) et proche (dernier quart) des monuments, '
          'dans le même arrondissement (€/an)\n' + effect.sort_values('écart', ascending=False).round(0).to_string())
    return effect


def save_tables(flats, path):
    table = flats[['label', 'typology', 'scenario', 'prix_nuit', 'nuits', 'revenu', 'achat', 'gain', 'annees']].rename(
        columns={'label': 'arrondissement', 'typology': 'logement', 'scenario': 'gestion', 'prix_nuit': 'prix par nuit (€)',
                 'nuits': 'nuits louées par an', 'revenu': 'revenu annuel (€)', 'achat': 'achat total (€)',
                 'gain': 'gain net annuel (€)', 'annees': 'années pour rembourser'}).round(0)
    table.to_csv(path, index=False)
    t2 = table[table['logement'] == 'T2'].pivot(
        index='arrondissement', columns='gestion', values=['nuits louées par an', 'revenu annuel (€)', 'années pour rembourser'])
    print('\nT2 type selon la gestion, du plus rapide au plus lent à rembourser (gestion seul)\n'
          + t2.sort_values(('années pour rembourser', 'Gestion seul')).to_string())
    print(f'Tableau complet (studio, T2, T3, deux gestions) : {path}')


def plot_monuments(effect, name, path):
    effect = effect.sort_values('écart') / 1000
    rows = np.arange(len(effect))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.hlines(rows, effect['loin'], effect['proche'], color=GRID, linewidth=2, zorder=1)
    for column, color, label in [('loin', BLUE, 'Loin des monuments (1er quart du score)'),
                                 ('proche', ORANGE, 'Proche des monuments (dernier quart)')]:
        ax.scatter(effect[column], rows, s=70, color=color, edgecolor='white', linewidth=1.5, zorder=2, label=label)
    labels = ax.get_yaxis_transform()  # x in axes fraction, y in rows.
    ax.text(1.02, len(effect) - 0.2, 'Écart', transform=labels, fontweight='bold', color=INK)
    for row, value in zip(rows, effect['écart']):
        ax.text(1.02, row, f'{value:+.1f} k€'.replace('.', ','), transform=labels, va='center', color=INK)
    ax.set_yticks(rows, effect.index)
    ax.set_xlabel('Revenu annuel prévu (k€)', color=INK)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=2, frameon=False)
    plain(ax, 'x')
    finish(fig, f'À arrondissement égal, être proche des monuments rapporte-t-il plus ? · {name}',
           'Revenu annuel prévu pour un T2 type géré seul, avec le score monuments du quart le plus éloigné et du quart le plus\n'
           'proche des annonces de son arrondissement. Le prix et les nuits louées changent ; la taille et le type, non.',
           path)
