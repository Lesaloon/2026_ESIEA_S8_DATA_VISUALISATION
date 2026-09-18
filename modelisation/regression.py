"""Airbnb price and revenue regressions, and payback time by arrondissement. No arguments."""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# Paths come from the project-root .env; relative paths are from the project root.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
DATABASE = (ROOT / os.getenv('DATABASE', 'ingestion/paris.sqlite')).as_posix()
OUTPUT = Path(__file__).resolve().parent  # Charts are saved next to this script.

SIZE = ['accommodates', 'bedrooms', 'bathrooms']
TARGETS = {'Prix par nuit': 'price_quote_price_per_night', 'Revenu annuel': 'estimated_revenue_l365d'}
REFERENCE = {'arrondissement': 11, 'property_type': 'Entire rental unit'}  # One-hot baselines.
TYPOLOGIES = {'Studio': 0, 'T2': 1, 'T3': 2}  # Airbnb bedrooms; DVF main rooms = bedrooms + 1.

# Assumptions for a self-managed flat, not data: adjust them here.
NOTARY_FEES = 0.08        # share of the purchase price
FURNISHING_EUR_M2 = 250   # one-off furnishing
AIRBNB_FEE = 0.03         # host service fee, share of revenue
SUPPLIES = 0.05           # linen, consumables, small repairs, share of revenue
CHARGES_EUR_M2 = 60       # co-ownership charges + property tax, per m² and year
FIXED_EUR = 1_400         # electricity, internet, insurance, per year

# Ordinal blues (small -> large flat), then text and chrome inks.
COLORS = {'Studio': '#86b6ef', 'T2': '#2a78d6', 'T3': '#104281'}
BLUES = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#2a78d6', '#0d366b'])
INK, MUTED, GRID = '#52514e', '#898781', '#e1e0d9'


def euros(value):
    return f'{value:,.0f} €'.replace(',', ' ')  # French thousands separator.


def load_data():
    with sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True) as db:
        airbnb = pd.read_sql_query('SELECT * FROM airbnb_listings', db)
        sales = pd.read_sql_query('SELECT * FROM paris_property_sales', db)
        names = dict(db.execute('SELECT arrondissement, name FROM arrondissements'))
    return airbnb, sales, names


def prepare_listings(airbnb):
    """Active short-term entire homes, reduced to what is known before buying."""
    market = airbnb[(airbnb['room_type'] == 'Entire home/apt')
                    & (airbnb['number_of_reviews_ltm'] > 0) & (airbnb['minimum_nights'] < 30)]
    # Occupancy is measured before the price filter: fully booked flats have no quote.
    occupancy = market.groupby('arrondissement')['estimated_occupancy_l365d'].median()
    # Quotes are for late June 2026; longer quotes have much lower nightly prices.
    quote_nights = (pd.to_datetime(market['price_quote_checkout_date'])
                    - pd.to_datetime(market['price_quote_checkin_date'])).dt.days
    listings = market[(market['quote_currency'] == 'EUR') & (market['price_quote_price_per_night'] > 0)
                      & (quote_nights <= 7)].copy()

    # Studios have no bedroom count in this source (never 0); missing bathrooms are in the text.
    listings['bedrooms'] = listings['bedrooms'].fillna(0)
    listings['bathrooms'] = listings['bathrooms'].fillna(
        pd.to_numeric(listings['bathrooms_text'].str.extract(r'([\d.]+)')[0], errors='coerce'))
    # Rare property types are grouped so that each one-hot column has enough listings.
    common = listings['property_type'].value_counts().loc[lambda n: n >= 100].index
    listings['property_type'] = listings['property_type'].where(listings['property_type'].isin(common), 'Autre')

    # Everything else is dropped: ids, texts, dates, latitude/longitude, reviews, host, other prices.
    data = listings[SIZE + list(REFERENCE) + list(TARGETS.values())].dropna()
    print(f'{len(market):,} logements actifs en courte durée, {len(data):,} avec un prix exploitable')
    return data, occupancy


def encode(frame, columns=None):
    """One-hot encode arrondissement and property type, without the baseline columns."""
    encoded = pd.get_dummies(frame[SIZE + list(REFERENCE)], columns=list(REFERENCE), dtype=int)
    if columns is None:
        return encoded.drop(columns=[f'{name}_{value}' for name, value in REFERENCE.items()])
    return encoded.reindex(columns=columns, fill_value=0)


def fit_models(data):
    """One linear regression per target, on the log of the amount (prices are very skewed)."""
    X = encode(data)
    models, checks = {}, {}
    for label, target in TARGETS.items():
        rows = data[target].between(*data[target].quantile([0.01, 0.99]))  # Drop extreme values.
        X_train, X_test, y_train, y_test = train_test_split(
            X[rows], np.log(data.loc[rows, target]), test_size=0.2, random_state=42)
        models[label] = LinearRegression().fit(X_train, y_train)
        predicted = models[label].predict(X_test)
        checks[label] = {'observed': np.exp(y_test), 'predicted': np.exp(predicted),
                         'r2': r2_score(y_test, predicted),
                         'error': mean_absolute_error(np.exp(y_test), np.exp(predicted))}
        print(f"{label} : R² = {checks[label]['r2']:.2f}, erreur moyenne = {checks[label]['error']:,.0f} €")
    return models, X.columns, checks


def predict_flats(models, columns, data):
    """Predicted price and revenue of a standard studio, T2 and T3 in each arrondissement."""
    # Standard flat: median guests and bathrooms among listings with the same bedroom count.
    profiles = pd.DataFrame([data.loc[data['bedrooms'] == b, SIZE].median() for b in TYPOLOGIES.values()],
                            index=pd.Index(TYPOLOGIES, name='typology')).reset_index()
    flats = profiles.merge(pd.DataFrame({'arrondissement': range(1, 21)}), how='cross')
    flats['property_type'] = REFERENCE['property_type']
    for label, model in models.items():
        flats[label] = np.exp(model.predict(encode(flats, columns)))

    # Coefficients are multiplicative: show them in euros for the baseline T2.
    base = flats[(flats['typology'] == 'T2') & (flats['arrondissement'] == REFERENCE['arrondissement'])].iloc[0]
    effects = pd.DataFrame({label: (np.exp(model.coef_) - 1) * base[label] for label, model in models.items()},
                           index=columns)
    print(f"\nEffet de chaque variable sur un T2 type du {REFERENCE['arrondissement']}e (€)")
    print(effects[~effects.index.str.startswith('arrondissement_')].round(0).to_string())
    return flats


def purchase_prices(sales):
    """Median DVF price and surface of single-flat sales, per arrondissement and typology."""
    # DVF has no sale id: rows sharing date, nature, price and commune form one sale.
    sales = sales.assign(flat=sales['code_type_local'] == 2, other=sales['code_type_local'].isin([1, 4]))
    per_sale = sales.groupby(['date_mutation', 'nature_mutation', 'valeur_fonciere', 'code_commune']).agg(
        flats=('flat', 'sum'), other=('other', 'sum'), rooms=('nombre_pieces_principales', 'max'),
        surface=('surface_reelle_bati', 'max'), arrondissement=('arrondissement', 'first')).reset_index()
    single = per_sale[(per_sale['flats'] == 1) & (per_sale['other'] == 0) & (per_sale['nature_mutation'] == 'Vente')
                      & per_sale['rooms'].between(1, 3) & (per_sale['surface'] >= 9)]
    price_m2 = single['valeur_fonciere'] / single['surface']
    single = single[price_m2.between(*price_m2.quantile([0.01, 0.99]))]
    single = single.assign(typology=single['rooms'].map({b + 1: name for name, b in TYPOLOGIES.items()}))
    print(f"\n{len(single):,} ventes DVF d'un seul appartement retenues")
    return single.groupby(['arrondissement', 'typology']).agg(
        price=('valeur_fonciere', 'median'), surface=('surface', 'median')).reset_index()


def payback(flats, purchase, occupancy):
    """Yearly net gain and years to pay back the purchase, with two revenue estimates."""
    flats = flats.merge(purchase, on=['arrondissement', 'typology'])
    flats['investment'] = flats['price'] * (1 + NOTARY_FEES) + flats['surface'] * FURNISHING_EUR_M2
    # a: predicted price x median occupancy; b: revenue predicted by the second model.
    flats['revenue_a'] = flats['Prix par nuit'] * flats['arrondissement'].map(occupancy)
    flats['revenue_b'] = flats['Revenu annuel']
    for key in 'ab':
        revenue = flats[f'revenue_{key}']
        flats[f'gain_{key}'] = revenue * (1 - AIRBNB_FEE - SUPPLIES) - flats['surface'] * CHARGES_EUR_M2 - FIXED_EUR
        flats[f'years_{key}'] = (flats['investment'] / flats[f'gain_{key}']).where(flats[f'gain_{key}'] > 0)
    agreement = flats['years_a'].corr(flats['years_b'], method='spearman')
    print(f'Accord entre les classements a et b (corrélation de rang, 1 = identiques) : {agreement:.2f}')
    return flats


def dot_plot(table, value_format, xlabel, title, subtitle, filename):
    """One row per arrondissement, one dot per typology; the T2 value is written on the right."""
    fig, ax = plt.subplots(figsize=(10, 9))
    rows = np.arange(len(table))
    ax.hlines(rows, table.min(axis=1), table.max(axis=1), color=GRID, linewidth=2, zorder=1)
    for typology, color in COLORS.items():
        ax.scatter(table[typology], rows, s=70, color=color, edgecolor='white', linewidth=1.5,
                   zorder=2, label=typology)
    labels = ax.get_yaxis_transform()  # x in axes fraction, y in rows.
    ax.text(1.02, -1, 'T2', transform=labels, fontweight='bold', color=INK)
    for row, value in zip(rows, table['T2']):
        ax.text(1.02, row, value_format.format(value), transform=labels, va='center', color=INK)

    ax.set_yticks(rows, table.index)
    ax.set_ylim(len(table) - 0.5, -1.5)  # First row on top, room for the column header.
    ax.set_xlim(0, table.max().max() * 1.05)
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=3, frameon=False)
    fig.suptitle(title, x=0.02, ha='left', fontsize=15)
    fig.text(0.02, 0.935, subtitle, ha='left', va='top', color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUTPUT / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Graphique : {OUTPUT / filename}')


def plot_models(checks, filename):
    """Predicted against real values on the test listings, for both regressions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (label, check) in zip(axes, checks.items()):
        cells = ax.hexbin(check['observed'], check['predicted'], gridsize=35, xscale='log', yscale='log',
                          cmap=BLUES, mincnt=1, linewidths=0.2)
        ends = [check['observed'].min(), check['observed'].max()]
        ax.plot(ends, ends, color=MUTED, linewidth=1)
        for axis in [ax.xaxis, ax.yaxis]:  # Plain amounts (100 €, 200 €, 1 k€...) instead of powers of ten.
            axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            axis.set_major_formatter(FuncFormatter(
                lambda value, _: f'{value / 1000:g} k€' if value >= 1000 else f'{value:.0f} €'))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{label} · R² {check['r2']:.2f} · erreur moyenne {euros(check['error'])}",
                     loc='left', fontsize=11)
        ax.set_xlabel('Valeur réelle (€)', color=INK)
        ax.set_ylabel('Valeur prévue (€)', color=INK)
        ax.tick_params(colors=INK)
        fig.colorbar(cells, ax=ax, label='annonces')
    fig.suptitle('Qualité des deux régressions (annonces de test, jamais vues à l’entraînement)',
                 x=0.02, ha='left', fontsize=14)
    fig.text(0.02, 0.9, 'Sur la diagonale grise, la prévision est exacte.', color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUTPUT / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Graphique : {OUTPUT / filename}')


def main():
    airbnb, sales, names = load_data()
    data, occupancy = prepare_listings(airbnb)
    models, columns, checks = fit_models(data)
    flats = payback(predict_flats(models, columns, data), purchase_prices(sales), occupancy)
    flats['label'] = [f'{n:02} · {names[n]}' for n in flats['arrondissement']]

    prices = flats.pivot(index='label', columns='typology', values='Prix par nuit').sort_values('T2', ascending=False)
    years = flats.pivot(index='label', columns='typology', values='years_a').sort_values('T2')
    print('\nPrix par nuit prévu (€)\n' + prices.round(0).to_string())
    t2 = flats[flats['typology'] == 'T2'].set_index('label').sort_values('years_a')
    print('\nT2 type : achat, revenus et années pour rembourser\n' + t2[
        ['investment', 'revenue_a', 'revenue_b', 'gain_a', 'years_a', 'years_b']].rename(columns={
            'investment': 'Achat total (€)', 'revenue_a': 'Revenu a (€/an)', 'revenue_b': 'Revenu b (€/an)',
            'gain_a': 'Gain net a (€/an)', 'years_a': 'Années a', 'years_b': 'Années b'}).round(0).to_string())

    guests = flats.groupby('typology')['accommodates'].first()
    dot_plot(prices, '{:.0f} €', 'Prix par nuit (€)',
             'Quel prix pratiquer ? Prix par nuit prévu selon l’arrondissement',
             f"Prix médian prévu par la régression (erreur moyenne {checks['Prix par nuit']['error']:.0f} €) "
             f"pour un appartement type : studio {guests['Studio']:.0f} pers., "
             f"T2 {guests['T2']:.0f} pers. et 1 chambre, T3 {guests['T3']:.0f} pers. et 2 chambres.\n"
             'Devis de fin juin 2026 (haute saison).',
             'prix_par_nuit.png')
    dot_plot(years, '{:.0f} ans', 'Années de revenus nets pour rembourser l’achat',
             'Où l’achat se rembourse-t-il le plus vite ?',
             'Achat (prix DVF médian + notaire + ameublement) ÷ gain net annuel (prix prévu × occupation médiane, '
             'moins les frais d’un propriétaire qui gère lui-même).\n'
             'Moins d’années = meilleur retour. Marché observé, réglementation non appliquée : c’est un minimum.',
             'annees_remboursement.png')
    plot_models(checks, 'modeles.png')


if __name__ == '__main__':
    main()
