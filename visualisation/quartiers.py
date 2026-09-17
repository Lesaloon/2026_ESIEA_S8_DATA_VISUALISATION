"""Corrélations et changements de tendance par arrondissement, sans arguments."""

import sqlite3

from correlations import BASE, DATABASE, load_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MIN_ROWS = 30


def save(fig, filename):
    fig.savefig(BASE + '/' + filename, dpi=150)
    plt.close(fig)
    print(f'Graphique : {BASE}/{filename}')


def correlations_by_area(data, price, names, title, filename):
    rows = {}
    for area, group in data.groupby('arrondissement'):
        matrix = group.drop(columns='arrondissement').corr(method='spearman', min_periods=MIN_ROWS)
        rows[f'{area:02} · {names[area]} (n={len(group)})'] = matrix[price].drop(price)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(pd.DataFrame(rows).T, annot=True, fmt='.2f', cmap='vlag',
                vmin=-1, vmax=1, center=0, ax=ax,
                cbar_kws={'label': 'Corrélation de Spearman avec le prix'})
    ax.set_title(title, pad=18)
    ax.tick_params(axis='x', rotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    fig.text(0.5, 0.015, 'Bleu : corrélation négative · Rouge : positive · Blanc sans chiffre : non calculable\n'
             'Au moins 30 paires valides ; n = total des lignes du quartier, avant exclusion des valeurs manquantes.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, filename)


def trends_by_area(data, feature, price, names, title, filename, discrete=False, scale=1):
    fig, axes = plt.subplots(5, 4, figsize=(17, 17))
    for area, ax in enumerate(axes.flat, 1):
        group = data.loc[data['arrondissement'] == area, [feature, price]].dropna()
        # Quantiles give reasonably populated bins despite the long-tailed data.
        bins = group[feature] if discrete else pd.qcut(group[feature], 10, duplicates='drop')
        points = group.groupby(bins, observed=True).agg(
            x=(feature, 'median'), y=(price, 'median'), n=(price, 'size'))
        # Leave gaps for small groups instead of drawing misleading tail segments.
        points.loc[points['n'] < MIN_ROWS, 'y'] = float('nan')
        points['y'] /= scale
        ax.plot(points['x'], points['y'], 'o-', color='#4c72b0', markersize=4)
        for i in range(1, len(points)):
            pair = points.iloc[i - 1:i + 1]
            if pair['y'].iloc[1] < pair['y'].iloc[0]:
                ax.plot(pair['x'], pair['y'], 'o-', color='#c44e52', markersize=4)
        for i, point in enumerate(points.itertuples()):
            if pd.notna(point.y):
                ax.annotate(str(point.n), (point.x, point.y), xytext=(0, 7 if i % 2 == 0 else -13),
                            textcoords='offset points', ha='center', fontsize=7)
        ax.set_title(f'{area:02} · {names[area]}', fontsize=10)
        if feature == 'Minimum nights':
            ax.set_xscale('log')
            ax.set_xticks([1, 3, 7, 30, 90], labels=['1', '3', '7', '30', '90'])
        ax.margins(y=0.15)
        ax.grid(alpha=0.2)
        ax.tick_params(labelbottom=True, labelleft=True, labelsize=8)
    fig.suptitle(title, fontsize=17)
    labels = {'Built area': 'Surface bâtie (m²)', 'Guests': 'Capacité d’accueil (personnes)',
              'Minimum nights': 'Séjour minimum (nuits, échelle logarithmique)'}
    fig.supxlabel(labels[feature], y=0.055)
    fig.supylabel('Prix de vente médian (k€)' if scale == 1000 else 'Prix par nuit médian (€)')
    grouping = 'Un point par capacité d’accueil.' if discrete else 'Un point par tranche de quantiles, propre à chaque arrondissement.'
    fig.text(0.5, 0.012, grouping + ' Chiffres = effectifs ; points de moins de 30 lignes masqués.\n'
             'Segments rouges = baisse de médiane, pas preuve d’une corrélation négative après un seuil.\n'
             'Échelles propres à chaque panneau. DVF : prix parfois répétés pour plusieurs biens. Airbnb : devis EUR positifs.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0.025, 0.08, 1, 0.97))
    save(fig, filename)


def main():
    sns.set_theme(style='white', font_scale=0.9)
    housing, airbnb = load_data()
    with sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True) as db:
        names = dict(db.execute('SELECT arrondissement, name FROM arrondissements'))
    correlations_by_area(housing, 'Sale price', names, 'Logements · Corrélations avec le prix par arrondissement · 2025',
                         'housing_correlations_quartiers.png')
    correlations_by_area(airbnb, 'Nightly price', names, 'Airbnb · Corrélations avec le prix par arrondissement · Juin 2026',
                         'airbnb_correlations_quartiers.png')
    trends_by_area(housing, 'Built area', 'Sale price', names, 'Logements · Surface bâtie (m²) et prix par arrondissement',
                   'housing_surface_quartiers.png', scale=1000)
    trends_by_area(airbnb, 'Guests', 'Nightly price', names, 'Airbnb · Capacité d’accueil et prix par arrondissement',
                   'airbnb_capacite_quartiers.png', discrete=True)
    trends_by_area(airbnb, 'Minimum nights', 'Nightly price', names, 'Airbnb · Séjour minimum (nuits) et prix par arrondissement',
                   'airbnb_sejour_quartiers.png')


if __name__ == '__main__':
    main()
