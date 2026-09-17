"""Two simple price-correlation charts from the Paris SQLite database."""

import sqlite3

import matplotlib
matplotlib.use('Agg')  # Save images without requiring a desktop window.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE = __file__.rsplit('/', 1)[0]
DATABASE = BASE + '/../ingestion/paris.sqlite'


def load_data():
    with sqlite3.connect(f'file:{DATABASE}?mode=ro', uri=True) as db:
        housing = pd.read_sql_query('''
            SELECT valeur_fonciere AS "Sale price",
                   surface_reelle_bati AS "Built area",
                   NULLIF(nombre_pieces_principales, 0) AS "Rooms",
                   nombre_de_lots AS "Lots",
                   (code_type_local = 1) AS "Is house"
            FROM paris_housing_sales
            WHERE nature_mutation = 'Vente'
              AND valeur_fonciere > 0 AND surface_reelle_bati > 0
        ''', db)
        airbnb = pd.read_sql_query('''
            SELECT price_quote_price_per_night AS "Nightly price",
                   accommodates AS "Guests",
                   bedrooms AS "Bedrooms",
                   beds AS "Beds",
                   bathrooms AS "Bathrooms",
                   (room_type = 'Entire home/apt') AS "Entire home",
                   minimum_nights AS "Minimum nights",
                   availability_365 AS "Availability (days)",
                   number_of_reviews AS "Reviews",
                   review_scores_rating AS "Rating",
                   host_is_superhost AS "Superhost"
            FROM airbnb_listings
            WHERE quote_currency = 'EUR' AND price_quote_price_per_night > 0
        ''', db)
    return housing, airbnb


def plot_correlations(data, price, title, filename, note):
    # Spearman compares ranks: less sensitive to the very large price outliers.
    # Missing values are excluded per pair, never filled with zeros.
    correlations = data.corr(method='spearman', min_periods=30)
    ranking = correlations[price].drop(price).dropna()
    ranking = ranking.loc[ranking.abs().sort_values().index]

    fig, (matrix, bars) = plt.subplots(
        1, 2, figsize=(17, 9), gridspec_kw={'width_ratios': [1.6, 1]})
    sns.heatmap(correlations, ax=matrix, annot=True, fmt='.2f',
                cmap='vlag', vmin=-1, vmax=1, center=0, square=True,
                cbar_kws={'shrink': 0.65, 'label': 'Spearman correlation'})
    matrix.set_title('Correlation matrix')
    matrix.tick_params(axis='x', rotation=55)
    for label in matrix.get_xticklabels():
        label.set_horizontalalignment('right')
    matrix.tick_params(axis='y', rotation=0)

    colors = ['#c44e52' if value < 0 else '#4c72b0' for value in ranking]
    bars.barh(ranking.index, ranking, color=colors)
    bars.axvline(0, color='grey', linewidth=0.8)
    bars.set_xlim(-1, 1)
    bars.set_title('Features ranked by absolute price correlation')
    bars.set_xlabel('Spearman correlation with price')
    bars.grid(axis='x', alpha=0.2)
    for i, value in enumerate(ranking):
        bars.text(value + (0.025 if value >= 0 else -0.025), i, f'{value:.2f}',
                  va='center', ha='left' if value >= 0 else 'right')

    fig.suptitle(f'{title} — {len(data):,} rows', fontsize=17)
    fig.text(0.5, 0.02, note + '\nCorrelation is association, not causal feature importance.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(BASE + '/' + filename, dpi=150)
    plt.close(fig)

    summary = pd.DataFrame({
        'correlation': ranking.iloc[::-1],
        'paired_rows': [len(data[[price, feature]].dropna()) for feature in ranking.index[::-1]],
    })
    print(f'\n{title} ({len(data):,} rows)\n{summary.round(3).to_string()}')
    print(f'Saved {BASE}/{filename}')


def main():
    sns.set_theme(style='white', font_scale=0.9)
    housing, airbnb = load_data()
    plot_correlations(housing, 'Sale price', 'Paris housing sales · 2025',
                      'housing_correlations.png',
                      'Ordinary apartment/house sales. DVF prices may cover multiple properties and repeat across rows.')
    plot_correlations(airbnb, 'Nightly price', 'Paris Airbnb · June 2026',
                      'airbnb_correlations.png',
                      'Positive EUR nightly quotes only. Missing values excluded per pair; minimum 30 paired rows.')


if __name__ == '__main__':
    main()
