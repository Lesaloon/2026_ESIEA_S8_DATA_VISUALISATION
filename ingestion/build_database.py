"""Paris data pipeline: ingest -> clean -> parse types -> SQLite."""

import re
import sqlite3

import pandas as pd


BASE = __file__.rsplit('/', 1)[0]
DATABASE = BASE + '/paris.sqlite'
ARRONDISSEMENTS = [
    'Louvre', 'Bourse', 'Temple', 'Hôtel-de-Ville', 'Panthéon',
    'Luxembourg', 'Palais-Bourbon', 'Élysée', 'Opéra', 'Entrepôt',
    'Popincourt', 'Reuilly', 'Gobelins', 'Observatoire', 'Vaugirard',
    'Passy', 'Batignolles-Monceau', 'Buttes-Montmartre', 'Buttes-Chaumont',
    'Ménilmontant',
]
AIR_INTEGER = '''id scrape_id host_id host_profile_id
hosts_time_as_user_years hosts_time_as_user_months hosts_time_as_host_years
hosts_time_as_host_months host_listings_count host_total_listings_count accommodates
bedrooms beds minimum_nights maximum_nights minimum_minimum_nights
maximum_minimum_nights minimum_maximum_nights maximum_maximum_nights
availability_30 availability_60 availability_90 availability_365 number_of_reviews
number_of_reviews_ltm number_of_reviews_l30d availability_eoy number_of_reviews_ly
estimated_occupancy_l365d calculated_host_listings_count
calculated_host_listings_count_entire_homes calculated_host_listings_count_private_rooms
calculated_host_listings_count_shared_rooms'''.split()
AIR_REAL = '''latitude longitude bathrooms price price_quote_total_price
price_quote_price_per_night minimum_nights_avg_ntm maximum_nights_avg_ntm
estimated_revenue_l365d review_scores_rating review_scores_accuracy
review_scores_cleanliness review_scores_checkin review_scores_communication
review_scores_location review_scores_value reviews_per_month
host_response_rate host_acceptance_rate'''.split()
AIR_BOOL = '''host_is_superhost host_has_profile_pic host_identity_verified
has_availability instant_bookable'''.split()
AIR_DATE = '''last_scraped host_since price_quote_checkin_date
price_quote_checkout_date calendar_last_scraped first_review last_review'''.split()


def ingest():
    # Read as strings first: IDs must never pass through floating-point numbers.
    # Filter the national file in chunks to keep memory use reasonable.
    paris = []
    for chunk in pd.read_csv(BASE + '/ValeursFoncieres-2025.txt.gz', sep='|',
                             dtype='string', keep_default_na=False, chunksize=100_000):
        chunk['source_row'] = chunk.index + 1
        paris.append(chunk.loc[chunk['Code departement'].str.strip() == '75'])
    sales = pd.concat(paris, ignore_index=True)
    airbnb = pd.read_csv(BASE + '/listing.csv.gz', dtype='string', keep_default_na=False)
    airbnb['source_row'] = airbnb.index + 1
    return sales, airbnb


def clean(sales, airbnb):
    sales.columns = [re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_') for c in sales.columns]
    for frame in (sales, airbnb):
        for column in frame.select_dtypes(include='string'):
            frame[column] = frame[column].str.strip().replace('', pd.NA)
    # This source uses the literal string "None" for missing host verifications.
    airbnb['host_verifications'] = airbnb['host_verifications'].replace('None', pd.NA)
    # Keep duplicate-looking DVF rows: they may represent different lots/locals.
    return sales, airbnb


def numeric(series, dtype, french=False):
    if french:
        series = series.str.replace(',', '.', regex=False)
    else:
        series = series.str.replace(r'[$,%]', '', regex=True)
    # Invalid numbers fail instead of silently becoming missing values.
    return pd.to_numeric(series, errors='raise').astype(dtype)


def parse_types(sales, airbnb):
    sales['date_mutation'] = pd.to_datetime(
        sales['date_mutation'], format='%d/%m/%Y', errors='raise')
    for column in ['nombre_de_lots', 'code_type_local', 'nombre_pieces_principales']:
        sales[column] = numeric(sales[column], 'Int64', french=True)
    for column in ['valeur_fonciere', 'surface_reelle_bati', 'surface_terrain'] + [
            c for c in sales if c.startswith('surface_carrez_')]:
        sales[column] = numeric(sales[column], 'Float64', french=True)
    sales['arrondissement'] = sales['code_commune'].astype('Int64') - 100

    for column in AIR_INTEGER:
        # Direct string -> integer conversion preserves even nullable 19-digit IDs.
        airbnb[column] = airbnb[column].astype('Int64')
    for column in AIR_REAL:
        airbnb[column] = numeric(airbnb[column], 'Float64')
    for column in AIR_BOOL:
        if not airbnb[column].dropna().isin(['t', 'f']).all():
            raise ValueError(f'Unexpected boolean in {column}')
        airbnb[column] = airbnb[column].map({'t': 1, 'f': 0}).astype('Int64')
    for column in AIR_DATE:
        airbnb[column] = pd.to_datetime(airbnb[column], format='%Y-%m-%d', errors='raise')
    mapping = {name: n for n, name in enumerate(ARRONDISSEMENTS, 1)}
    airbnb['arrondissement'] = airbnb['neighbourhood_cleansed'].map(mapping).astype('Int64')
    airbnb['registration_status'] = pd.Series('other', index=airbnb.index, dtype='string')
    license = airbnb['license']
    airbnb.loc[license.isna(), 'registration_status'] = 'missing'
    airbnb.loc[license.str.startswith('Exempt', na=False), 'registration_status'] = 'exempt'
    airbnb.loc[license.str.contains('mobility lease', na=False), 'registration_status'] = 'mobility_lease'
    airbnb.loc[license.str.fullmatch(r'[0-9]{5}[a-zA-Z0-9]{8}', na=False),
               'registration_status'] = 'declared_number'

    for frame in (sales, airbnb):
        if frame['arrondissement'].isna().any() or not frame['arrondissement'].between(1, 20).all():
            raise ValueError('Missing or unknown Paris arrondissement')
    if airbnb['id'].isna().any() or airbnb['id'].duplicated().any():
        raise ValueError('Airbnb IDs must be present and unique')
    return sales, airbnb


def write_sqlite(sales, airbnb):
    with sqlite3.connect(DATABASE) as connection:
        # Remove views first so reruns also work after schema changes.
        connection.execute('DROP VIEW IF EXISTS paris_housing_sales')
        connection.execute('DROP VIEW IF EXISTS registered_airbnb_listings')
        for name, frame in [('paris_property_sales', sales), ('airbnb_listings', airbnb)]:
            frame = frame.copy()
            types = {}
            for column in frame:
                dtype = frame[column].dtype
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    frame[column] = frame[column].dt.strftime('%Y-%m-%d')
                    types[column] = 'TEXT'
                elif pd.api.types.is_integer_dtype(dtype):
                    types[column] = 'INTEGER'
                elif pd.api.types.is_float_dtype(dtype):
                    types[column] = 'REAL'
                else:
                    types[column] = 'TEXT'
            frame.to_sql(name, connection, if_exists='replace', index=False, dtype=types, chunksize=1000)
            print(f'{name}: {len(frame):,} rows')
        # Use SQLite JSON extraction: no extra Python library needed.
        connection.execute('ALTER TABLE airbnb_listings ADD COLUMN quote_currency TEXT')
        connection.execute("UPDATE airbnb_listings SET quote_currency = json_extract(price_quote_raw, '$.quote.currency')")
        pd.DataFrame({'arrondissement': range(1, 21), 'name': ARRONDISSEMENTS}).to_sql(
            'arrondissements', connection, if_exists='replace', index=False)
        connection.executescript('''
            DROP TABLE IF EXISTS ingestion_metadata;
            CREATE UNIQUE INDEX sales_source_row ON paris_property_sales(source_row);
            CREATE INDEX sales_arrondissement_date ON paris_property_sales(arrondissement, date_mutation);
            CREATE UNIQUE INDEX airbnb_id ON airbnb_listings(id);
            CREATE INDEX airbnb_arrondissement ON airbnb_listings(arrondissement);
            CREATE INDEX airbnb_license ON airbnb_listings(license);
            CREATE VIEW paris_housing_sales AS
                SELECT * FROM paris_property_sales WHERE code_type_local IN (1, 2)
                AND nature_mutation IN ('Vente', 'Vente en l''état futur d''achèvement',
                                       'Vente terrain à bâtir', 'Adjudication');
            CREATE VIEW registered_airbnb_listings AS
                SELECT * FROM airbnb_listings WHERE registration_status = 'declared_number';
        ''')
        if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed')
    print(f'Database ready: {DATABASE}')


def main():
    sales, airbnb = ingest()
    sales, airbnb = clean(sales, airbnb)
    sales, airbnb = parse_types(sales, airbnb)
    write_sqlite(sales, airbnb)


if __name__ == '__main__':
    main()
