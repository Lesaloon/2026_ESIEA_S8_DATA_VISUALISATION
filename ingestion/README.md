# Paris SQLite pipeline

```sh
.venv/bin/python ingestion/build_database.py
```

Requires pandas, NumPy, python-dotenv and SQLite with JSON support (see
`requirements.txt`). Input/output paths use `DVF_FILE`, `AIRBNB_FILE`, and
`DATABASE` from the project-root `.env`, with defaults under `ingestion/`.
Relative paths are resolved from the project root.

**Pipeline:** `ingest()` → `clean()` → `parse_types()` →
`score_monument_proximity()` → `write_sqlite()`.
The national DVF file (`ValeursFoncieres-2025.txt.gz`) is read directly from gzip
into pandas in chunks, retaining department 75. No manual extraction is needed.
Whitespace and empty values are cleaned, types parsed explicitly, and pandas
`to_sql()` writes **`ingestion/paris.sqlite`**. Re-running replaces the tables.

## Contents

| Table/view | Contents |
| --- | --- |
| `paris_property_sales` | All 84,440 Paris DVF rows from the 2025 file, with all original fields in lowercase snake_case. Includes commercial property, dependencies, and exchanges. |
| `paris_housing_sales` (view) | 37,821 house/apartment rows sold through ordinary sales, future-completion sales, building-land sales or adjudications; excludes exchanges. |
| `airbnb_listings` | All 77,679 listings in the June 2026 Paris snapshot, preserving all original fields. |
| `registered_airbnb_listings` (view) | 53,758 listings with a declared registration-shaped licence: five digits followed by eight alphanumeric characters. This is **not official verification**. |
| `arrondissements` | Arrondissement numbers 1–20 and names, for joining/labeling charts. |
| `monuments` | Reference sites and source attributes from `monuments_paris_importance.json`. |

Both main tables contain `arrondissement` and `source_row` (one-based input data
record, excluding header). Airbnb IDs have a unique index. Indexes also support
arrondissement, sale date, and licence lookups.

## Types and interpretation

- INTEGER: counts, listing/host IDs, arrondissement, boolean fields (0/1/NULL).
  IDs are parsed directly from strings to avoid floating-point precision loss.
- REAL: prices, areas in m², coordinates, ratings, fractional bathroom counts.
  Response rates use percentages 0–100. DVF prices are euros.
- TEXT: ISO dates (`YYYY-MM-DD`), descriptions, identifiers/codes such as postal
  codes and licences (preserving leading zeros), source JSON/list strings.
- Missing values are SQL NULL; invalid numbers/dates stop parsing. Empty columns
  still receive their intended types. Rows are not deduplicated.
- `registration_status`: `missing`, `exempt`, `mobility_lease`, `declared_number`,
  or `other`. The original `license` remains available, including unknown formats.
- Airbnb `price` has `$` formatting even when quote JSON says EUR. The numeric
  value is preserved and `quote_currency` is extracted from `price_quote_raw`.
  Use the quote currency with quote prices; do not assume `$` means USD.

**DVF rows are not unique transactions.** A single sale can span several properties,
lots and dependencies, repeating the full transaction price. No transaction ID is
invented and no price-per-m² is calculated. Do not sum row prices as turnover or
count rows as distinct sales without addressing this source granularity.

Coverage is limited to the supplied files: DVF dates run from 2025-01-02 through
2025-12-31, whereas Airbnb is a June 2026 snapshot, not a complete official registry.

## Tourist proximity score

Each Airbnb listing receives three additional columns, also exposed through
`registered_airbnb_listings`:

- `nearest_monument_name`: closest site in the reference catalogue.
- `nearest_monument_distance_km`: Haversine distance, using Earth's mean radius
  of 6,371.0088 km; this is a straight-line distance, not a walking route.
- `tourist_proximity_score`: score from 0 to 100, computed from the three nearest
  sites at distances `d1 <= d2 <= d3` in kilometres:

  `100 * (0.5 * 2**(-d1) + 0.3 * 2**(-d2) + 0.2 * 2**(-d3))`

Each additional kilometre halves a site's contribution. Distances of 0.2, 0.5,
and 0.8 km yield about 76.23. Scores are stored without rounding. Listings with
missing or out-of-range coordinates retain NULL in all three derived columns.

The fixed input catalogue determines which sites count, including nearby sites
listed separately. Its visitor counts and importance weights are preserved in
`monuments` but do not enter this distance-only score. Changing the catalogue
requires regenerating the scores; compare scores using the same catalogue.

## Next step: use in pandas

From the project root (use `paris.sqlite` when running inside `ingestion/`):

```python
import sqlite3
import pandas as pd

with sqlite3.connect('file:ingestion/paris.sqlite?mode=ro', uri=True) as db:
    sales = pd.read_sql_query('SELECT * FROM paris_housing_sales', db,
                             parse_dates=['date_mutation'])
    airbnb = pd.read_sql_query('SELECT * FROM airbnb_listings', db)
```

Aggregate before joining to avoid multiplying individual sales and listings:

```sql
WITH sales AS (
    SELECT arrondissement, count(*) AS housing_sale_rows
    FROM paris_housing_sales GROUP BY arrondissement
), listings AS (
    SELECT arrondissement, count(*) AS airbnb_count
    FROM airbnb_listings GROUP BY arrondissement
)
SELECT a.*, s.housing_sale_rows, l.airbnb_count
FROM arrondissements a
LEFT JOIN sales s USING (arrondissement)
LEFT JOIN listings l USING (arrondissement)
ORDER BY arrondissement;
```
