"""Interactive Paris rental market dashboard and nightly-price estimator."""

import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def project_path(environment_name, default):
    path = Path(os.getenv(environment_name, default))
    return path if path.is_absolute() else ROOT / path


DATABASE = project_path("DATABASE", "ingestion/paris.sqlite")
MODEL_PATH = project_path("MODEL_PATH", "web/model.pkl")
PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": True, "responsive": True}
COLORS = {
    "paper": "#f3f0e8",
    "panel": "#fffdf8",
    "ink": "#18262e",
    "muted": "#66777e",
    "blue": "#176b87",
    "blue_light": "#8cc4cd",
    "red": "#df5338",
    "line": "#d6d5cd",
}
NOTARY_FEES = 0.08
FURNISHING_EUR_M2 = 250
AIRBNB_FEE = 0.03
SUPPLIES = 0.05
CHARGES_EUR_M2 = 60
FIXED_EUR = 1_400


def load_market():
    if not DATABASE.exists():
        raise FileNotFoundError(
            f"Database not found at {DATABASE}. Run ingestion/build_database.py first."
        )
    query = """
        SELECT id, name, latitude, longitude, arrondissement, property_type,
               room_type, accommodates, bedrooms, bathrooms, minimum_nights,
               number_of_reviews_ltm, review_scores_rating,
               estimated_occupancy_l365d, price_quote_price_per_night,
               nearest_monument_name, nearest_monument_distance_km,
               tourist_proximity_score
        FROM airbnb_listings
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
          AND price_quote_price_per_night > 0
          AND quote_currency = 'EUR'
    """
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        listings = pd.read_sql_query(query, connection)
        areas = pd.read_sql_query(
            "SELECT arrondissement, name FROM arrondissements ORDER BY arrondissement",
            connection,
        )
        monuments = pd.read_sql_query(
            "SELECT nom, lat, long FROM monuments ORDER BY nom", connection
        )
    listings["bedrooms"] = listings["bedrooms"].fillna(0)
    listings["bathrooms"] = listings["bathrooms"].fillna(1)
    listings["area_label"] = listings["arrondissement"].map(
        dict(zip(areas["arrondissement"], areas["name"]))
    )
    listings["area_label"] = (
        listings["arrondissement"].astype(str).str.zfill(2)
        + " · "
        + listings["area_label"]
    )
    return listings, areas, monuments


def load_property_sales():
    """Prepare median DVF apartment sales by arrondissement and bedroom count."""
    query = """
        SELECT date_mutation, nature_mutation, valeur_fonciere, code_commune,
               code_type_local, nombre_pieces_principales, surface_reelle_bati,
               arrondissement
        FROM paris_property_sales
        WHERE valeur_fonciere > 0 AND arrondissement BETWEEN 1 AND 20
    """
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        sales = pd.read_sql_query(query, connection)
    sales = sales.assign(
        flat=sales["code_type_local"] == 2,
        other=sales["code_type_local"].isin([1, 4]),
    )
    per_sale = sales.groupby(
        ["date_mutation", "nature_mutation", "valeur_fonciere", "code_commune"],
        as_index=False,
    ).agg(
        flats=("flat", "sum"),
        other=("other", "sum"),
        rooms=("nombre_pieces_principales", "max"),
        surface=("surface_reelle_bati", "max"),
        arrondissement=("arrondissement", "first"),
    )
    single = per_sale[
        (per_sale["flats"] == 1)
        & (per_sale["other"] == 0)
        & (per_sale["nature_mutation"] == "Vente")
        & per_sale["rooms"].between(1, 6)
        & (per_sale["surface"] >= 9)
    ].copy()
    price_m2 = single["valeur_fonciere"] / single["surface"]
    single = single[price_m2.between(*price_m2.quantile([0.01, 0.99]))]
    single["bedrooms"] = single["rooms"] - 1
    return single.groupby(["arrondissement", "bedrooms"], as_index=False).agg(
        sale_price=("valeur_fonciere", "median"),
        surface=("surface", "median"),
        sales=("valeur_fonciere", "size"),
    )


def load_area_occupancy():
    query = """
        SELECT arrondissement, estimated_occupancy_l365d
        FROM airbnb_listings
        WHERE room_type = 'Entire home/apt'
          AND number_of_reviews_ltm > 0
          AND minimum_nights < 30
          AND estimated_occupancy_l365d IS NOT NULL
    """
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        occupancy = pd.read_sql_query(query, connection)
    return occupancy.groupby("arrondissement")["estimated_occupancy_l365d"].median()


LISTINGS, AREAS, MONUMENTS = load_market()
PROPERTY_SALES = load_property_sales()
AREA_OCCUPANCY = load_area_occupancy()
AREA_NAMES = dict(zip(AREAS["arrondissement"], AREAS["name"]))
AREA_OPTIONS = [
    {"label": f"{row.arrondissement:02d} · {row.name}", "value": row.arrondissement}
    for row in AREAS.itertuples()
]
ROOM_OPTIONS = [
    {"label": value, "value": value}
    for value in sorted(LISTINGS["room_type"].dropna().unique())
]
PROPERTY_OPTIONS = [
    {"label": value, "value": value}
    for value in LISTINGS["property_type"].value_counts().head(12).index
]
AREA_CENTERS = (
    LISTINGS.groupby(["arrondissement", "area_label"], as_index=False)
    .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
)
INITIAL_CENTER = AREA_CENTERS.loc[AREA_CENTERS["arrondissement"] == 11].iloc[0]
INITIAL_LOCATION = {
    "latitude": float(INITIAL_CENTER["latitude"]),
    "longitude": float(INITIAL_CENTER["longitude"]),
    "arrondissement": 11,
}


def placement_grid():
    """Create an invisible click surface and assign each point to its nearest area center."""
    latitudes, longitudes = np.meshgrid(
        np.linspace(48.815, 48.905, 61),
        np.linspace(2.225, 2.470, 91),
        indexing="ij",
    )
    points = pd.DataFrame(
        {"latitude": latitudes.ravel(), "longitude": longitudes.ravel()}
    )
    point_lat = np.radians(points["latitude"].to_numpy())[:, None]
    point_lon = np.radians(points["longitude"].to_numpy())[:, None]
    center_lat = np.radians(AREA_CENTERS["latitude"].to_numpy())[None, :]
    center_lon = np.radians(AREA_CENTERS["longitude"].to_numpy())[None, :]
    distance = (
        (point_lat - center_lat) ** 2
        + (np.cos(point_lat) * (point_lon - center_lon)) ** 2
    )
    nearest = np.argmin(distance, axis=1)
    points["arrondissement"] = AREA_CENTERS.iloc[nearest]["arrondissement"].to_numpy()
    return points


PLACEMENT_GRID = placement_grid()


def blank_figure(message):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": COLORS["muted"]},
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def style_figure(figure, margin=None):
    figure.update_layout(
        template="plotly_white",
        font={"family": "IBM Plex Sans, Arial, sans-serif", "color": COLORS["ink"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=margin or {"l": 44, "r": 18, "t": 35, "b": 40},
        hoverlabel={"bgcolor": COLORS["ink"], "font_color": "white"},
    )
    figure.update_xaxes(gridcolor=COLORS["line"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["line"], zeroline=False)
    return figure


def market_map(frame):
    sample = frame.sample(min(3500, len(frame)), random_state=42)
    figure = px.scatter_map(
        sample,
        lat="latitude",
        lon="longitude",
        color="price_quote_price_per_night",
        color_continuous_scale=[COLORS["blue_light"], COLORS["blue"], COLORS["red"]],
        range_color=(
            float(frame["price_quote_price_per_night"].quantile(0.05)),
            float(frame["price_quote_price_per_night"].quantile(0.95)),
        ),
        hover_name="name",
        hover_data={
            "price_quote_price_per_night": ":.0f",
            "area_label": True,
            "accommodates": True,
            "latitude": False,
            "longitude": False,
        },
        labels={
            "price_quote_price_per_night": "Prix / nuit (€)",
            "area_label": "Arrondissement",
            "accommodates": "Voyageurs",
        },
        zoom=10.8,
        center={"lat": 48.8566, "lon": 2.3522},
        opacity=0.63,
    )
    figure.update_traces(marker={"size": 7})
    figure.update_layout(
        map_style="carto-positron",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar={
            "title": "€/nuit",
            "orientation": "h",
            "y": 0.02,
            "x": 0.5,
            "len": 0.48,
            "thickness": 9,
            "bgcolor": "rgba(255,253,248,.88)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision="market-map",
    )
    return figure


def price_distribution(frame):
    clipped = frame[frame["price_quote_price_per_night"] <= frame["price_quote_price_per_night"].quantile(0.98)]
    figure = px.histogram(
        clipped,
        x="price_quote_price_per_night",
        color="room_type",
        nbins=45,
        barmode="overlay",
        opacity=0.68,
        color_discrete_sequence=[COLORS["blue"], COLORS["red"], "#77a48f", "#e6ad4f"],
        labels={"price_quote_price_per_night": "Prix par nuit (€)", "count": "Annonces", "room_type": "Type"},
    )
    figure.update_layout(showlegend=False, bargap=0.04)
    return style_figure(figure)


def area_prices(frame):
    summary = (
        frame.groupby(["arrondissement", "area_label"], as_index=False)
        .agg(price=("price_quote_price_per_night", "median"), listings=("id", "size"))
        .sort_values("price")
    )
    figure = px.bar(
        summary,
        x="price",
        y="area_label",
        orientation="h",
        color="price",
        color_continuous_scale=[COLORS["blue_light"], COLORS["blue"]],
        custom_data=["listings"],
        labels={"price": "Prix médian (€)", "area_label": ""},
    )
    figure.update_traces(
        hovertemplate="%{y}<br><b>%{x:.0f} €</b> / nuit<br>%{customdata[0]:,} annonces<extra></extra>"
    )
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, {"l": 125, "r": 18, "t": 20, "b": 40})


def price_correlations(frame):
    columns = {
        "accommodates": "Voyageurs",
        "bedrooms": "Chambres",
        "bathrooms": "Salles de bain",
        "minimum_nights": "Séjour minimum",
        "number_of_reviews_ltm": "Avis récents",
        "review_scores_rating": "Note",
        "estimated_occupancy_l365d": "Occupation annuelle",
        "tourist_proximity_score": "Proximité monuments",
    }
    values = frame[["price_quote_price_per_night", *columns]].corr(
        method="spearman", min_periods=30
    )["price_quote_price_per_night"].drop("price_quote_price_per_night").dropna()
    values = values.sort_values()
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[columns[name] for name in values.index],
            orientation="h",
            marker_color=[COLORS["red"] if value < 0 else COLORS["blue"] for value in values],
            text=[f"{value:+.2f}" for value in values],
            textposition="outside",
            hovertemplate="%{y}<br>Corrélation : %{x:.2f}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=COLORS["line"])
    figure.update_xaxes(range=[-1, 1], title="Corrélation de Spearman avec le prix")
    return style_figure(figure, {"l": 130, "r": 35, "t": 20, "b": 45})


def capacity_counts(frame):
    capacity = frame["accommodates"].clip(upper=8).value_counts().sort_index()
    figure = go.Figure(
        go.Bar(
            x=[f"{int(value)}" if value < 8 else "8+" for value in capacity.index],
            y=capacity.values,
            marker_color=COLORS["blue"],
            hovertemplate="%{x} voyageur(s)<br><b>%{y:,}</b> annonces<extra></extra>",
        )
    )
    figure.update_xaxes(title="Capacité d’accueil")
    figure.update_yaxes(title="Nombre d’annonces")
    return style_figure(figure)


def occupancy_by_area(frame):
    summary = (
        frame.dropna(subset=["estimated_occupancy_l365d"])
        .groupby("area_label", as_index=False)["estimated_occupancy_l365d"]
        .median()
        .sort_values("estimated_occupancy_l365d")
    )
    figure = px.bar(
        summary,
        x="estimated_occupancy_l365d",
        y="area_label",
        orientation="h",
        color="estimated_occupancy_l365d",
        color_continuous_scale=[COLORS["blue_light"], COLORS["blue"]],
        labels={"estimated_occupancy_l365d": "Nuits occupées / an", "area_label": ""},
    )
    figure.update_traces(hovertemplate="%{y}<br><b>%{x:.0f}</b> nuits occupées / an<extra></extra>")
    figure.update_layout(coloraxis_showscale=False)
    return style_figure(figure, {"l": 125, "r": 18, "t": 20, "b": 40})


def monument_proximity(latitude, longitude):
    """Apply the same three-nearest-monuments score as database ingestion."""
    lat = np.radians(float(latitude))
    lon = np.radians(float(longitude))
    monument_lat = np.radians(MONUMENTS["lat"].to_numpy(dtype=float))
    monument_lon = np.radians(MONUMENTS["long"].to_numpy(dtype=float))
    haversine = (
        np.sin((monument_lat - lat) / 2) ** 2
        + np.cos(lat) * np.cos(monument_lat) * np.sin((monument_lon - lon) / 2) ** 2
    )
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    nearest = np.argsort(distances, kind="stable")[:3]
    closest = distances[nearest]
    score = 100 * (np.exp2(-closest) @ np.array([0.5, 0.3, 0.2]))
    return {
        "nearest_monument_name": MONUMENTS.iloc[nearest[0]]["nom"],
        "nearest_monument_distance_km": float(closest[0]),
        "tourist_proximity_score": float(score),
        "nearest_indices": nearest,
    }


def estimator_map(area, location):
    proximity = monument_proximity(location["latitude"], location["longitude"])
    figure = go.Figure()
    figure.add_trace(
        go.Scattermap(
            lat=PLACEMENT_GRID["latitude"], lon=PLACEMENT_GRID["longitude"], mode="markers",
            marker={"size": 24, "color": "rgba(23,107,135,0.002)"},
            customdata=np.column_stack(
                [
                    np.full(len(PLACEMENT_GRID), "placement"),
                    PLACEMENT_GRID[["latitude", "longitude", "arrondissement"]].to_numpy(),
                ]
            ),
            hoverinfo="none",
        )
    )
    centers = AREA_CENTERS.copy()
    centers["label"] = centers["arrondissement"].map(lambda value: f"{value:02d}")
    figure.add_trace(
        go.Scattermap(
            lat=centers["latitude"], lon=centers["longitude"], mode="markers+text",
            marker={"size": 15, "color": COLORS["blue"], "opacity": 0.9},
            text=centers["label"], textposition="top center",
            customdata=np.column_stack(
                [
                    np.full(len(centers), "placement"),
                    centers[["latitude", "longitude", "arrondissement"]].to_numpy(),
                ]
            ),
            hovertext=centers["area_label"],
            hovertemplate="%{hovertext}<br>Cliquez pour sélectionner<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scattermap(
            lat=MONUMENTS["lat"], lon=MONUMENTS["long"], mode="markers",
            marker={"size": 7, "color": "#25845b", "opacity": 0.75}, text=MONUMENTS["nom"],
            hovertemplate="%{text}<extra></extra>", hoverinfo="text",
        )
    )
    figure.add_trace(
        go.Scattermap(
            lat=[location["latitude"]], lon=[location["longitude"]], mode="markers",
            marker={"size": 17, "color": COLORS["red"]},
            text=["Emplacement choisi"], hovertemplate="%{text}<extra></extra>",
        )
    )
    center = AREA_CENTERS.loc[AREA_CENTERS["arrondissement"] == area].iloc[0]
    figure.update_layout(
        map_style="carto-positron",
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event",
        map={"zoom": 10.5, "center": {"lat": center["latitude"], "lon": center["longitude"]}},
        uirevision=f"estimator-map-{area}",
    )
    return figure


def load_model():
    """Load a trusted local pickle. Never use this with an untrusted model file."""
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_price(features):
    bundle = load_model()
    if bundle is None:
        raise FileNotFoundError(f"Ajoutez le modèle dans {MODEL_PATH.relative_to(ROOT)}")

    metadata = bundle if isinstance(bundle, dict) else {}
    model = metadata.get("model", bundle) if isinstance(bundle, dict) else bundle
    columns = metadata.get("feature_columns")
    if columns is None:
        columns = getattr(model, "feature_names_in_", None)
    raw = pd.DataFrame([features])
    capacity = f"8+" if features["accommodates"] >= 8 else str(int(features["accommodates"]))
    rooms = f"4+" if features["bedrooms"] >= 4 else str(int(features["bedrooms"]))
    supported_properties = {
        "Entire rental unit", "Entire condo", "Entire home", "Entire loft",
        "Entire serviced apartment", "Entire townhouse",
    }
    model_property = features["property_type"] if features["property_type"] in supported_properties else "Autre"

    if columns is not None:
        prepared = {}
        for column in columns:
            if column in raw.columns:
                prepared[column] = raw.loc[0, column]
            elif column == f"arrondissement_{features['arrondissement']}":
                prepared[column] = 1
            elif column == f"property_type_{model_property}":
                prepared[column] = 1
            elif column == f"room_type_{features['room_type']}":
                prepared[column] = 1
            elif column == f"capacity_{capacity}":
                prepared[column] = 1
            elif column == f"rooms_{rooms}":
                prepared[column] = 1
            else:
                prepared[column] = 0
        model_input = pd.DataFrame([prepared], columns=list(columns))
    else:
        model_input = raw

    prediction = float(np.asarray(model.predict(model_input)).ravel()[0])
    transform = metadata.get("target_transform", "none")
    if transform == "log":
        prediction = float(np.exp(prediction))
    elif transform == "log1p":
        prediction = float(np.expm1(prediction))
    prediction *= float(metadata.get("duan_factor", 1))
    if not np.isfinite(prediction) or prediction < 0:
        raise ValueError("Le modèle a renvoyé une estimation invalide.")
    return prediction


def investment_metrics(area, bedrooms, nightly_price):
    """Convert a nightly estimate into an indicative DVF investment scenario."""
    comparable = PROPERTY_SALES[PROPERTY_SALES["arrondissement"] == int(area)].copy()
    if comparable.empty:
        raise ValueError("Aucune transaction DVF comparable pour cet arrondissement.")
    comparable["bedroom_gap"] = (comparable["bedrooms"] - int(bedrooms)).abs()
    sale = comparable.sort_values(["bedroom_gap", "sales"], ascending=[True, False]).iloc[0]
    nights = float(AREA_OCCUPANCY.get(int(area), np.nan))
    if not np.isfinite(nights):
        raise ValueError("Occupation indisponible pour cet arrondissement.")
    gross_revenue = float(nightly_price) * nights
    investment = sale["sale_price"] * (1 + NOTARY_FEES) + sale["surface"] * FURNISHING_EUR_M2
    net_revenue = (
        gross_revenue * (1 - AIRBNB_FEE - SUPPLIES)
        - sale["surface"] * CHARGES_EUR_M2
        - FIXED_EUR
    )
    payback = investment / net_revenue if net_revenue > 0 else np.nan
    return {
        "sale_price": float(sale["sale_price"]),
        "surface": float(sale["surface"]),
        "sales": int(sale["sales"]),
        "bedrooms": int(sale["bedrooms"]),
        "occupancy": nights,
        "gross_revenue": gross_revenue,
        "payback": float(payback),
    }


def investment_options(budget):
    """Rank median properties whose full acquisition cost fits the budget."""
    rows = []
    candidates = PROPERTY_SALES[PROPERTY_SALES["sales"] >= 5]
    for sale in candidates.itertuples():
        total_cost = (
            sale.sale_price * (1 + NOTARY_FEES)
            + sale.surface * FURNISHING_EUR_M2
        )
        if total_cost > budget:
            continue
        comparable = LISTINGS[
            (LISTINGS["arrondissement"] == sale.arrondissement)
            & (LISTINGS["bedrooms"] == sale.bedrooms)
        ]
        if comparable.empty:
            comparable = LISTINGS[LISTINGS["bedrooms"] == sale.bedrooms]
        if comparable.empty:
            continue
        guests = int(np.clip(round(comparable["accommodates"].median()), 1, 16))
        bathrooms = float(comparable["bathrooms"].median())
        monument_score = float(
            LISTINGS.loc[
                LISTINGS["arrondissement"] == sale.arrondissement,
                "tourist_proximity_score",
            ].median()
        )
        nightly_price = predict_price(
            {
                "arrondissement": int(sale.arrondissement),
                "latitude": 0,
                "longitude": 0,
                "accommodates": guests,
                "bedrooms": float(sale.bedrooms),
                "bathrooms": bathrooms,
                "minimum_nights": 2,
                "property_type": "Entire rental unit",
                "room_type": "Entire home/apt",
                "nearest_monument_distance_km": 0,
                "tourist_proximity_score": monument_score,
            }
        )
        nights = float(AREA_OCCUPANCY.get(int(sale.arrondissement), np.nan))
        if not np.isfinite(nights):
            continue
        gross_revenue = nightly_price * nights
        net_revenue = (
            gross_revenue * (1 - AIRBNB_FEE - SUPPLIES)
            - sale.surface * CHARGES_EUR_M2
            - FIXED_EUR
        )
        if net_revenue <= 0:
            continue
        typology = "Studio" if sale.bedrooms == 0 else f"T{int(sale.bedrooms) + 1}"
        rows.append(
            {
                "arrondissement": int(sale.arrondissement),
                "area": AREA_NAMES[int(sale.arrondissement)],
                "typology": typology,
                "label": f"{int(sale.arrondissement):02d} · {AREA_NAMES[int(sale.arrondissement)]} · {typology}",
                "total_cost": float(total_cost),
                "sale_price": float(sale.sale_price),
                "surface": float(sale.surface),
                "sales": int(sale.sales),
                "nightly_price": nightly_price,
                "occupancy": nights,
                "gross_revenue": gross_revenue,
                "net_revenue": net_revenue,
                "roi": 100 * net_revenue / total_cost,
                "payback": total_cost / net_revenue,
            }
        )
    return pd.DataFrame(rows).sort_values(["roi", "sales"], ascending=[False, False]) if rows else pd.DataFrame()


# "Optimiser pour" choices: the investment_options() column ranked highest first, and how it reads.
OBJECTIVES = {
    "roi": {
        "label": "Rendement net",
        "panel": "Meilleurs rendements accessibles",
        "axis": "Rendement net annuel (%)",
        "text": lambda value: f"{value:.1f} %",
    },
    "net_revenue": {
        "label": "Gain net par an",
        "panel": "Meilleurs gains nets accessibles",
        "axis": "Gain net annuel (€)",
        "text": lambda value: f"{value:,.0f} €".replace(",", " "),
    },
}


def investment_chart(options, objective="roi"):
    spec = OBJECTIVES[objective]
    shown = options.head(10).sort_values(objective)
    figure = go.Figure(
        go.Bar(
            x=shown[objective],
            y=shown["label"],
            orientation="h",
            marker_color=COLORS["blue"],
            customdata=shown[["total_cost", "net_revenue", "payback", "roi"]],
            text=[spec["text"](value) for value in shown[objective]],
            textposition="outside",
            hovertemplate=(
                "%{y}<br><b>%{customdata[3]:.2f} % net / an</b>"
                "<br>Coût total : %{customdata[0]:,.0f} €"
                "<br>Gain net : %{customdata[1]:,.0f} € / an"
                "<br>Remboursement : %{customdata[2]:.1f} ans<extra></extra>"
            ),
        )
    )
    # Room on the right for the value written after the longest bar.
    figure.update_xaxes(title=spec["axis"], range=[0, float(shown[objective].max()) * 1.18])
    return style_figure(figure, {"l": 175, "r": 55, "t": 20, "b": 45})


def field(label, component, hint=None):
    children = [html.Label(label), component]
    if hint:
        children.append(html.Small(hint))
    return html.Div(children, className="field")


def number_field(label, field_id, minimum, maximum, step, value, hint=None):
    control = html.Div(
        [
            html.Button("-", id=f"{field_id}-minus", n_clicks=0, type="button", **{"aria-label": f"Réduire {label.lower()}"}),
            dcc.Input(
                id=field_id,
                type="text",
                inputMode="decimal" if step < 1 else "numeric",
                value=value,
            ),
            html.Button("+", id=f"{field_id}-plus", n_clicks=0, type="button", **{"aria-label": f"Augmenter {label.lower()}"}),
        ],
        className="number-control",
    )
    return field(label, control, hint)


app = Dash(
    __name__,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
    title="Paris, à la nuit",
    update_title="Calcul en cours…",
)
server = app.server

app.layout = html.Div(
    [
        dcc.Store(id="estimate-location", data=INITIAL_LOCATION),
        dcc.Interval(id="model-check", interval=5_000, n_intervals=0),
        html.Header(
            [
                html.Div("Locations Airbnb à Paris", className="brand"),
                html.Nav(
                    [
                        html.A("Données", href="#explorer"),
                        html.A("Estimation", href="#estimer"),
                        html.A("Investissement", href="#investir"),
                    ]
                ),
            ],
            className="topbar",
        ),
        html.Main(
            [
                html.Section(
                    [
                        html.H1("Données Airbnb", className="page-title"),
                        html.Div(
                            [
                                field("Arrondissements", dcc.Dropdown(id="area-filter", options=AREA_OPTIONS, multi=True, placeholder="Tout Paris")),
                                field("Type de location", dcc.Dropdown(id="room-filter", options=ROOM_OPTIONS, multi=True, placeholder="Tous les types")),
                                field("Chambres", dcc.RangeSlider(id="bedroom-filter", min=0, max=5, step=1, value=[0, 5], marks={i: str(i) if i < 5 else "5+" for i in range(6)})),
                            ],
                            className="filter-bar",
                        ),
                        html.Div(id="market-summary", className="market-summary"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Prix par nuit des annonces", className="panel-label"),
                                        dcc.Graph(id="market-map", config=PLOT_CONFIG, className="market-map"),
                                    ],
                                    className="panel map-panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Répartition des prix", className="panel-label"),
                                        dcc.Graph(id="price-distribution", config={"displayModeBar": False}, className="small-chart"),
                                    ],
                                    className="panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Prix médian par arrondissement", className="panel-label"),
                                        dcc.Graph(id="area-prices", config={"displayModeBar": False}, className="small-chart"),
                                    ],
                                    className="panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Corrélation avec le prix", id="correlation-label", className="panel-label"),
                                        dcc.Graph(id="price-correlations", config={"displayModeBar": False}, className="small-chart"),
                                    ],
                                    className="panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Nombre d’annonces par capacité", className="panel-label"),
                                        dcc.Graph(id="capacity-counts", config={"displayModeBar": False}, className="small-chart"),
                                    ],
                                    className="panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Occupation par arrondissement", className="panel-label"),
                                        dcc.Graph(id="area-occupancy", config={"displayModeBar": False}, className="small-chart"),
                                    ],
                                    className="panel",
                                ),
                            ],
                            className="dashboard-grid",
                        ),
                    ],
                    id="explorer",
                    className="section",
                ),
                html.Section(
                    [
                        html.H2("Estimation du prix par nuit", className="page-title"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Emplacement — cliquez sur la carte", className="panel-label"),
                                        html.Div(
                                            [
                                                html.Div([html.Strong("—", id="monument-score"), html.Span("score monument")]),
                                                html.Div([html.Strong("—", id="nearest-monument"), html.Span("monument le plus proche")]),
                                                html.Div([html.Strong("—", id="monument-distance"), html.Span("distance")]),
                                            ],
                                            className="location-summary",
                                        ),
                                        field("Arrondissement", dcc.Dropdown(id="estimate-area", options=AREA_OPTIONS, value=11, clearable=False)),
                                        dcc.Graph(id="location-map", figure=estimator_map(11, INITIAL_LOCATION), config=PLOT_CONFIG, className="location-map"),
                                    ],
                                    className="panel location-panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Logement", className="panel-label"),
                                        html.Div(
                                            [
                                                number_field("Voyageurs", "estimate-guests", 1, 16, 1, 2),
                                                number_field("Chambres", "estimate-bedrooms", 0, 10, 1, 1),
                                                number_field("Salles de bain", "estimate-bathrooms", 0.5, 10, 0.5, 1),
                                                number_field("Séjour minimum", "estimate-minimum", 1, 365, 1, 2, "En nuits"),
                                                field("Type de bien", dcc.Dropdown(id="estimate-property", options=PROPERTY_OPTIONS, value="Entire rental unit", clearable=False), "Les 12 catégories les plus fréquentes"),
                                                field("Type de location", dcc.Dropdown(id="estimate-room", options=ROOM_OPTIONS, value="Entire home/apt", clearable=False)),
                                            ],
                                            className="form-grid",
                                        ),
                                        html.Button("Estimer le prix par nuit", id="estimate-button", n_clicks=0, className="estimate-submit"),
                                    ],
                                    className="panel form-panel",
                                ),
                                html.Div(
                                    [
                                        html.P("Prix estimé par nuit", className="result-label"),
                                        html.Div("—", id="estimate-value", className="result-value"),
                                        html.P("Modèle non disponible.", id="estimate-message", className="result-message"),
                                        html.Div(
                                            [
                                                html.Div([html.Span("Revenu brut / an"), html.Strong("—", id="annual-revenue")]),
                                                html.Div([html.Span("Prix de vente DVF"), html.Strong("—", id="sale-value")]),
                                                html.Div([html.Span("Remboursement estimé"), html.Strong("—", id="payback-value")]),
                                            ],
                                            className="investment-summary",
                                        ),
                                        html.P("Prix de vente et surface médians de transactions comparables dans l’arrondissement.", id="investment-note", className="investment-note"),
                                        html.Div(
                                            [
                                                html.Span("MODÈLE"),
                                                html.Strong("PRÊT" if MODEL_PATH.exists() else "EN ATTENTE", id="model-status"),
                                            ],
                                            className="model-status",
                                        ),
                                    ],
                                    className="result-panel",
                                ),
                            ],
                            className="estimator-grid",
                        ),
                    ],
                    id="estimer",
                    className="section estimator-section",
                ),
                html.Section(
                    [
                        html.H2("Où investir avec mon budget ?", className="page-title"),
                        html.Div(
                            [
                                field(
                                    "Budget total",
                                    dcc.Input(
                                        id="investment-budget",
                                        type="text",
                                        inputMode="numeric",
                                        value="400000",
                                    ),
                                    "Prix d’achat, notaire et ameublement inclus",
                                ),
                                field(
                                    "Optimiser pour",
                                    dcc.Dropdown(
                                        id="investment-objective",
                                        options=[{"label": spec["label"], "value": key} for key, spec in OBJECTIVES.items()],
                                        value="roi",
                                        clearable=False,
                                    ),
                                ),
                                html.Button(
                                    "Comparer les investissements",
                                    id="investment-button",
                                    n_clicks=0,
                                    className="investment-button",
                                ),
                                html.Div(id="investment-budget-summary", className="budget-summary"),
                            ],
                            className="budget-toolbar",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Meilleurs rendements accessibles", id="investment-ranking-label", className="panel-label"),
                                        dcc.Graph(
                                            id="investment-ranking",
                                            config={"displayModeBar": False},
                                            className="investment-chart",
                                        ),
                                    ],
                                    className="panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Cinq premières options", className="panel-label"),
                                        html.Div(id="investment-table", className="investment-table"),
                                    ],
                                    className="panel",
                                ),
                            ],
                            className="investment-grid",
                        ),
                        html.P(
                            "Rendement indicatif d’un logement entier géré directement. "
                            "Financement, fiscalité et réglementation des locations de courte durée non inclus.",
                            className="method-note",
                        ),
                    ],
                    id="investir",
                    className="section investment-section",
                ),
            ]
        ),
    ]
)


@callback(
    Output("market-map", "figure"),
    Output("price-distribution", "figure"),
    Output("area-prices", "figure"),
    Output("price-correlations", "figure"),
    Output("capacity-counts", "figure"),
    Output("area-occupancy", "figure"),
    Output("correlation-label", "children"),
    Output("market-summary", "children"),
    Input("area-filter", "value"),
    Input("room-filter", "value"),
    Input("bedroom-filter", "value"),
)
def update_market(areas, room_types, bedroom_range):
    frame = LISTINGS
    if areas:
        frame = frame[frame["arrondissement"].isin(areas)]
    if room_types:
        frame = frame[frame["room_type"].isin(room_types)]
    low, high = bedroom_range
    frame = frame[(frame["bedrooms"] >= low) & ((frame["bedrooms"] <= high) if high < 5 else True)]

    if frame.empty:
        empty = blank_figure("Aucune annonce pour ces filtres")
        return (
            empty, empty, empty, empty, empty, empty,
            "Corrélation avec le prix",
            html.P("Élargissez les filtres pour retrouver des annonces.", className="empty-message"),
        )

    median_price = frame["price_quote_price_per_night"].median()
    median_rating = frame["review_scores_rating"].median()
    occupancy = frame["estimated_occupancy_l365d"].median()
    summary = [
        html.Div([html.Strong(f"{len(frame):,.0f}".replace(",", " ")), html.Span("annonces cotées")]),
        html.Div([html.Strong(f"{median_price:,.0f} €".replace(",", " ")), html.Span("prix médian / nuit")]),
        html.Div([html.Strong(f"{median_rating:.2f}" if pd.notna(median_rating) else "—"), html.Span("note médiane / 5")]),
        html.Div([html.Strong(f"{occupancy:,.0f}".replace(",", " ") if pd.notna(occupancy) else "—"), html.Span("nuits occupées / an")]),
    ]
    if areas and len(areas) == 1:
        correlation_label = f"Corrélation avec le prix · {AREA_NAMES[areas[0]]}"
    else:
        correlation_label = "Corrélation avec le prix · sélection actuelle"
    return (
        market_map(frame), price_distribution(frame), area_prices(frame),
        price_correlations(frame), capacity_counts(frame), occupancy_by_area(frame),
        correlation_label, summary,
    )


@callback(Output("model-status", "children"), Input("model-check", "n_intervals"))
def update_model_status(_interval):
    if not MODEL_PATH.exists():
        return "EN ATTENTE"
    try:
        bundle = load_model()
    except Exception:
        return "ERREUR"
    if isinstance(bundle, dict) and bundle.get("name"):
        return f"PRÊT · {bundle['name']}"
    return "PRÊT"


@callback(
    Output("investment-ranking", "figure"),
    Output("investment-ranking-label", "children"),
    Output("investment-budget-summary", "children"),
    Output("investment-table", "children"),
    Input("investment-button", "n_clicks"),
    Input("investment-objective", "value"),
    State("investment-budget", "value"),
)
def compare_investments(_clicks, objective, budget_value):
    objective = objective if objective in OBJECTIVES else "roi"
    panel = OBJECTIVES[objective]["panel"]
    try:
        normalized = str(budget_value).replace(" ", "").replace("\u202f", "").replace("€", "").replace(",", ".")
        budget = float(normalized)
    except (TypeError, ValueError):
        return blank_figure("Budget invalide"), panel, "Saisissez un montant en euros.", ""
    if not 50_000 <= budget <= 10_000_000:
        return blank_figure("Budget hors limites"), panel, "Budget accepté : 50 000 € à 10 000 000 €.", ""
    try:
        options = investment_options(budget)
    except FileNotFoundError as error:
        return blank_figure("Modèle indisponible"), panel, str(error), ""
    except Exception as error:
        return blank_figure("Calcul impossible"), panel, f"Erreur du modèle : {error}", ""
    if options.empty:
        return (
            blank_figure("Aucun bien médian accessible avec ce budget"),
            panel,
            f"Aucune option comparable sous {budget:,.0f} €".replace(",", " "),
            "",
        )

    options = options.sort_values([objective, "sales"], ascending=[False, False])
    euros = lambda value: f"{value:,.0f} €".replace(",", " ")
    best = options.iloc[0]
    summary = html.Div(
        [
            html.Strong(best["label"]),
            html.Span(
                f"{best['roi']:.2f} % net/an · {euros(best['net_revenue'])} net/an · {best['payback']:.1f} ans · "
                f"{euros(best['total_cost'])} tout compris"
            ),
        ]
    )
    columns = [("Option", None), ("Coût total", "total_cost"), ("Prix/nuit", "nightly_price"),
               ("Gain net/an", "net_revenue"), ("Rendement", "roi"), ("Retour", "payback")]
    headings = [f"{title} ▼" if key == objective else title for title, key in columns]
    body = []
    for row in options.head(5).itertuples():
        body.append(
            html.Tr(
                [
                    html.Th(row.label, scope="row"),
                    html.Td(euros(row.total_cost)),
                    html.Td(euros(row.nightly_price)),
                    html.Td(euros(row.net_revenue)),
                    html.Td(f"{row.roi:.2f} %"),
                    html.Td(f"{row.payback:.1f} ans"),
                ]
            )
        )
    table = html.Table(
        [html.Thead(html.Tr([html.Th(heading) for heading in headings])), html.Tbody(body)]
    )
    return investment_chart(options, objective), panel, summary, table


@callback(
    Output("estimate-area", "value"),
    Output("estimate-location", "data"),
    Input("location-map", "clickData"),
    prevent_initial_call=True,
)
def choose_location(click_data):
    if click_data and click_data.get("points"):
        custom = click_data["points"][0].get("customdata")
        if custom is not None and len(custom) >= 4 and custom[0] == "placement":
            area = int(float(custom[3]))
            return area, {
                "latitude": float(custom[1]),
                "longitude": float(custom[2]),
                "arrondissement": area,
            }
    return no_update, no_update


@callback(
    Output("estimate-location", "data", allow_duplicate=True),
    Input("estimate-area", "value"),
    State("estimate-location", "data"),
    prevent_initial_call=True,
)
def center_selected_area(area, location):
    if ctx.triggered_id != "estimate-area":
        return no_update
    if location and location.get("arrondissement") == area:
        return no_update
    center = AREA_CENTERS.loc[AREA_CENTERS["arrondissement"] == area].iloc[0]
    return {
        "latitude": float(center["latitude"]),
        "longitude": float(center["longitude"]),
        "arrondissement": int(area),
    }


@callback(
    Output("location-map", "figure"),
    Output("monument-score", "children"),
    Output("nearest-monument", "children"),
    Output("monument-distance", "children"),
    Input("estimate-area", "value"),
    Input("estimate-location", "data"),
)
def show_selected_location(area, location):
    proximity = monument_proximity(location["latitude"], location["longitude"])
    return (
        estimator_map(area, location),
        f"{proximity['tourist_proximity_score']:.1f} / 100",
        proximity["nearest_monument_name"],
        f"{proximity['nearest_monument_distance_km']:.2f} km",
    )


@callback(
    Output("estimate-guests", "value"),
    Output("estimate-bedrooms", "value"),
    Output("estimate-bathrooms", "value"),
    Output("estimate-minimum", "value"),
    Input("estimate-guests-minus", "n_clicks"),
    Input("estimate-guests-plus", "n_clicks"),
    Input("estimate-bedrooms-minus", "n_clicks"),
    Input("estimate-bedrooms-plus", "n_clicks"),
    Input("estimate-bathrooms-minus", "n_clicks"),
    Input("estimate-bathrooms-plus", "n_clicks"),
    Input("estimate-minimum-minus", "n_clicks"),
    Input("estimate-minimum-plus", "n_clicks"),
    State("estimate-guests", "value"),
    State("estimate-bedrooms", "value"),
    State("estimate-bathrooms", "value"),
    State("estimate-minimum", "value"),
    prevent_initial_call=True,
)
def step_estimate_fields(_gm, _gp, _rm, _rp, _bm, _bp, _mm, _mp, guests, bedrooms, bathrooms, minimum):
    controls = {
        "estimate-guests-minus": (0, -1, 1, 16),
        "estimate-guests-plus": (0, 1, 1, 16),
        "estimate-bedrooms-minus": (1, -1, 0, 10),
        "estimate-bedrooms-plus": (1, 1, 0, 10),
        "estimate-bathrooms-minus": (2, -0.5, 0.5, 10),
        "estimate-bathrooms-plus": (2, 0.5, 0.5, 10),
        "estimate-minimum-minus": (3, -1, 1, 365),
        "estimate-minimum-plus": (3, 1, 1, 365),
    }
    values = [guests, bedrooms, bathrooms, minimum]
    control = controls.get(ctx.triggered_id)
    if control is None:
        return values
    index, change, minimum_value, maximum_value = control
    try:
        current = float(values[index])
    except (TypeError, ValueError):
        current = minimum_value
    updated = round(min(maximum_value, max(minimum_value, current + change)), 10)
    values[index] = int(updated) if float(change).is_integer() else updated
    return values


@callback(
    Output("estimate-value", "children"),
    Output("estimate-message", "children"),
    Output("annual-revenue", "children"),
    Output("sale-value", "children"),
    Output("payback-value", "children"),
    Output("investment-note", "children"),
    Input("estimate-button", "n_clicks"),
    State("estimate-area", "value"),
    State("estimate-guests", "value"),
    State("estimate-bedrooms", "value"),
    State("estimate-bathrooms", "value"),
    State("estimate-minimum", "value"),
    State("estimate-property", "value"),
    State("estimate-room", "value"),
    State("estimate-location", "data"),
    prevent_initial_call=True,
)
def estimate(_clicks, area, guests, bedrooms, bathrooms, minimum_nights, property_type, room_type, location):
    values = [area, guests, bedrooms, bathrooms, minimum_nights, property_type, room_type, location]
    if any(value is None for value in values):
        return "—", "Complétez tous les champs avant de lancer l’estimation.", "—", "—", "—", "Données insuffisantes."
    try:
        guests = int(guests)
        bedrooms = int(bedrooms)
        bathrooms = float(bathrooms)
        minimum_nights = int(minimum_nights)
    except (TypeError, ValueError):
        return "—", "Saisissez uniquement des valeurs numériques valides.", "—", "—", "—", "Données invalides."
    if not (1 <= guests <= 16 and 0 <= bedrooms <= 10 and 0.5 <= bathrooms <= 10 and 1 <= minimum_nights <= 365):
        return "—", "Une valeur numérique dépasse les limites autorisées.", "—", "—", "—", "Données invalides."

    proximity = monument_proximity(location["latitude"], location["longitude"])
    features = {
        "arrondissement": int(area),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
        "accommodates": guests,
        "bedrooms": float(bedrooms),
        "bathrooms": bathrooms,
        "minimum_nights": minimum_nights,
        "property_type": property_type,
        "room_type": room_type,
        "nearest_monument_distance_km": proximity["nearest_monument_distance_km"],
        "tourist_proximity_score": proximity["tourist_proximity_score"],
    }
    try:
        price = predict_price(features)
    except FileNotFoundError as error:
        return "—", str(error), "—", "—", "—", "Modèle indisponible."
    except Exception as error:
        return "Erreur", f"Le modèle n’a pas pu calculer ce logement : {error}", "—", "—", "—", "Calcul impossible."
    euros = lambda value: f"{value:,.0f} €".replace(",", " ")
    message = f"par nuit · {AREA_NAMES[area]} · estimation du modèle"
    # Range saved with the model by modele_lineaire_enrichi/entrainement.py, when present.
    bundle = load_model()
    factors = bundle.get("price_range", {}).get(int(area)) if isinstance(bundle, dict) else None
    if factors:
        message += (f" · fourchette {euros(price * factors[0])} à {euros(price * factors[1])} "
                    f"({bundle.get('range_label', 'la moitié des logements comparables')})")
    try:
        investment = investment_metrics(area, bedrooms, price)
        payback = f"{investment['payback']:.1f} ans" if np.isfinite(investment["payback"]) else "Non rentable"
        note = (
            f"Référence : {investment['sales']} ventes DVF, {investment['surface']:.0f} m² médians, "
            f"{investment['bedrooms']} chambre(s), {investment['occupancy']:.0f} nuits occupées/an. "
            "Remboursement indicatif avec notaire, ameublement et charges, hors financement et réglementation."
        )
        return (
            euros(price), message, euros(investment["gross_revenue"]),
            euros(investment["sale_price"]), payback, note,
        )
    except ValueError as error:
        return euros(price), message, "—", "—", "—", str(error)


if __name__ == "__main__":
    app.run(debug=os.getenv("DASH_DEBUG", "0") == "1")
