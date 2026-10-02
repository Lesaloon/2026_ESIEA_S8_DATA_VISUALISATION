"""Interactive Paris rental market dashboard and nightly-price estimator."""

import os
import pickle
import sqlite3
from functools import lru_cache
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


LISTINGS, AREAS, MONUMENTS = load_market()
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
            marker={"size": 13, "color": "rgba(0,0,0,0.01)"},
            customdata=np.column_stack(
                [
                    np.full(len(PLACEMENT_GRID), "placement"),
                    PLACEMENT_GRID[["latitude", "longitude", "arrondissement"]].to_numpy(),
                ]
            ),
            hoverinfo="skip",
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
        clickmode="event+select",
        map={"zoom": 10.5, "center": {"lat": center["latitude"], "lon": center["longitude"]}},
        uirevision=f"estimator-map-{area}",
    )
    return figure


@lru_cache(maxsize=1)
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


def field(label, component, hint=None):
    children = [html.Label(label), component]
    if hint:
        children.append(html.Small(hint))
    return html.Div(children, className="field")


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
        html.Header(
            [
                html.Div("Locations Airbnb à Paris", className="brand"),
                html.Nav([html.A("Données", href="#explorer"), html.A("Estimation", href="#estimer")]),
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
                                        dcc.Graph(id="location-map", figure=estimator_map(11, INITIAL_LOCATION), config=PLOT_CONFIG, className="location-map"),
                                        field("Arrondissement", dcc.Dropdown(id="estimate-area", options=AREA_OPTIONS, value=11, clearable=False)),
                                        html.Div(
                                            [
                                                html.Div([html.Strong("—", id="monument-score"), html.Span("score monument")]),
                                                html.Div([html.Strong("—", id="nearest-monument"), html.Span("monument le plus proche")]),
                                                html.Div([html.Strong("—", id="monument-distance"), html.Span("distance")]),
                                            ],
                                            className="location-summary",
                                        ),
                                    ],
                                    className="panel location-panel",
                                ),
                                html.Div(
                                    [
                                        html.Div("Logement", className="panel-label"),
                                        html.Div(
                                            [
                                                field("Voyageurs", dcc.Input(id="estimate-guests", type="number", min=1, max=16, step=1, value=2)),
                                                field("Chambres", dcc.Input(id="estimate-bedrooms", type="number", min=0, max=10, step=1, value=1)),
                                                field("Salles de bain", dcc.Input(id="estimate-bathrooms", type="number", min=0.5, max=10, step=0.5, value=1)),
                                                field("Séjour minimum", dcc.Input(id="estimate-minimum", type="number", min=1, max=365, step=1, value=2), "En nuits"),
                                                field("Type de bien", dcc.Dropdown(id="estimate-property", options=PROPERTY_OPTIONS, value="Entire rental unit", clearable=False), "Les 12 catégories les plus fréquentes"),
                                                field("Type de location", dcc.Dropdown(id="estimate-room", options=ROOM_OPTIONS, value="Entire home/apt", clearable=False)),
                                            ],
                                            className="form-grid",
                                        ),
                                        html.Button("Estimer le prix par nuit", id="estimate-button", n_clicks=0),
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
            ]
        ),
    ]
)


@callback(
    Output("market-map", "figure"),
    Output("price-distribution", "figure"),
    Output("area-prices", "figure"),
    Output("price-correlations", "figure"),
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
        return empty, empty, empty, empty, "Corrélation avec le prix", html.P("Élargissez les filtres pour retrouver des annonces.", className="empty-message")

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
        price_correlations(frame), correlation_label, summary,
    )


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
    Output("estimate-value", "children"),
    Output("estimate-message", "children"),
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
        return "—", "Complétez tous les champs avant de lancer l’estimation."

    proximity = monument_proximity(location["latitude"], location["longitude"])
    features = {
        "arrondissement": int(area),
        "latitude": float(location["latitude"]),
        "longitude": float(location["longitude"]),
        "accommodates": int(guests),
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "minimum_nights": int(minimum_nights),
        "property_type": property_type,
        "room_type": room_type,
        "nearest_monument_distance_km": proximity["nearest_monument_distance_km"],
        "tourist_proximity_score": proximity["tourist_proximity_score"],
    }
    try:
        price = predict_price(features)
    except FileNotFoundError as error:
        return "—", str(error)
    except Exception as error:
        return "Erreur", f"Le modèle n’a pas pu calculer ce logement : {error}"
    return f"{price:,.0f} €".replace(",", " "), f"par nuit · {AREA_NAMES[area]} · estimation du modèle"


if __name__ == "__main__":
    app.run(debug=os.getenv("DASH_DEBUG", "0") == "1")
