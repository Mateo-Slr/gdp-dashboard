# streamlit_app.py
# ------------------------------------------------------------
# ODISSE — visualisations multi-datasets (cartes & visus auto)
# ------------------------------------------------------------
import io
import time
import requests
import pandas as pd
import streamlit as st

# Visus (optionnels si non dispo)
try:
    import plotly.express as px
except Exception:
    px = None  # on gérera plus bas

st.set_page_config(page_title="ODISSE — Multi-cartes", layout="wide")
st.title("🧬 ODISSE — Couvertures & Grippe (multi-cartes)")

# ===== 1) LISTE D'ENDPOINTS (ajoute/modifie ici) ==============================
ODISSE_ENDPOINTS = [
    {
        "label": "Couvertures vaccinales — ados & adultes (département)",
        "url": (
            "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/"
            "couvertures-vaccinales-des-adolescent-et-adultes-departement/records?limit=100"
        ),
        # priorité pour choisir la colonne de valeur sur la carte
        "value_pref": ["couverture", "taux", "pct", "pourcent", "value"],
    },
    {
        "label": "Grippe — passages aux urgences & actes SOS Médecins (département)",
        "url": (
            "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/"
            "grippe-passages-aux-urgences-et-actes-sos-medecins-departement/records?limit=100"
        ),
        # priorités possibles selon le schéma du jeu (nombre, taux, incidence…)
        "value_pref": ["passages", "actes", "nb", "taux", "incidence", "value"],
    },
]
# ==============================================================================

# ===== 2) Utilitaires =========================================================
DEP_CANDIDATES = [
    "dep", "code_dep", "departement", "code_departement",
    "code_dpt", "departement_code", "CODDEP", "DEP", "code"
]
YEAR_CANDIDATES = ["annee", "year", "Annee", "ANNEE"]
DATE_CANDIDATES = ["date", "Date", "DATE"]

def _requests_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "streamlit-odisse/1.0"})
    return s

def _friendly_network_error(e: Exception) -> str:
    msg = str(e)
    tips = []
    if any(k in msg for k in ["Failed to establish a new connection", "Device or resource busy", "Errno 16"]):
        tips.append("Sortie réseau bloquée : active une External Access Integration pour `odisse.santepubliquefrance.fr:443` et rattache-la à l’app.")
    if "HTTPSConnectionPool" in msg:
        tips.append("Vérifie le DNS/SSL plateforme et l’URL.")
    return " ".join(tips) or msg

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_odisse(url: str) -> pd.DataFrame:
    """Télécharge l'endpoint ODISSE avec retries + parsers JSON/CSV robustes."""
    s = _requests_session()
    last_exc = None
    for attempt in range(3):
        try:
            r = s.get(url, timeout=(5, 30))
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()

            if "json" in ct or url.endswith(".json"):
                data = r.json()
                if isinstance(data, dict) and "results" in data:
                    return pd.json_normalize(data["results"])
                if isinstance(data, list):
                    return pd.json_normalize(data)
                return pd.json_normalize(data)

            # fallback CSV si mauvais content-type
            try:
                return pd.read_csv(io.BytesIO(r.content))
            except Exception:
                raise ValueError("Réponse non JSON/CSV. L’API devrait renvoyer du JSON.")
        except Exception as e:
            last_exc = e
            if attempt == 2:
                raise RuntimeError(_friendly_network_error(e)) from e
            time.sleep(1 + attempt)  # backoff
    raise RuntimeError(_friendly_network_error(last_exc or Exception("Erreur inconnue")))

def norm_dep_code(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    dep_col = None
    lower_map = {c.lower(): c for c in df.columns}
    for cand in DEP_CANDIDATES:
        if cand in df.columns or cand.lower() in lower_map:
            dep_col = cand if cand in df.columns else lower_map[cand.lower()]
            break
    if dep_col is None:
        return df, None
    out = df.copy()
    out.rename(columns={dep_col: "dep"}, inplace=True)
    out["dep"] = out["dep"].astype(str).str.strip()
    out["dep"] = out["dep"].apply(
        lambda x: x if x.upper() in ["2A", "2B"]
        else (x.zfill(2) if x.isdigit() and len(x) <= 2 else x)
    )
    return out, "dep"

def infer_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # parse années
    for c in YEAR_CANDIDATES:
        if c in out.columns:
            try:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
            except Exception:
                pass
    # parse dates génériques
    for c in DATE_CANDIDATES:
        if c in out.columns:
            try:
                out[c] = pd.to_datetime(out[c], errors="coerce")
            except Exception:
                pass
    # tentative large prudente
    for c in out.columns:
        if out[c].dtype == object:
            try:
                parsed = pd.to_datetime(out[c])  # laisse lever si vraiment illisible
                if pd.api.types.is_datetime64_any_dtype(parsed):
                    out[c] = parsed
            except Exception:
                pass
    return out

def pick_numeric(df: pd.DataFrame) -> list[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    pref = [c for c in num_cols if any(k in c.lower()
            for k in ["couverture", "grippe", "covid", "hpv",
                      "meningo", "taux", "ratio", "pourcent", "pct",
                      "value", "nb", "passages", "actes", "incidence"])]
    return pref if pref else num_cols

def choose_value_col(df: pd.DataFrame, prefs: list[str] | None) -> str | None:
    if prefs:
        for p in prefs:
            for col in df.columns:
                if col.lower() == p.lower():
                    return col
                # match contains
                if p.lower() in col.lower():
                    return col
    num_cols = pick_numeric(df)
    return num_cols[0] if num_cols else None

def show_choropleth_dep(df: pd.DataFrame, dep_col: str, val_col: str, title="Carte par département"):
    try:
        import geopandas as gpd, folium, json as _json
    except Exception:
        st.info("Carte non disponible (geopandas/folium non installés).")
        return
    try:
        geo_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
        gdf_geo = gpd.read_file(geo_url)[["code", "geometry"]].rename(columns={"code": "dep"})
        tmp = df[[dep_col, val_col]].dropna().copy()
        tmp.rename(columns={dep_col: "dep", val_col: "value"}, inplace=True)
        tmp["dep"] = tmp["dep"].astype(str).str.strip().apply(
            lambda x: x if x.upper() in ["2A", "2B"]
            else (x.zfill(2) if x.isdigit() and len(x) <= 2 else x)
        )
        merged = gdf_geo.merge(
            tmp.groupby("dep", as_index=False)["value"].mean(), on="dep", how="left"
        )
        m = folium.Map(location=(46.6, 2.5), zoom_start=5, tiles="cartodbpositron")
        folium.Choropleth(
            geo_data=_json.loads(merged.to_json()),
            data=merged,
            columns=["dep", "value"],
            key_on="feature.properties.dep",
            fill_opacity=0.8,
            line_opacity=0.4,
            legend_name=val_col,
        ).add_to(m)
        st.subheader(title)
        st.components.v1.html(m._repr_html_(), height=600, scrolling=False)
    except Exception as e:
        st.error(f"Carte indisponible : {e}")

def show_time_series(df: pd.DataFrame, title="Séries temporelles (auto)"):
    if px is None:
        st.info("Plotly non disponible, séries temporelles désactivées.")
        return
    # choisir time_col
    time_col = None
    for c in DATE_CANDIDATES + YEAR_CANDIDATES:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        return
    local = df.dropna(subset=[time_col]).copy()
    if not pd.api.types.is_datetime64_any_dtype(local[time_col]):
        if pd.api.types.is_integer_dtype(local[time_col]) or local[time_col].astype(str).str.fullmatch(r"\d{4}").any():
            try:
                local[time_col] = pd.to_datetime(local[time_col].astype(str) + "-01-01", errors="coerce")
            except Exception:
                pass
        else:
            try:
                local[time_col] = pd.to_datetime(local[time_col], errors="coerce")
            except Exception:
                pass
    local = local.dropna(subset=[time_col]).sort_values(time_col)
    if local.empty:
        return
    y_cols = pick_numeric(local)[:3]
    if not y_cols:
        return
    fig = px.line(local, x=time_col, y=y_cols, title=title)
    st.plotly_chart(fig, width="stretch")

def safe_scatter(df: pd.DataFrame, title: str):
    if px is None:
        return
    num_cols = pick_numeric(df)
    if len(num_cols) < 2:
        return
    x, y = num_cols[0], num_cols[1]
    tmp = df[[x, y]].dropna()
    if tmp.empty:
        return
    # trendline seulement si statsmodels est dispo
    try:
        import statsmodels.api as sm  # noqa: F401
        trend = "ols"
    except Exception:
        trend = None
    fig = px.scatter(tmp, x=x, y=y, trendline=trend, title=title)
    st.plotly_chart(fig, width="stretch")

def render_dataset_panel(ep: dict):
    url = ep["url"]
    label = ep["label"]
    with st.spinner(f"Chargement — {label}"):
        try:
            raw = fetch_odisse(url)
        except Exception as e:
            st.error(f"Erreur de chargement pour « {label} » : {e}")
            return

    df, dep_col = norm_dep_code(raw)
    df = infer_dates(df)

    st.caption(f"Source: ODISSE • {label} • Lignes: {len(df):,}")
    st.dataframe(df.head(50), use_container_width=True)

    # Choix de la valeur pour la carte
    val_col = choose_value_col(df, ep.get("value_pref"))

    # Carte
    if dep_col and val_col:
        show_choropleth_dep(df, dep_col, val_col, f"{label} — {val_col}")
    else:
        st.info("Pas de couple (département, valeur) détecté pour la carte.")

    # Séries & Corrélation
    show_time_series(df, "Évolution temporelle (auto)")
    safe_scatter(df, "Corrélation (2 premières numériques)")

# ===== 3) UI — sélection & rendu =============================================
st.markdown("### 🗺️ Sélectionne les datasets à afficher")
choices = st.multiselect(
    "Datasets ODISSE",
    options=[ep["label"] for ep in ODISSE_ENDPOINTS],
    default=[ODISSE_ENDPOINTS[0]["label"], ODISSE_ENDPOINTS[1]["label"]],
)

tabs = st.tabs(choices if choices else ["Aucun dataset"])
if choices:
    for tab, label in zip(tabs, choices):
        with tab:
            ep = next(ep for ep in ODISSE_ENDPOINTS if ep["label"] == label)
            render_dataset_panel(ep)
else:
    with tabs[0]:
        st.info("Sélectionne au moins un dataset pour afficher les visuels.")

st.markdown("---")

