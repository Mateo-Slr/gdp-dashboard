# app/app.py
# ------------------------------------------------------------
# Grippe — visualisation ODISSE (endpoint câblé en dur)
# ------------------------------------------------------------
import io
import json
import time
import requests
import pandas as pd
import streamlit as st

# visus
try:
    import plotly.express as px
except Exception:
    px = None  # on gérera plus bas

st.set_page_config(page_title="ODISSE — Couvertures vaccinales", layout="wide")
st.title("🧬 ODISSE — Couvertures vaccinales (ados & adultes)")

# ===== 1) URL API EN DUR =====
ODISSE_URL = (
    "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/"
    "couvertures-vaccinales-des-adolescent-et-adultes-departement/records?limit=100"
)

# ===== 2) Utilitaires =====
DEP_CANDIDATES = [
    "dep", "code_dep", "departement", "code_departement",
    "code_dpt", "departement_code", "CODDEP", "DEP", "code"
]
YEAR_CANDIDATES = ["annee", "year", "Annee", "ANNEE"]
DATE_CANDIDATES = ["date", "Date", "DATE"]

def _requests_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "streamlit-snowflake-odisse/1.0 (+https://santepubliquefrance.fr)"
    })
    return s

def _friendly_network_error(e: Exception) -> str:
    msg = str(e)
    # cas typiques dans Snowflake sans EAI
    hints = []
    if "Failed to establish a new connection" in msg or "Device or resource busy" in msg or "Errno 16" in msg:
        hints.append("L’accès réseau sortant est bloqué. Active une **External Access Integration** sur `odisse.santepubliquefrance.fr:443` et rattache-la à l’app Streamlit.")
    if "HTTPSConnectionPool" in msg:
        hints.append("Vérifie aussi le DNS/SSL côté plateforme et que l’URL est correcte.")
    return " ".join(hints) or msg

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_odisse(url: str) -> pd.DataFrame:
    """Télécharge l'endpoint ODISSE avec retries + parsers JSON/CSV robustes."""
    s = _requests_session()

    # retries simples (3 tentatives, backoff 1s, 2s)
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

            # fallback CSV si le serveur répond mal son Content-Type
            try:
                return pd.read_csv(io.BytesIO(r.content))
            except Exception:
                raise ValueError("Réponse non JSON/CSV. L’API devrait renvoyer du JSON.")
        except Exception as e:
            last_exc = e
            # dernier essai -> on remonte
            if attempt == 2:
                raise RuntimeError(_friendly_network_error(e)) from e
            time.sleep(1 + attempt)  # backoff

    # ne devrait pas arriver
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
    # Zéro à gauche pour 2 chiffres (hors 2A/2B)
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
    # parse dates “génériques”
    for c in DATE_CANDIDATES:
        if c in out.columns:
            try:
                out[c] = pd.to_datetime(out[c], errors="coerce", utc=False)
            except Exception:
                pass
    # tentative large
    for c in out.columns:
        if out[c].dtype == object:
            try:
                parsed = pd.to_datetime(out[c], errors="ignore")
                if pd.api.types.is_datetime64_any_dtype(parsed):
                    out[c] = parsed
            except Exception:
                pass
    return out

def pick_numeric(df: pd.DataFrame) -> list[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    pref = [c for c in num_cols if any(k in c.lower()
            for k in ["couverture", "grippe", "covid", "hpv", "meningo", "taux", "ratio", "pourcent", "pct", "value"])]
    return pref if pref else num_cols

def show_choropleth_dep(df: pd.DataFrame, dep_col: str, val_col: str, title="Carte par département"):
    try:
        import geopandas as gpd, folium, json as _json
    except Exception as e:
        st.info("Carte non disponible (geopandas/folium non installés).")
        return
    if dep_col is None or val_col is None:
        st.info("Aucune colonne département/valeur trouvée pour la carte.")
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
        merged = gdf_geo.merge(tmp.groupby("dep", as_index=False)["value"].mean(), on="dep", how="left")
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

def show_time_series(df: pd.DataFrame, time_col: str, y_cols: list[str], title="Séries temporelles"):
    if px is None:
        st.info("Plotly non disponible, séries temporelles désactivées.")
        return
    if time_col not in df.columns or not y_cols:
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
    fig = px.line(local, x=time_col, y=y_cols, title=title)
    st.plotly_chart(fig, use_container_width=True)

def safe_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    if px is None:
        return
    if x not in df.columns or y not in df.columns:
        return
    tmp = df[[x, y]].dropna()
    if tmp.empty:
        return
    fig = px.scatter(tmp, x=x, y=y, trendline="ols", title=title)
    st.plotly_chart(fig, use_container_width=True)

# ===== 3) Chargement =====
try:
    raw = fetch_odisse(ODISSE_URL)
except Exception as e:
    st.error(f"Erreur de chargement ODISSE : {e}")
    st.stop()

df, dep_col = norm_dep_code(raw)
df = infer_dates(df)

st.caption(f"Source: ODISSE • Endpoint fixé dans le code • Lignes: {len(df):,}")
st.subheader("Aperçu")
st.dataframe(df.head(50), use_container_width=True)

# ===== 4) Visus automatiques =====
num_cols = pick_numeric(df)
year_col = next((c for c in YEAR_CANDIDATES if c in df.columns), None)
date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)

# (A) Carte par département (moyenne) si possible
val_for_map = num_cols[0] if num_cols else None
if dep_col and val_for_map:
    show_choropleth_dep(df, dep_col, val_for_map, f"Carte — {val_for_map}")

# (B) Série temporelle par année/date, si disponible
time_col = date_col or year_col
if time_col and num_cols:
    y_cols = num_cols[:3]  # max 3 pour lisibilité
    show_time_series(df, time_col, y_cols, "Évolution temporelle (auto)")

# (C) Corrélation simple entre deux 1ères variables numériques
if len(num_cols) >= 2:
    safe_scatter(df, num_cols[0], num_cols[1], f"Corrélation — {num_cols[0]} vs {num_cols[1]}")

# ===== 5) Export rapide =====
st.markdown("---")
st.download_button(
    "⬇️ Export CSV (table courante)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="odisse_couvertures_export.csv",
    mime="text/csv",
)
