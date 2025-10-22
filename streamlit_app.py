# streamlit_app.py
# ------------------------------------------------------------
# ODISSE — visualisations multi-sources (APIs ODISSE + CSV + CSV prédiction)
# ------------------------------------------------------------
import io
import os
import time
import requests
import pandas as pd
import streamlit as st

# Visus (optionnels si non dispo)
try:
    import plotly.express as px
except Exception:
    px = None  # on gérera plus bas

st.set_page_config(page_title="ODISSE — Multi-cartes & Prédictions", layout="wide")


# ===== 1) DATASETS ============================================================
# kind: "api" ou "csv"
# is_prediction: True pour activer l'UI prédiction (KPI, badges, graph dédié)
DATASETS = [
    # --- APIs ODISSE (limitées à 100) ---
    {
        "label": "Couvertures vaccinales — ados & adultes (département)",
        "kind": "api",
        "path": (
            "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/"
            "couvertures-vaccinales-des-adolescent-et-adultes-departement/records?limit=100"
        ),
        "value_pref": ["couverture", "taux", "pct", "pourcent", "value"],
    },
    {
        "label": "Grippe — urgences & SOS Médecins (département)",
        "kind": "api",
        "path": (
            "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/"
            "grippe-passages-aux-urgences-et-actes-sos-medecins-departement/records?limit=100"
        ),
        "value_pref": ["passages", "actes", "nb", "taux", "incidence", "value"],
    },
    {
        "label": "Grippe — urgences & SOS Médecins (France)",
        "kind": "api",
        "path": (
            "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/"
            "grippe-passages-aux-urgences-et-actes-sos-medecins-france/records?limit=100"
        ),
        # dataset national → pas de carte si pas de colonne département
        "value_pref": ["passages", "actes", "nb", "taux", "incidence", "value"],
    },

    # --- CSV locaux (mets-les dans ./data/) ---
    {
        "label": "Couvertures locales 2024 (département)",
        "kind": "csv",
        "path": "data/couverture-2024.csv",
        "value_pref": ["couverture", "taux", "pct", "pourcent", "value"],
    },
    {
        "label": "Campagne 2024 (département)",
        "kind": "csv",
        "path": "data/campagne-2024.csv",
        "value_pref": ["passages", "actes", "nb", "taux", "incidence", "value"],
    },
    {
        "label": "Doses & actes 2024 (département)",
        "kind": "csv",
        "path": "data/doses-actes-2024.csv",
        "value_pref": ["doses", "actes", "nb", "taux", "incidence", "value"],
    },

    # --- CSV PRÉDICTION (année à venir) ---
    # Adapte le nom de fichier ci-dessous à ton CSV réel dans ./data/
    {
        "label": "Prédiction vaccination — année à venir",
        "kind": "csv",
        "path": "data/prediction-vaccination-annee-prochaine.csv",
        "is_prediction": True,
        # colonnes que tu m'as décrites : Serie, année, y_pred, population,
        # dose par schéma (pas essentiel), Doses totales arrondies,
        # Doses avec marges arrondies
        "value_pref": ["y_pred", "taux", "couverture"],  # pour graph lignes/barres
    },
    {
        "label": "Prédiction risque grippe 2025 (régions)",
        "kind": "csv",
        "path": "data/predictions_risque_grippe_2025.csv",
        "is_prediction": True,
        "is_risk_map": True,  # Active le rendu de carte de risque
        "value_pref": ["Score_Risque_Predit_2025"],
    },
]
# ============================================================================

# ===== 2) Utilitaires communs ===============================================
DEP_CANDIDATES = [
    "dep", "code_dep", "departement", "code_departement",
    "code_dpt", "departement_code", "CODDEP", "DEP", "code"
]
YEAR_CANDIDATES = ["annee", "year", "Annee", "ANNEE"]
DATE_CANDIDATES = ["date", "Date", "DATE"]

# Dictionnaire de définitions médicales/techniques
DEFINITIONS = {
    "hpv1": "HPV1 : Première dose du vaccin contre le papillomavirus humain (protection contre cancers et verrues génitales)",
    "hpv2": "HPV2 : Deuxième dose du vaccin HPV (complète le schéma vaccinal 11-14 ans)",
    "hpv3": "HPV3 : Troisième dose du vaccin HPV (schéma de rattrapage 15-19 ans)",
    "dtpolio": "dTPolio : Vaccin diphtérie-tétanos-poliomyélite (rappel adulte)",
    "grippe": "Grippe : Vaccination contre la grippe saisonnière",
    "covid": "COVID-19 : Vaccination contre le coronavirus SARS-CoV-2",
    "meningo": "Méningocoque : Vaccination contre les méningites bactériennes",
    "passages": "Passages aux urgences : Nombre de consultations aux services d'urgences",
    "actes": "Actes SOS Médecins : Nombre d'interventions de SOS Médecins",
    "incidence": "Incidence : Nombre de nouveaux cas pour 100 000 habitants",
    "couverture": "Couverture vaccinale : Pourcentage de la population vaccinée",
    "score_risque": "Score de risque : Indicateur de risque grippe (< 25 = faible, 25-50 = moyen, > 50 = élevé)",
    "risque": "Risque grippe : Probabilité d'augmentation des cas par rapport à l'année précédente",
}

def get_legend_for_data(df: pd.DataFrame) -> list[str]:
    """Détecte les termes médicaux présents dans les données et retourne leurs définitions."""
    legends = []
    cols_lower = [c.lower() for c in df.columns]
    
    for term, definition in DEFINITIONS.items():
        if any(term in col for col in cols_lower):
            legends.append(definition)
    
    return legends

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et normalise les noms de colonnes pour un affichage cohérent."""
    rename_map = {}
    for col in df.columns:
        new_name = col
        # Supprimer préfixes techniques
        if new_name.startswith(("api.", "fields.", "properties.")):
            new_name = new_name.split(".", 1)[1]
        
        # Remplacements standards pour lisibilité
        replacements = {
            "_": " ",
            "code_dep": "Département",
            "code_departement": "Département", 
            "departement": "Département",
            "dep": "Département",
            "annee": "Année",
            "year": "Année",
            "date": "Date",
            "couverture": "Couverture (%)",
            "taux": "Taux (%)",
            "pct": "Pourcentage",
            "y_pred": "Taux prédit (%)",
            "population": "Population",
            "doses": "Doses",
            "actes": "Actes",
            "passages": "Passages",
            "incidence": "Incidence",
            "serie": "Public concerné",
            "Serie": "Public concerné",
        }
        
        for old, new in replacements.items():
            if old in new_name.lower():
                new_name = new_name.replace(old, new)
                break
        
        # Capitaliser première lettre
        new_name = " ".join(word.capitalize() for word in new_name.split())
        rename_map[col] = new_name
    
    return df.rename(columns=rename_map)

def _requests_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "streamlit-odisse/1.0"})
    return s

def _friendly_network_error(e: Exception) -> str:
    msg = str(e)
    tips = []
    if any(k in msg for k in ["Failed to establish a new connection", "Device or resource busy", "Errno 16"]):
        tips.append("Sortie réseau bloquée : dans Snowflake, activer une External Access Integration pour "
                    "`odisse.santepubliquefrance.fr:443` et `raw.githubusercontent.com:443` puis l’attacher à l’app.")
    if "HTTPSConnectionPool" in msg:
        tips.append("Vérifier DNS/SSL plateforme et l’URL.")
    return " ".join(tips) or msg

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_api(url: str) -> pd.DataFrame:
    """Télécharge un endpoint ODISSE (retries + JSON/CSV robustes)."""
    s = _requests_session()
    last_exc = None
    for attempt in range(3):
        try:
            # S’assure que limit=100 est présent
            if "limit=" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}limit=100"

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

@st.cache_data(show_spinner=True)
def fetch_csv(path: str) -> pd.DataFrame:
    """Charge un CSV local (dans le repo)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path} (mets-le dans le repo)")
    
    # Essayer différents séparateurs et encodages
    for sep in (",", ";"):
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                # Vérifier que le CSV a été bien parsé (plus d'une colonne ou colonnes valides)
                if len(df.columns) > 1 or (len(df.columns) == 1 and not df.columns[0].count(";")):
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    
    # Fallback par défaut
    return pd.read_csv(path)

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
                parsed = pd.to_datetime(out[c])  # lève si illisible → on ignore
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
                      "value", "nb", "passages", "actes", "incidence", "doses", "y_pred"])]
    return pref if pref else num_cols

def choose_value_col(df: pd.DataFrame, prefs: list[str] | None) -> str | None:
    if prefs:
        for p in prefs:
            for col in df.columns:
                if col.lower() == p.lower() or p.lower() in col.lower():
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

def show_risk_map_regions(df: pd.DataFrame, region_col: str, risk_col: str, title="Carte de risque grippe par région"):
    """Affiche une carte avec code couleur : <25 = moins de risque (vert), >25 = plus de risque (rouge)."""
    try:
        import geopandas as gpd, folium, json as _json
        from folium import GeoJson
    except Exception:
        st.info("Carte non disponible (geopandas/folium non installés).")
        return
    try:
        # GeoJSON des régions françaises
        geo_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson"
        gdf_geo = gpd.read_file(geo_url)[["nom", "geometry"]].rename(columns={"nom": "region"})
        
        tmp = df[[region_col, risk_col]].dropna().copy()
        tmp.rename(columns={region_col: "region", risk_col: "risk_score"}, inplace=True)
        
        # Normaliser les noms de régions
        tmp["region"] = tmp["region"].str.strip()
        gdf_geo["region"] = gdf_geo["region"].str.strip()
        
        merged = gdf_geo.merge(tmp, on="region", how="left")
        
        # Créer la carte
        m = folium.Map(location=(46.6, 2.5), zoom_start=6, tiles="cartodbpositron")
        
        # Fonction de style basée sur le score de risque
        def style_function(feature):
            risk = feature['properties'].get('risk_score')
            if risk is None:
                return {'fillColor': 'gray', 'color': 'black', 'weight': 1, 'fillOpacity': 0.3}
            elif risk < 25:
                # Moins de risque : vert
                return {'fillColor': 'green', 'color': 'black', 'weight': 1, 'fillOpacity': 0.6}
            else:
                # Plus de risque : rouge (plus foncé si score élevé)
                intensity = min((risk - 25) / 75, 1)  # normaliser entre 25 et 100
                red_val = int(255)
                green_val = int(255 * (1 - intensity))
                color = f'#{red_val:02x}{green_val:02x}00'
                return {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
        
        # Ajouter les régions avec style conditionnel
        GeoJson(
            merged,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['region', 'risk_score'],
                aliases=['Région:', 'Score de risque:'],
                localize=True
            )
        ).add_to(m)
        
        # Légende personnalisée
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                    padding: 10px">
        <p style="margin: 0; font-weight: bold;">Niveau de risque grippe</p>
        <p style="margin: 5px 0;"><span style="background-color: green; padding: 3px 10px; color: white;">■</span> Faible risque (score < 25)</p>
        <p style="margin: 5px 0;"><span style="background-color: orange; padding: 3px 10px; color: white;">■</span> Risque moyen (25-50)</p>
        <p style="margin: 5px 0;"><span style="background-color: red; padding: 3px 10px; color: white;">■</span> Risque élevé (> 50)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        st.subheader(title)
        st.components.v1.html(m._repr_html_(), height=600, scrolling=False)
    except Exception as e:
        st.error(f"Carte de risque indisponible : {e}")

def show_time_series(df: pd.DataFrame, title="Séries temporelles (auto)"):
    if px is None:
        st.info("Plotly non disponible, séries temporelles désactivées.")
        return
    # choisir une colonne temporelle
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
    st.plotly_chart(fig, use_container_width=True)

def get_unit_label(col_name: str) -> str:
    """Détermine l'unité d'une colonne selon son nom."""
    col_lower = col_name.lower()
    if any(term in col_lower for term in ["taux", "couverture", "pct", "pourcent", "%"]):
        return "%"
    elif any(term in col_lower for term in ["population", "hab"]):
        return "habitants"
    elif any(term in col_lower for term in ["doses", "actes", "passages", "nombre", "nb"]):
        return "nombre"
    elif "incidence" in col_lower:
        return "pour 100k hab."
    else:
        return ""

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
    
    # Construire un titre explicite avec les noms de variables et unités
    x_unit = get_unit_label(x)
    y_unit = get_unit_label(y)
    x_label = f"{x} ({x_unit})" if x_unit else x
    y_label = f"{y} ({y_unit})" if y_unit else y
    scatter_title = f"Relation entre {x} et {y}"
    
    # trendline seulement si statsmodels est dispo
    try:
        import statsmodels.api as sm  # noqa: F401
        trend = "ols"
    except Exception:
        trend = None
    
    fig = px.scatter(tmp, x=x, y=y, trendline=trend, title=scatter_title,
                     labels={x: x_label, y: y_label})
    st.plotly_chart(fig, use_container_width=True)

# ===== 3) Rendu spécifique PRÉDICTION =======================================
def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise quelques colonnes usuelles du CSV de prédiction."""
    out = df.copy()
    # Uniformise quelques noms (sans casser si absents)
    rename_map = {}
    for c in out.columns:
        lc = c.strip().lower()
        if lc == "serie":
            rename_map[c] = "Serie"
        elif lc in ("annee", "année", "annee calculée", "année calculée"):
            rename_map[c] = "annee"
        elif lc in ("y_pred", "ŷ", "yhat", "y_hat", "prediction", "prédiction"):
            rename_map[c] = "y_pred"
        elif lc in ("population", "pop"):
            rename_map[c] = "population"
        elif "doses totales" in lc:
            rename_map[c] = "Doses totales arrondies"
        elif "marges" in lc:
            rename_map[c] = "Doses avec marges arrondies"
        elif "dose par schéma" in lc or "doses par schema" in lc:
            rename_map[c] = "dose par schéma"
    if rename_map:
        out = out.rename(columns=rename_map)

    # Types
    if "annee" in out.columns:
        try:
            out["annee"] = pd.to_numeric(out["annee"], errors="coerce").astype("Int64")
        except Exception:
            pass
    for col in ["y_pred", "population", "Doses totales arrondies", "Doses avec marges arrondies"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out

def render_prediction_panel(df: pd.DataFrame, label: str, is_risk_map: bool = False):
    st.markdown("**🟣 Prédiction pour l'année en cours**")
    df = normalize_prediction_columns(df)

    # Filtre année si présence de plusieurs années
    if "annee" in df.columns and df["annee"].notna().any():
        years = sorted({int(x) for x in df["annee"].dropna().unique()})
        year_sel = st.selectbox("Année", options=years, index=len(years)-1 if years else 0)
        df = df[df["annee"] == year_sel]

    # Nettoyer les noms de colonnes et masquer les colonnes vides
    df_display = clean_column_names(df)
    # Supprimer colonnes complètement vides
    df_display = df_display.dropna(axis=1, how='all')
    
    # Afficher légende si termes médicaux détectés
    legends = get_legend_for_data(df)
    if legends:
        with st.expander("ℹ️ Légende et définitions", expanded=False):
            for leg in legends:
                st.markdown(f"- {leg}")
    
    # Formater les colonnes numériques pour un meilleur affichage
    df_display_formatted = df_display.copy()
    for col in df_display_formatted.columns:
        if pd.api.types.is_numeric_dtype(df_display_formatted[col]):
            # Arrondir à 2 décimales pour les nombres
            df_display_formatted[col] = df_display_formatted[col].round(2)
    
    st.dataframe(
        df_display_formatted, 
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                format="%.2f" if pd.api.types.is_float_dtype(df_display_formatted[col]) else "%d"
            ) for col in df_display_formatted.columns if pd.api.types.is_numeric_dtype(df_display_formatted[col])
        }
    )

    # Si carte de risque grippe par régions
    if is_risk_map:
        # Chercher colonne région et score de risque
        region_col = None
        risk_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if "region" in col_lower or col == "Region":
                region_col = col
            if "risque" in col_lower or "risk" in col_lower or "score" in col_lower:
                risk_col = col
        
        if region_col and risk_col:
            # Convertir le score en numérique
            df[risk_col] = pd.to_numeric(df[risk_col], errors='coerce')
            
            show_risk_map_regions(df, region_col, risk_col, "Prédiction risque grippe 2025 par région")
            
            # KPIs spécifiques
            st.markdown("#### 📊 Analyse du risque")
            cols = st.columns(3, gap="large")
            risk_values = df[risk_col].dropna()
            if not risk_values.empty:
                cols[0].metric("Score moyen", f"{risk_values.mean():.1f}")
                cols[1].metric("Régions à risque élevé (>50)", len(risk_values[risk_values > 50]))
                cols[2].metric("Régions à faible risque (<25)", len(risk_values[risk_values < 25]))
        return

    # KPIs si colonnes présentes (pour autres prédictions)
    cols = st.columns(3, gap="large")
    if "y_pred" in df.columns:
        cols[0].metric("Taux prédit (moy.)", f"{df['y_pred'].mean():.2f}")
    if "Doses totales arrondies" in df.columns:
        cols[1].metric("Doses totales (arr.)", f"{int(round(df['Doses totales arrondies'].sum())):,}".replace(",", " "))
    if "Doses avec marges arrondies" in df.columns:
        cols[2].metric("Doses + marge 5% (arr.)", f"{int(round(df['Doses avec marges arrondies'].sum())):,}".replace(",", " "))

    # Graph barres par Serie (public concerné)
    if px is not None and "Serie" in df.columns:
        ycol = "y_pred" if "y_pred" in df.columns else choose_value_col(df, ["taux", "couverture"])
        if ycol:
            fig = px.bar(df.dropna(subset=[ycol]), x="Serie", y=ycol, title="Taux de vaccination par public (%)")
            st.plotly_chart(fig, use_container_width=True)

# ===== 4) Chargement & rendu générique =======================================
def load_dataset(ds: dict) -> pd.DataFrame:
    if ds["kind"] == "api":
        return fetch_api(ds["path"])
    elif ds["kind"] == "csv":
        return fetch_csv(ds["path"])
    else:
        raise ValueError(f"Type de dataset inconnu: {ds['kind']}")

def render_dataset_panel(ds: dict):
    label = ds["label"]
    with st.spinner(f"Chargement — {label}"):
        try:
            raw = load_dataset(ds)
        except Exception as e:
            st.error(f"Erreur de chargement pour « {label} » : {e}")
            return

    # Si dataset de prédiction : UI dédiée, pas de carte
    if ds.get("is_prediction"):
        render_prediction_panel(raw, label, is_risk_map=ds.get("is_risk_map", False))
        return

    # Sinon : pipeline normal
    df, dep_col = norm_dep_code(raw)
    df = infer_dates(df)

    # Nettoyer les noms de colonnes et masquer les colonnes vides
    df_display = clean_column_names(df)
    # Supprimer colonnes complètement vides
    df_display = df_display.dropna(axis=1, how='all')
    
    # Afficher légende si termes médicaux détectés
    legends = get_legend_for_data(df)
    if legends:
        with st.expander("ℹ️ Légende et définitions", expanded=False):
            for leg in legends:
                st.markdown(f"- {leg}")
    
    # Formater les colonnes numériques pour un meilleur affichage
    df_display_formatted = df_display.head(50).copy()
    for col in df_display_formatted.columns:
        if pd.api.types.is_numeric_dtype(df_display_formatted[col]):
            # Arrondir à 2 décimales pour les nombres
            df_display_formatted[col] = df_display_formatted[col].round(2)
    
    st.dataframe(
        df_display_formatted, 
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                format="%.2f" if pd.api.types.is_float_dtype(df_display_formatted[col]) else "%d"
            ) for col in df_display_formatted.columns if pd.api.types.is_numeric_dtype(df_display_formatted[col])
        }
    )

    # Carte uniquement si on a une colonne département
    val_col = choose_value_col(df, ds.get("value_pref"))
    if dep_col and val_col:
        show_choropleth_dep(df, dep_col, val_col, f"{label} — {val_col}")
    else:
        st.info("Pas de couple (département, valeur) détecté pour la carte.")

    # Séries & Corrélation (si données temporelles et numériques)
    show_time_series(df, "Évolution temporelle (auto)")
    safe_scatter(df, "Analyse de corrélation")

# ===== 5) UI — sélection & rendu =============================================
st.markdown("### 🗺️ Sélectionne les datasets à afficher")
options = [ds["label"] for ds in DATASETS]
default_sel = [ds["label"] for ds in DATASETS]  # tout coché par défaut
choices = st.multiselect("Datasets", options=options, default=default_sel)

tabs = st.tabs(choices if choices else ["Aucun dataset"])
if choices:
    for tab, label in zip(tabs, choices):
        with tab:
            ds = next(d for d in DATASETS if d["label"] == label)
            render_dataset_panel(ds)
else:
    with tabs[0]:
        st.info("Sélectionne au moins un dataset pour afficher les visuels.")


