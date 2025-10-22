from pathlib import Path
import sys
import pandas as pd

BASE = Path(__file__).parent
OUTPUT_DIR = BASE / "pycaret_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- utilitaire : nettoyage strict annuel ----------
def clean_ts_annual_strict(ts_df: pd.DataFrame, min_points: int = 5):
    """
    ts_df: DataFrame index=DatetimeIndex (annuel), colonne 'y' éventuellement avec NaN.
    Règles:
      - On garde uniquement de la 1ʳᵉ valeur réelle à la dernière valeur réelle (fenêtre valide).
      - Réindexation annuelle complète à l'intérieur de la fenêtre seulement.
      - Interpolation linéaire à l'intérieur (pas d'extrapolation bords).
      - On retourne None si < min_points points après nettoyage.
    """
    ts = ts_df.copy()
    if ts.empty:
        return None

    # indices valides
    first = ts["y"].first_valid_index()
    last  = ts["y"].last_valid_index()
    if first is None or last is None or last <= first:
        return None

    # tronquer à la fenêtre valide
    ts = ts.loc[first:last]

    # fréquence annuelle fin déc + réindexation sur toutes les années de la fenêtre
    ts = ts.asfreq("A-DEC")
    full_idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq="A-DEC")
    ts = ts.reindex(full_idx)

    # interpolation uniquement à l'intérieur (les bords sont déjà sur fenêtre valide)
    ts["y"] = ts["y"].interpolate(method="linear", limit_direction="both")

    # longueur minimale
    if ts["y"].dropna().shape[0] < min_points:
        return None

    return ts

# ---------- 1) CSV long existant ----------
DATA_PATH = BASE / "vaccination_coverage_long.csv"
if not DATA_PATH.exists():
    sys.exit("❌ vaccination_coverage_long.csv introuvable. Lance d'abord: python rebuild_long_strict.py")

# ---------- 2) Charger le long CSV SANS IMPUTATION ----------
df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
print("[INFO] Chargé:", DATA_PATH.name, "lignes:", len(df))

# ---------- 3) Série exemple ----------
SERIE_NAME = "Grippe 65 ans et plus"
one = df[df["serie"] == SERIE_NAME].sort_values("ds").reset_index(drop=True)
if one.empty:
    SERIE_NAME = df["serie"].iloc[0]
    print(f"[WARN] Série '{SERIE_NAME}' choisie par défaut")
    one = df[df["serie"] == SERIE_NAME].sort_values("ds").reset_index(drop=True)

one = one[["ds","couverture_pct"]].rename(columns={"ds":"index","couverture_pct":"y"}).set_index("index")
one = clean_ts_annual_strict(one)
if one is None:
    sys.exit(f"❌ Série '{SERIE_NAME}' trop courte/invalide après nettoyage strict.")

from pycaret.time_series import setup, compare_models, predict_model, finalize_model, save_model

# Annuel => pas de saisonnalité
exp = setup(
    data=one,
    target="y",
    fold=3,
    fh=1,
    seasonal_period=1,
    numeric_imputation_target="linear",  # ok pour imputer de petites lacunes internes
    n_jobs=-1,
    session_id=42,
    verbose=True
)

# modèles sobres, pas de tuning (évite SARIMA saisonnier)
best = compare_models(include=["ets","theta","naive"], sort="MASE")
final_best = finalize_model(best)

fc = predict_model(final_best, fh=1)
print(f"\nPrévision '{SERIE_NAME}' (année+1):\n", fc)

save_model(final_best, OUTPUT_DIR / f"model_{SERIE_NAME.replace(' ', '_')}")
fc.to_csv(OUTPUT_DIR / f"forecast_{SERIE_NAME.replace(' ', '_')}.csv", index=False)

# ---------- 4) Multi-séries propre ----------
all_fc, skipped = [], []

for serie, g in df.groupby("serie"):
    # ignorer d'emblée les séries vides
    if g["couverture_pct"].notna().sum() < 5:
        skipped.append(serie); print(f"[SKIP] {serie}: <5 valeurs réelles"); continue

    ts = g.sort_values("ds")[["ds","couverture_pct"]].rename(columns={"ds":"index","couverture_pct":"y"}).set_index("index")
    ts = clean_ts_annual_strict(ts)
    if ts is None:
        skipped.append(serie); print(f"[SKIP] {serie}: invalide après nettoyage"); continue

    exp_i = setup(
        data=ts, target="y",
        fold=3, fh=1,
        seasonal_period=1,
        numeric_imputation_target="linear",
        n_jobs=-1, session_id=42, verbose=False
    )
    best_i = compare_models(include=["ets","theta","naive"], sort="MASE", turbo=True)
    final_i = finalize_model(best_i)
    fc_i = predict_model(final_i, fh=1)
    fc_i["serie"] = serie
    all_fc.append(fc_i)

out_all = OUTPUT_DIR / "all_series_forecasts.csv"
if all_fc:
    pd.concat(all_fc, ignore_index=True).to_csv(out_all, index=False)
    print("\n[OK] Prévisions multi-séries ->", out_all.resolve())
else:
    print("\n[WARN] Aucune série valide pour la prévision multi-séries.")

if skipped:
    print("[INFO] Séries ignorées:", ", ".join(skipped))