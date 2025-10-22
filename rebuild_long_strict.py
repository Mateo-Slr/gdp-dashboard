from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
RAW = sorted(list(BASE.glob("couvertures*.csv")) + list(BASE.glob("*vaccin*.csv")))
assert RAW, "Aucun CSV source trouvé."
RAW = RAW[0]

OUT = BASE / "vaccination_coverage_long.csv"

df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]

# Normaliser 'Année'
if "Année" not in df.columns:
    for alt in ["Annee","annee","année"]:
        if alt in df.columns:
            df = df.rename(columns={alt:"Année"})
            break
assert "Année" in df.columns, "Colonne 'Année' absente."

# NE PAS interpoler ici : on garde les NaN d'origine
value_cols = [c for c in df.columns if c != "Année"]
for c in value_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("Année").reset_index(drop=True)

# Long format sans aucune imputation
long_df = df.melt(id_vars=["Année"], value_vars=value_cols,
                  var_name="serie", value_name="couverture_pct")

# Datetime de fin d'année
long_df["ds"] = pd.to_datetime(long_df["Année"].astype(str) + "-12-31")

# Ordonner et sauver
long_df = long_df[["ds","Année","serie","couverture_pct"]].sort_values(["serie","ds"])
long_df.to_csv(OUT, index=False)
print("[OK] long strict ->", OUT.resolve())