import pandas as pd
from pathlib import Path

SRC = Path("couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv")
OUT = Path("vaccination_coverage_long.csv")

df = pd.read_csv(SRC)
df.columns = [c.strip() for c in df.columns]
assert "Année" in df.columns, "Colonne 'Année' absente du CSV"

value_cols = [c for c in df.columns if c != "Année"]
for c in value_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("Année").reset_index(drop=True)
df[value_cols] = df[value_cols].interpolate(method="linear", limit_direction="both")

long_df = df.melt(id_vars=["Année"], value_vars=value_cols,
                  var_name="serie", value_name="couverture_pct")
long_df["ds"] = pd.to_datetime(long_df["Année"].astype(str) + "-12-31")
long_df = long_df[["ds", "Année", "serie", "couverture_pct"]].sort_values(["serie","ds"])
long_df.to_csv(OUT, index=False)
print("OK ->", OUT.resolve())
