import pandas as pd
import math

# ---- Fichiers d'entrée / sortie ----
INPUT = "pycaret_outputs/forecast_with_population_2025.csv"
OUTPUT = "pycaret_outputs/doses_final_2025.csv"

# ---- Lecture ----
df = pd.read_csv(INPUT)

# ---- Étape 1 : nombre de doses par schéma ----
df["doses_par_schema"] = 1.0  # par défaut 1 dose
df.loc[df["serie"].str.contains("HPV filles 2 doses", case=False, na=False), "doses_par_schema"] = 2.0

# ---- Étape 2 : calcul des doses ----
df["couverture_ratio"] = df["y_pred"] / 100.0
df["doses_totales"] = df["population_eligible"] * df["couverture_ratio"] * df["doses_par_schema"]

# ---- Étape 3 : marge de sécurité (5 %) ----
MARGE = 0.05
df["doses_avec_marge"] = df["doses_totales"] * (1 + MARGE)

# ---- Arrondir à l'entier supérieur ----
df["doses_totales_arrondies"] = df["doses_totales"].apply(lambda x: math.ceil(x))
df["doses_avec_marge_arrondies"] = df["doses_avec_marge"].apply(lambda x: math.ceil(x))

# ---- Sauvegarde ----
cols = ["serie", "annee", "y_pred", "population_eligible", "doses_par_schema",
        "doses_totales_arrondies", "doses_avec_marge_arrondies"]
df[cols].to_csv(OUTPUT, index=False, encoding="utf-8")

print(f"✅ Fichier créé : {OUTPUT}")