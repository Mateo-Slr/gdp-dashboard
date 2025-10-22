from pathlib import Path
import argparse, math, re, unicodedata
import pandas as pd

def normalize(txt: str) -> str:
    if txt is None: return ""
    # minuscules, sans accents, espaces -> simple espace, retirer ponctuation légère
    t = str(txt).strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"\s+", " ", t)
    t = t.replace("’","'")  # apostrophe typographique -> simple
    return t

def find_header_row(df_raw):
    # Cherche une ligne qui contient un libellé ressemblant à "age revolu" et/ou "ensemble"
    for i in range(min(25, len(df_raw))):
        row = [normalize(x) for x in df_raw.iloc[i].tolist()]
        if any("age revolu" in c or c == "age" or "age" == c for c in row) and any("ensemble" in c or "nombre de femmes" in c or "nombre d'hommes" in c for c in row):
            return i
    return None

def to_int_age(x):
    try:
        s = str(x).strip()
        s = s.replace("ans","").replace("Ans","").strip()
        return int(float(s))
    except:
        return None

def to_num(x):
    if pd.isna(x): return None
    s = str(x).replace("\xa0"," ").replace(" ","").replace(",",".")
    s = re.sub(r"[^0-9.\-eE]", "", s)
    try:
        return float(s)
    except:
        return None

def sum_range(df_age_pop, a_min, a_max):
    sub = df_age_pop[(df_age_pop["AGE"] >= a_min) & (df_age_pop["AGE"] <= a_max)]
    return float(sub["POP"].sum()) if not sub.empty else 0.0

def get_age(df_age_pop, age):
    sub = df_age_pop[df_age_pop["AGE"] == age]
    return float(sub["POP"].sum()) if not sub.empty else 0.0

parser = argparse.ArgumentParser()
parser.add_argument("--insee", required=True, help="Chemin du fichier INSEE (.xls/.xlsx)")
parser.add_argument("--sheet", default=None, help="Nom d'onglet (facultatif)")
parser.add_argument("--forecast", default="pycaret_outputs/all_series_forecasts.csv")
parser.add_argument("--annee", type=int, default=2025)
parser.add_argument("--risk-pct", type=float, default=5.0)
parser.add_argument("--out-pop", default="population_eligible_2025.csv")
parser.add_argument("--out-merge", default="pycaret_outputs/forecast_with_population_2025.csv")
parser.add_argument("--out-doses", default="pycaret_outputs/doses_besoin_2025.csv")
parser.add_argument("--compute-doses", action="store_true")
args = parser.parse_args()

insee_path = Path(args.insee)
xl = pd.ExcelFile(insee_path)
df = xl.parse(args.sheet) if args.sheet else xl.parse(xl.sheet_names[0])

# 1) si les colonnes ne sont pas bonnes, on tente de détecter la ligne d'en-tête
orig_cols = [normalize(c) for c in df.columns]
want_cols = {"age revolu","age","ensemble","nombre d'hommes","nombre de femmes"}
if not (("age revolu" in orig_cols or "age" in orig_cols) and ("ensemble" in orig_cols or "nombre d'hommes" in orig_cols or "nombre de femmes" in orig_cols)):
    # relire sans header et détecter la bonne ligne
    df2 = xl.parse(args.sheet, header=None) if args.sheet else xl.parse(xl.sheet_names[0], header=None)
    hdr_row = find_header_row(df2)
    if hdr_row is None:
        raise ValueError("Impossible de détecter l'en-tête (Âge révolu / Ensemble / Hommes / Femmes).")
    df = xl.parse(args.sheet, header=hdr_row) if args.sheet else xl.parse(xl.sheet_names[0], header=hdr_row)

# 2) normaliser les noms de colonnes
colmap = {normalize(c): c for c in df.columns}
def pick(*cands):
    for cand in cands:
        if cand in colmap: return colmap[cand]
    return None

col_age = pick("age revolu","age")
col_tot = pick("ensemble","total","population")
col_h   = pick("nombre d'hommes","hommes")
col_f   = pick("nombre de femmes","femmes")

if col_age is None:
    raise ValueError("Colonne âge introuvable (Âge révolu / Age).")

# 3) construire tables POP_ALL (ensemble) et POP_F (filles)
tmp = df.copy()
tmp["AGE"] = tmp[col_age].apply(to_int_age)
tmp = tmp.dropna(subset=["AGE"]).copy()
tmp["AGE"] = tmp["AGE"].astype(int)

POP_ALL = None
if col_tot is not None:
    tmp["POP_TOT"] = tmp[col_tot].apply(to_num)
    POP_ALL = tmp[["AGE","POP_TOT"]].dropna().groupby("AGE", as_index=False).sum().rename(columns={"POP_TOT":"POP"})
else:
    if col_h is None or col_f is None:
        raise ValueError("Ni 'Ensemble' ni (Hommes+Femmes) détectés.")
    tmp["H"] = tmp[col_h].apply(to_num)
    tmp["F"] = tmp[col_f].apply(to_num)
    t = tmp.dropna(subset=["H","F"]).copy()
    t["POP"] = t["H"] + t["F"]
    POP_ALL = t.groupby("AGE", as_index=False)["POP"].sum()

POP_F = None
if col_f is not None:
    t = tmp.copy()
    t["F"] = t[col_f].apply(to_num)
    POP_F = t[["AGE","F"]].dropna().groupby("AGE", as_index=False).sum().rename(columns={"F":"POP"})

ANNEE = args.annee
risk_ratio = float(args.risk_pct)/100.0

rows = []
rows.append({"serie":"Grippe 65 ans et plus","annee":ANNEE,"population_eligible":sum_range(POP_ALL,65,150)})
rows.append({"serie":"Grippe moins de 65 ans à risque","annee":ANNEE,"population_eligible":sum_range(POP_ALL,18,64)*risk_ratio})
if POP_F is not None:
    pop_f15 = get_age(POP_F,15); pop_f16 = get_age(POP_F,16)
else:
    # fallback 50/50
    pop_f15 = 0.5*get_age(POP_ALL,15); pop_f16 = 0.5*get_age(POP_ALL,16)
rows.append({"serie":"HPV filles 1 dose à 15 ans","annee":ANNEE,"population_eligible":pop_f15})
rows.append({"serie":"HPV filles 2 doses à 16 ans","annee":ANNEE,"population_eligible":pop_f16})
rows.append({"serie":"Méningocoque C 10-14 ans","annee":ANNEE,"population_eligible":sum_range(POP_ALL,10,14)})
rows.append({"serie":"Méningocoque C 15-19 ans","annee":ANNEE,"population_eligible":sum_range(POP_ALL,15,19)})

pop_df = pd.DataFrame(rows)
pop_df.to_csv(args.out_pop, index=False, encoding="utf-8")

# fusion avec les prévisions
fc = pd.read_csv(args.forecast)
fc.columns = [c.strip() for c in fc.columns]
if "serie" not in fc.columns:
    for c in fc.columns:
        if c.lower()=="series": fc = fc.rename(columns={c:"serie"})
if "y_pred" not in fc.columns:
    for c in fc.columns:
        if c.lower() in ("yhat","forecast","prediction"): fc = fc.rename(columns={c:"y_pred"})

fc["serie"] = fc["serie"].str.strip()
pop_df["serie"] = pop_df["serie"].str.strip()

merged = fc.merge(pop_df, on="serie", how="left")
merged["annee"] = ANNEE
Path(args.out_merge).parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(args.out_merge, index=False, encoding="utf-8")

print(f"✅ Population éligible -> {Path(args.out_pop).resolve()}")
print(f"✅ Fusion prévision + population -> {Path(args.out_merge).resolve()}")

if args.compute_doses:
    df = merged.copy()
    df["doses_par_schema"] = 1.0
    df["couverture_ratio"] = df["y_pred"]/100.0
    df["beneficiaires_couverts"] = df["population_eligible"]*df["couverture_ratio"]
    df["doses_totales"] = df["beneficiaires_couverts"]*df["doses_par_schema"]
    df["doses_totales_arrondies"] = df["doses_totales"].apply(lambda x: int(math.ceil(x if pd.notna(x) else 0)))
    df[["serie","annee","y_pred","population_eligible","doses_par_schema","beneficiaires_couverts","doses_totales","doses_totales_arrondies"]].to_csv(args.out_doses, index=False, encoding="utf-8")
    print(f"✅ Doses estimées -> {Path(args.out_doses).resolve()}")