# =============================================================================
# AgroEspacio Colombia – Streamlit
# SpVAR + ESDA + mapas + 3D | Datos EVA reales
# =============================================================================
# Ejecutar:
#   streamlit run agroespacio_streamlit.py
# =============================================================================

import sys
import subprocess
import importlib
import warnings
import zipfile
import io
import tempfile
import os
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import inv, pinv
from scipy.spatial.distance import pdist, squareform
from libpysal.weights import Queen, Rook, KNN, DistanceBand, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error, GM_Combo
import statsmodels.api as sm
import folium
from splot.esda import lisa_cluster, moran_scatterplot
import plotly.express as px
import plotly.graph_objects as go
import requests

# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroEspacio Colombia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Instalación automática de dependencias (solo si faltan)
# -----------------------------------------------------------------------------
def _install(pkg, imp=None):
    imp = imp or pkg.split("==")[0].split("[")[0]
    try:
        importlib.import_module(imp)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

REQUIRED = [
    ("libpysal", "libpysal"),
    ("esda", "esda"),
    ("spreg", "spreg"),
    ("geopandas", "geopandas"),
    ("statsmodels", "statsmodels"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("folium", "folium"),
    ("mapclassify", "mapclassify"),
    ("splot", "splot"),
    ("plotly", "plotly"),
    ("streamlit-folium", "streamlit_folium"),  # para mapas Folium interactivos
]

with st.spinner("Verificando dependencias…"):
    for pkg, imp in REQUIRED:
        _install(pkg, imp)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Estado de sesión
# -----------------------------------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.update({
        "gdf_static": None,
        "panel": None,
        "gdf_avg": None,
        "W_opt": None,
        "W_name": None,
        "panel_sp": None,
        "res1": None,
        "res2": None,
        "spec1": None,
        "spec2": None,
        "depts": None,
        "ready": False,
        "years": [],
    })

# -----------------------------------------------------------------------------
# Pipeline de carga de datos (idéntico al original)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_pipeline():
    """Carga geometría GADM + EVA y construye SpVAR. Cacheado 1h."""
    progress = st.progress(0, text="Geometría GADM…")

    # --- Geometría ---
    SHP_DIR = Path("./gadm_col")
    SHP_DIR.mkdir(exist_ok=True, parents=True)
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_COL_shp.zip"
    if not list(SHP_DIR.glob("*_1.shp")):
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(SHP_DIR)

    gdf = gpd.read_file(list(SHP_DIR.glob("*_1.shp"))[0])
    gdf = gdf.rename(columns={"NAME_1": "departamento", "GID_1": "dept_id"})
    gdf["departamento"] = (
        gdf["departamento"].str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    gdf = gdf[["dept_id", "departamento", "geometry"]].drop_duplicates("departamento")

    progress.progress(0.25, text="Descargando EVA (datos.gov.co)…")

    def fetch_eva(u, limit=50000):
        frames, offset = [], 0
        while True:
            resp = requests.get(u, params={"$limit": limit, "$offset": offset}, timeout=120)
            resp.raise_for_status()
            chunk = resp.json()
            if not chunk:
                break
            frames.append(pd.DataFrame(chunk))
            if len(chunk) < limit:
                break
            offset += limit
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    raw = pd.concat([
        fetch_eva("https://www.datos.gov.co/resource/2pnw-mmge.json"),
        fetch_eva("https://www.datos.gov.co/resource/uejq-wxrr.json"),
    ], ignore_index=True)

    progress.progress(0.45, text="Normalizando panel…")

    cols = list(raw.columns)
    c_dept = next((c for c in cols if "departamento" in c.lower()
                   and "codigo" not in c.lower() and "c_digo" not in c.lower()), None)
    c_year = next((c for c in cols if c.lower().replace("_", "") in
                   ("ao", "ano", "anio", "año", "year", "a_o") or "a_o" in c.lower()), None)
    c_prod = next((c for c in cols if "producci" in c.lower()), None)
    c_area = next((c for c in cols if "sembrada" in c.lower()), None)

    eva = pd.DataFrame({
        "departamento": raw[c_dept].astype(str),
        "year": raw[c_year],
        "produccion_t": raw[c_prod],
        "area_sembrada": raw[c_area],
    })
    eva["departamento"] = (
        eva["departamento"].str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    for c in ["year", "produccion_t", "area_sembrada"]:
        eva[c] = pd.to_numeric(eva[c], errors="coerce")
    eva["produccion_t"] = eva["produccion_t"].fillna(0)
    eva["area_sembrada"] = eva["area_sembrada"].fillna(0)
    eva = eva.dropna(subset=["departamento", "year"])

    panel = (
        eva.groupby(["departamento", "year"], as_index=False)
        .agg(agriculture=("produccion_t", "sum"), planted_area=("area_sembrada", "sum"))
    )
    panel = panel[panel["year"].between(2010, 2024)]
    panel = panel[(panel["agriculture"] > 0) | (panel["planted_area"] > 0)]
    panel["yield_tha"] = np.where(
        panel["planted_area"] > 0, panel["agriculture"] / panel["planted_area"], np.nan
    )

    depts_panel = sorted(panel["departamento"].unique())
    gdf = (
        gdf[gdf["departamento"].isin(depts_panel)]
        .drop_duplicates("departamento")
        .sort_values("departamento")
        .reset_index(drop=True)
    )
    depts = list(gdf["departamento"])
    panel = panel[panel["departamento"].isin(depts)].copy()

    progress.progress(0.65, text="Seleccionando matriz W óptima…")

    avg = panel.groupby("departamento")[["agriculture", "planted_area", "yield_tha"]].mean().reindex(depts).reset_index()
    gdf_avg = gdf.merge(avg, on="departamento", how="left").fillna(0)

    candidates = {}
    try:
        candidates["Queen"] = Queen.from_dataframe(gdf, silence_warnings=True)
    except Exception:
        pass
    try:
        candidates["Rook"] = Rook.from_dataframe(gdf, silence_warnings=True)
    except Exception:
        pass
    for k in [3, 4, 5]:
        try:
            candidates[f"KNN_{k}"] = KNN.from_dataframe(gdf, k=k, silence_warnings=True)
        except Exception:
            pass

    scores = {n: 0.0 for n in candidates}
    for var in ["agriculture", "planted_area"]:
        y = gdf_avg[var].values
        for name, w in candidates.items():
            w.transform = "r"
            if len(y) != w.n:
                continue
            try:
                mi = Moran(y, w, permutations=199)
                scores[name] += abs(mi.I) * (1.0 if mi.p_sim < 0.05 else 0.25)
            except Exception:
                pass

    W_name = max(scores, key=scores.get)
    W_opt = candidates[W_name]
    W_opt.transform = "r"

    progress.progress(0.80, text="Construyendo SpVAR…")

    rows = []
    for yr in sorted(panel["year"].unique()):
        sub = panel[panel["year"] == yr].set_index("departamento")
        if not all(d in sub.index for d in depts):
            continue
        ya = np.array([sub.loc[d, "agriculture"] for d in depts])
        yp = np.array([sub.loc[d, "planted_area"] for d in depts])
        Wa, Wp = lag_spatial(W_opt, ya), lag_spatial(W_opt, yp)
        for i, d in enumerate(depts):
            rows.append({
                "departamento": d, "year": int(yr),
                "agriculture": ya[i], "planted_area": yp[i],
                "W_agriculture": Wa[i], "W_planted_area": Wp[i],
            })
    sp = pd.DataFrame(rows).sort_values(["departamento", "year"])
    for v in ["agriculture", "planted_area", "W_agriculture", "W_planted_area"]:
        sp[f"{v}_l1"] = sp.groupby("departamento")[v].shift(1)
    panel_sp = sp.dropna().reset_index(drop=True)

    def select_spec(y, X, W, name):
        results = {}
        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)
        for label, builder in [
            ("OLS", lambda: OLS(y, X, name_y=name)),
            ("SAR", lambda: ML_Lag(y, X, W, name_y=name)),
            ("SEM", lambda: ML_Error(y, X, W, name_y=name)),
        ]:
            try:
                m = builder()
                results[label] = dict(
                    aic=m.aic, rho=getattr(m, "rho", None),
                    params=m.betas.flatten(), se=getattr(m, "std_err", None), spec=label
                )
            except Exception:
                pass
        if not results:
            raise RuntimeError("Sin modelos estimables")
        best = min(results.items(), key=lambda kv: kv[1]["aic"] if not np.isnan(kv[1]["aic"]) else 1e9)
        return best[0], best[1]

    y1 = panel_sp["agriculture"].values
    X1 = sm.add_constant(panel_sp[[
        "agriculture_l1", "planted_area_l1", "W_agriculture_l1", "W_planted_area_l1"
    ]].values)
    y2 = panel_sp["planted_area"].values
    X2 = sm.add_constant(panel_sp[[
        "planted_area_l1", "agriculture_l1", "W_planted_area_l1", "W_agriculture_l1"
    ]].values)

    spec1, res1 = select_spec(y1, X1, W_opt, "Agri")
    spec2, res2 = select_spec(y2, X2, W_opt, "Area")

    progress.progress(1.0, text="¡Listo!")
    years = sorted(panel["year"].unique().astype(int).tolist())

    return {
        "gdf_static": gdf,
        "panel": panel,
        "gdf_avg": gdf_avg,
        "W_opt": W_opt,
        "W_name": W_name,
        "panel_sp": panel_sp,
        "res1": res1,
        "res2": res2,
        "spec1": spec1,
        "spec2": spec2,
        "depts": depts,
        "years": years,
    }

# -----------------------------------------------------------------------------
# Funciones de visualización
# -----------------------------------------------------------------------------
def plot_choropleth(year, variable, panel, gdf):
    year = int(year)
    sub = panel[panel["year"] == year]
    g = gdf.merge(sub, on="departamento", how="left")
    col_map = {
        "Producción (t)": ("agriculture", "YlGn", "toneladas"),
        "Área sembrada (ha)": ("planted_area", "YlOrBr", "hectáreas"),
        "Rendimiento (t/ha)": ("yield_tha", "plasma", "t/ha"),
    }
    col, cmap, title_u = col_map[variable]
    g[col] = g[col].fillna(0)

    fig, ax = plt.subplots(figsize=(9, 9))
    g.plot(
        column=col, cmap=cmap, scheme="quantiles", k=7,
        legend=True, ax=ax, edgecolor="black", linewidth=0.35,
        legend_kwds={"loc": "lower left", "title": title_u, "fontsize": 8},
    )
    ax.set_title(f"{variable} · {year}", fontweight="bold", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig

def plot_lisa(variable, gdf_avg, W):
    col_map = {
        "Producción (t)": "agriculture",
        "Área sembrada (ha)": "planted_area",
        "Rendimiento (t/ha)": "yield_tha",
    }
    col = col_map[variable]
    y = gdf_avg[col].values
    lisa = Moran_Local(y, W, permutations=499)

    fig, ax = plt.subplots(figsize=(9, 9))
    lisa_cluster(lisa, gdf_avg, p=0.05, ax=ax, legend=True)
    ax.set_title(f"LISA · {variable}\nHH rojo · LL azul · outliers HL/LH", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig

def moran_card(variable, gdf_avg, W):
    col_map = {
        "Producción (t)": "agriculture",
        "Área sembrada (ha)": "planted_area",
        "Rendimiento (t/ha)": "yield_tha",
    }
    col = col_map[variable]
    mi = Moran(gdf_avg[col].values, W, permutations=499)
    sig = "sí hay dependencia espacial significativa" if mi.p_sim < 0.05 else "no hay evidencia fuerte de dependencia"
    return (
        f"### Diagnóstico espacial — {variable}\n\n"
        f"- **Índice de Moran I** = `{mi.I:.4f}`\n"
        f"- **p-valor simulado** = `{mi.p_sim:.4f}`\n"
        f"- **Interpretación:** {sig}.\n\n"
        f"_Si I > 0 y significativo, los departamentos vecinos se parecen: "
        f"las políticas no pueden diseñarse solo a escala local._"
    )

def plot_3d_static(year, panel, gdf):
    year = int(year)
    g = gdf.merge(panel[panel["year"] == year], on="departamento", how="left")
    g["lon"] = g.geometry.centroid.x
    g["lat"] = g.geometry.centroid.y
    g = g.dropna(subset=["agriculture", "planted_area"])

    fig = px.scatter_3d(
        g, x="lon", y="lat", z="agriculture",
        color="yield_tha", size="planted_area",
        hover_name="departamento",
        color_continuous_scale="plasma",
        title=f"Grafología 3D · {year}",
        labels={"agriculture": "Producción (t)", "yield_tha": "Rendimiento", "planted_area": "Área"},
    )
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=50, b=0))
    return fig

def plot_3d_anim(panel, gdf):
    cent = gdf.copy()
    cent["lon"] = cent.geometry.centroid.x
    cent["lat"] = cent.geometry.centroid.y
    p = panel.merge(cent[["departamento", "lon", "lat"]], on="departamento")
    p = p.dropna(subset=["agriculture", "planted_area", "yield_tha"])

    fig = px.scatter_3d(
        p, x="lon", y="lat", z="agriculture",
        color="yield_tha", size="planted_area",
        animation_frame="year", hover_name="departamento",
        color_continuous_scale="plasma",
        range_z=[0, p["agriculture"].max() * 1.05],
        title="Evolución 3D dinámica EVA",
    )
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
    return fig

def spvar_story(res1, res2, spec1, spec2, W_name):
    p1, p2 = res1["params"], res2["params"]
    rho1 = res1["rho"] if res1["rho"] is not None else 0.0
    rho2 = res2["rho"] if res2["rho"] is not None else 0.0

    return f"""
### Resultados SpVAR — narrativa para decisión

**Especificaciones elegidas**
- Producción → `{spec1}`
- Área sembrada → `{spec2}`
- Matriz W → `{W_name}`

#### Ecuación de producción
| Mecanismo                     | Coeficiente   |
|-------------------------------|---------------|
| Autodinámica (propia)         | `{p1[1]:.4f}` |
| Cruce local (área)            | `{p1[2]:.4f}` |
| Espacial directa W·producción | `{p1[3]:.4f}` |
| Espacial cruzada W·área       | `{p1[4]:.4f}` |
| ρ contemporáneo               | `{rho1:.4f}`  |

#### Ecuación de área sembrada
| Mecanismo                     | Coeficiente   |
|-------------------------------|---------------|
| Autodinámica (propia)         | `{p2[1]:.4f}` |
| Cruce local (producción)      | `{p2[2]:.4f}` |
| Espacial directa W·área       | `{p2[3]:.4f}` |
| Espacial cruzada W·producción | `{p2[4]:.4f}` |
| ρ contemporáneo               | `{rho2:.4f}`  |

**Mensaje clave para stakeholders**  
Los efectos no son solo “del departamento”: lo que ocurre en los vecinos  
(área y producción) entra en la dinámica. Útil para focalizar programas  
de extensión agrícola, infraestructura rural y ordenamiento territorial.
"""

def build_folium(year, variable, panel, gdf):
    year = int(year)
    col_map = {
        "Producción (t)": ("agriculture", "YlGn"),
        "Área sembrada (ha)": ("planted_area", "YlOrBr"),
        "Rendimiento (t/ha)": ("yield_tha", "YlOrRd"),
    }
    col, fill = col_map[variable]
    g = gdf.merge(panel[panel["year"] == year], on="departamento", how="left")
    g[col] = g[col].fillna(0)
    g["lon"] = 
