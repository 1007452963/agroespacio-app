# =============================================================================
# AgroEspacio Colombia – Streamlit (versión estable)
# =============================================================================

import streamlit as st
import warnings
import zipfile
import io
import requests
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error
import statsmodels.api as sm
from splot.esda import lisa_cluster
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AgroEspacio Colombia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False

# -----------------------------------------------------------------------------
# Funciones de carga (cacheadas)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Descargando geometría GADM...")
def load_geometry():
    SHP_DIR = Path("/tmp/gadm_col")
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
    return gdf[["dept_id", "departamento", "geometry"]].drop_duplicates("departamento")

@st.cache_data(show_spinner="Descargando datos EVA...")
def fetch_eva():
    def _fetch(u, limit=50000):
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
        _fetch("https://www.datos.gov.co/resource/2pnw-mmge.json"),
        _fetch("https://www.datos.gov.co/resource/uejq-wxrr.json"),
    ], ignore_index=True)
    return raw

def run_pipeline():
    gdf = load_geometry()
    raw = fetch_eva()

    # Normalización
    cols = list(raw.columns)
    c_dept = next((c for c in cols if "departamento" in c.lower() and "codigo" not in c.lower()), None)
    c_year = next((c for c in cols if any(x in c.lower() for x in ["a_o", "ano", "anio", "year", "año"])), None)
    c_prod = next((c for c in cols if "producci" in c.lower()), None)
    c_area = next((c for c in cols if "sembrada" in c.lower()), None)

    eva = pd.DataFrame({
        "departamento": raw[c_dept].astype(str),
        "year": pd.to_numeric(raw[c_year], errors="coerce"),
        "produccion_t": pd.to_numeric(raw[c_prod], errors="coerce").fillna(0),
        "area_sembrada": pd.to_numeric(raw[c_area], errors="coerce").fillna(0),
    })
    eva["departamento"] = (
        eva["departamento"].str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    eva = eva.dropna(subset=["departamento", "year"])

    panel = (
        eva.groupby(["departamento", "year"], as_index=False)
        .agg(agriculture=("produccion_t", "sum"), planted_area=("area_sembrada", "sum"))
    )
    panel = panel[panel["year"].between(2010, 2024)]
    panel = panel[(panel["agriculture"] > 0) | (panel["planted_area"] > 0)]
    panel["yield_tha"] = np.where(panel["planted_area"] > 0,
                                  panel["agriculture"] / panel["planted_area"], np.nan)

    depts_panel = sorted(panel["departamento"].unique())
    gdf = gdf[gdf["departamento"].isin(depts_panel)].drop_duplicates("departamento").sort_values("departamento").reset_index(drop=True)
    depts = list(gdf["departamento"])
    panel = panel[panel["departamento"].isin(depts)].copy()

    # Matriz W
    avg = panel.groupby("departamento")[["agriculture", "planted_area", "yield_tha"]].mean().reindex(depts).reset_index()
    gdf_avg = gdf.merge(avg, on="departamento", how="left").fillna(0)

    candidates = {}
    try:
        candidates["Queen"] = Queen.from_dataframe(gdf, silence_warnings=True)
    except:
        pass
    try:
        candidates["Rook"] = Rook.from_dataframe(gdf, silence_warnings=True)
    except:
        pass
    for k in [3, 4, 5]:
        try:
            candidates[f"KNN_{k}"] = KNN.from_dataframe(gdf, k=k, silence_warnings=True)
        except:
            pass

    scores = {n: 0.0 for n in candidates}
    for var in ["agriculture", "planted_area"]:
        y = gdf_avg[var].values
        for name, w in candidates.items():
            try:
                w.transform = "r"
                mi = Moran(y, w, permutations=99)
                scores[name] += abs(mi.I) * (1.0 if mi.p_sim < 0.05 else 0.25)
            except:
                pass

    W_name = max(scores, key=scores.get) if scores else list(candidates.keys())[0]
    W_opt = candidates[W_name]
    W_opt.transform = "r"

    # SpVAR simple
    rows = []
    for yr in sorted(panel["year"].unique()):
        sub = panel[panel["year"] == yr].set_index("departamento")
        if not all(d in sub.index for d in depts):
            continue
        ya = np.array([sub.loc[d, "agriculture"] for d in depts])
        yp = np.array([sub.loc[d, "planted_area"] for d in depts])
        Wa = lag_spatial(W_opt, ya)
        Wp = lag_spatial(W_opt, yp)
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
                results[label] = {
                    "aic": m.aic,
                    "rho": getattr(m, "rho", 0.0),
                    "params": m.betas.flatten()
                }
            except:
                pass
        if not results:
            return "OLS", {"params": np.zeros(5), "rho": 0.0, "aic": 999}
        best = min(results.items(), key=lambda kv: kv[1]["aic"] if not np.isnan(kv[1]["aic"]) else 1e9)
        return best[0], best[1]

    y1 = panel_sp["agriculture"].values
    X1 = sm.add_constant(panel_sp[["agriculture_l1", "planted_area_l1", "W_agriculture_l1", "W_planted_area_l1"]].values)
    y2 = panel_sp["planted_area"].values
    X2 = sm.add_constant(panel_sp[["planted_area_l1", "agriculture_l1", "W_planted_area_l1", "W_agriculture_l1"]].values)

    spec1, res1 = select_spec(y1, X1, W_opt, "Agri")
    spec2, res2 = select_spec(y2, X2, W_opt, "Area")

    mi_best = Moran(gdf_avg["agriculture"].values, W_opt, permutations=199)

    # Guardar todo en session_state
    st.session_state.gdf_static = gdf
    st.session_state.panel = panel
    st.session_state.gdf_avg = gdf_avg
    st.session_state.W_opt = W_opt
    st.session_state.W_name = W_name
    st.session_state.panel_sp = panel_sp
    st.session_state.res1 = res1
    st.session_state.res2 = res2
    st.session_state.spec1 = spec1
    st.session_state.spec2 = spec2
    st.session_state.depts = depts
    st.session_state.years = sorted(panel["year"].unique().astype(int).tolist())
    st.session_state.mi_best = mi_best
    st.session_state.ready = True

# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:1.2rem; background:linear-gradient(135deg,#0f766e,#134e4a); 
            color:white; border-radius:12px; margin-bottom:1.5rem;">
    <h1 style="margin-bottom:0.3rem;">🌾 AgroEspacio Colombia</h1>
    <p>SpVAR · ESDA · Mapas · 3D | Datos EVA reales</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.ready:
    st.info("Haz clic en el botón para cargar los datos y calibrar el modelo (puede tardar 2-4 minutos).")
    if st.button("🚀 Cargar datos EVA + calibrar modelo", type="primary"):
        with st.spinner("Ejecutando pipeline completo..."):
            run_pipeline()
        st.success("¡Modelo listo!")
        st.rerun()
    st.stop()

# Controles
st.sidebar.header("Controles")
year = st.sidebar.selectbox("Año", st.session_state.years, index=len(st.session_state.years)-1)
variable = st.sidebar.selectbox(
    "Variable",
    ["Producción (t)", "Área sembrada (ha)", "Rendimiento (t/ha)"]
)

tab1, tab2, tab3, tab4 = st.tabs([
    "① Explorar territorio",
    "② Diagnóstico ESDA",
    "③ Inmersión 3D",
    "④ Resultados SpVAR"
])

# ---------- TAB 1: Mapa ----------
with tab1:
    st.subheader(f"Mapa Coroplético – {variable} ({year})")
    
    panel = st.session_state.panel
    gdf = st.session_state.gdf_static
    sub = panel[panel["year"] == year]
    g = gdf.merge(sub, on="departamento", how="left")
    
    col_map = {
        "Producción (t)": ("agriculture", "YlGn"),
        "Área sembrada (ha)": ("planted_area", "YlOrBr"),
        "Rendimiento (t/ha)": ("yield_tha", "plasma"),
    }
    col, cmap = col_map[variable]
    g[col] = g[col].fillna(0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    g.plot(column=col, cmap=cmap, scheme="quantiles", k=7,
           legend=True, ax=ax, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{variable} · {year}", fontweight="bold")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# ---------- TAB 2: ESDA ----------
with tab2:
    st.subheader("Diagnóstico Espacial (ESDA)")
    
    col_map = {
        "Producción (t)": "agriculture",
        "Área sembrada (ha)": "planted_area",
        "Rendimiento (t/ha)": "yield_tha",
    }
    col = col_map[variable]
    y = st.session_state.gdf_avg[col].values
    W = st.session_state.W_opt
    
    mi = Moran(y, W, permutations=199)
    st.markdown(f"""
    - **Moran I** = `{mi.I:.4f}`
    - **p-valor** = `{mi.p_sim:.4f}`
    - **Matriz W usada**: `{st.session_state.W_name}`
    """)
    
    if mi.p_sim < 0.05:
        st.success("Existe autocorrelación espacial significativa.")
    else:
        st.info("No se detectó fuerte dependencia espacial global.")
    
    lisa = Moran_Local(y, W, permutations=299)
    fig, ax = plt.subplots(figsize=(8, 8))
    lisa_cluster(lisa, st.session_state.gdf_avg, p=0.05, ax=ax, legend=True)
    ax.set_title(f"LISA – {variable}")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# ---------- TAB 3: 3D ----------
with tab3:
    st.subheader("Visualización 3D")
    
    g = st.session_state.gdf_static.merge(
        st.session_state.panel[st.session_state.panel["year"] == year],
        on="departamento", how="left"
    )
    g["lon"] = g.geometry.centroid.x
    g["lat"] = g.geometry.centroid.y
    g = g.dropna(subset=["agriculture", "planted_area"])
    
    fig = px.scatter_3d(
        g, x="lon", y="lat", z="agriculture",
        color="yield_tha", size="planted_area",
        hover_name="departamento",
        color_continuous_scale="plasma",
        title=f"3D · {year}",
        labels={"agriculture": "Producción", "yield_tha": "Rendimiento"}
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 4: SpVAR ----------
with tab4:
    st.subheader("Resultados SpVAR")
    
    r1 = st.session_state.res1
    r2 = st.session_state.res2
    p1 = r1["params"]
    p2 = r2["params"]
    
    st.markdown(f"""
    **Especificaciones elegidas**
    - Producción → `{st.session_state.spec1}`
    - Área sembrada → `{st.session_state.spec2}`
    - Matriz W → `{st.session_state.W_name}`
    
    #### Ecuación de Producción
    | Mecanismo | Coeficiente |
    |-----------|-------------|
    | Autodinámica | `{p1[1]:.4f}` |
    | Cruce local (área) | `{p1[2]:.4f}` |
    | Espacial W·producción | `{p1[3]:.4f}` |
    | Espacial cruzada W·área | `{p1[4]:.4f}` |
    | ρ | `{r1.get('rho', 0):.4f}` |
    
    #### Ecuación de Área Sembrada
    | Mecanismo | Coeficiente |
    |-----------|-------------|
    | Autodinámica | `{p2[1]:.4f}` |
    | Cruce local (producción) | `{p2[2]:.4f}` |
    | Espacial W·área | `{p2[3]:.4f}` |
    | Espacial cruzada W·producción | `{p2[4]:.4f}` |
    | ρ | `{r2.get('rho', 0):.4f}` |
    """)
    
    st.info("Los choques agrícolas se propagan entre departamentos vecinos. Las políticas deben diseñarse de forma regional.")

st.markdown("---")
st.caption("AgroEspacio Colombia · Datos EVA reales · SpVAR + ESDA")
