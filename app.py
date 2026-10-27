# =============================================================================
# AgroEspacio Colombia – Streamlit
# SpVAR + ESDA + mapas + 3D | Datos EVA reales
# =============================================================================

import streamlit as st
import sys, subprocess, importlib, warnings, zipfile, io, requests, os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import inv, pinv
from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error
import statsmodels.api as sm
from splot.esda import lisa_cluster
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AgroEspacio Colombia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# INSTALACIÓN DE DEPENDENCIAS (solo la primera vez)
# -----------------------------------------------------------------------------
def ensure_packages():
    packages = [
        ("libpysal", "libpysal"),
        ("esda", "esda"),
        ("spreg", "spreg"),
        ("geopandas", "geopandas"),
        ("mapclassify", "mapclassify"),
        ("splot", "splot"),
    ]
    for pkg, imp in packages:
        try:
            importlib.import_module(imp)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_packages()

# -----------------------------------------------------------------------------
# ESTADO (session_state)
# -----------------------------------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False
    st.session_state.gdf_static = None
    st.session_state.panel = None
    st.session_state.gdf_avg = None
    st.session_state.W_opt = None
    st.session_state.W_name = None
    st.session_state.panel_sp = None
    st.session_state.res1 = None
    st.session_state.res2 = None
    st.session_state.spec1 = None
    st.session_state.spec2 = None
    st.session_state.depts = None
    st.session_state.auto_insight = ""
    st.session_state.years = []

# -----------------------------------------------------------------------------
# PIPELINE DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
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
    gdf = gdf[["dept_id", "departamento", "geometry"]].drop_duplicates("departamento")
    return gdf

@st.cache_data(show_spinner=False)
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
    progress = st.progress(0, text="Iniciando pipeline...")
    
    # 1. Geometría
    progress.progress(10, text="Cargando geometría GADM...")
    gdf = load_geometry()
    
    # 2. Datos EVA
    progress.progress(30, text="Descargando datos EVA (datos.gov.co)...")
    raw = fetch_eva()
    
    progress.progress(50, text="Normalizando panel...")
    cols = list(raw.columns)
    c_dept = next((c for c in cols if "departamento" in c.lower() and "codigo" not in c.lower() and "c_digo" not in c.lower()), None)
    c_year = next((c for c in cols if c.lower().replace("_", "") in ("ao", "ano", "anio", "año", "year", "a_o") or "a_o" in c.lower()), None)
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
    gdf = gdf[gdf["departamento"].isin(depts_panel)].drop_duplicates("departamento").sort_values("departamento").reset_index(drop=True)
    depts = list(gdf["departamento"])
    panel = panel[panel["departamento"].isin(depts)].copy()
    
    # 3. Matriz W y ESDA
    progress.progress(70, text="Seleccionando matriz W óptima y ESDA...")
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
            w.transform = "r"
            if len(y) != w.n:
                continue
            try:
                mi = Moran(y, w, permutations=199)
                scores[name] += abs(mi.I) * (1.0 if mi.p_sim < 0.05 else 0.25)
            except:
                pass
    
    W_name = max(scores, key=scores.get) if scores else "Queen"
    W_opt = candidates.get(W_name)
    if W_opt:
        W_opt.transform = "r"
    
    # 4. SpVAR
    progress.progress(85, text="Construyendo SpVAR...")
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
    
    # Insight automático
    mi_best = Moran(gdf_avg["agriculture"].values, W_opt, permutations=199)
    auto_insight = f"""
### 🧠 Interpretabilidad Automática & Diagnóstico Territorial
- **Efecto Espacial Global (Moran I):** `{mi_best.I:.4f}` (p-valor: `{mi_best.p_sim:.4f}`).
- *Interpretación:* {'Existe autocorrelación espacial positiva fuerte; la producción agrícola tiende a concentrarse en clusters regionales.' if mi_best.p_sim < 0.05 else 'Distribución sin fuerte clustering global dominante.'}
- **Dinámica SpVAR Óptima:**
  - Producción modelada mediante **{spec1}**
  - Área sembrada modelada mediante **{spec2}**
  - Matriz de vecindad seleccionada: **{W_name}**
- **Implicación de Negocio:** Las políticas de incentivo a la producción y extensión agrícola no deben aislarse por departamento, ya que los choques y el rendimiento se propagan entre vecinos a través de la red espacial.
"""
    
    # Guardar en session_state
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
    st.session_state.auto_insight = auto_insight
    st.session_state.years = sorted(panel["year"].unique().astype(int).tolist())
    st.session_state.ready = True
    
    progress.progress(100, text="¡Listo!")
    return True

# -----------------------------------------------------------------------------
# VISUALIZACIONES
# -----------------------------------------------------------------------------
def plot_choropleth(year, variable):
    panel = st.session_state.panel
    gdf = st.session_state.gdf_static
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

def plot_lisa(variable):
    gdf_avg = st.session_state.gdf_avg
    W = st.session_state.W_opt
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
    ax.set_title(f"LISA ESDA · {variable}\nHH rojo · LL azul · outliers HL/LH", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig

def moran_card(variable):
    gdf_avg = st.session_state.gdf_avg
    W = st.session_state.W_opt
    col_map = {
        "Producción (t)": "agriculture",
        "Área sembrada (ha)": "planted_area",
        "Rendimiento (t/ha)": "yield_tha",
    }
    col = col_map[variable]
    mi = Moran(gdf_avg[col].values, W, permutations=499)
    sig = "sí hay dependencia espacial significativa" if mi.p_sim < 0.05 else "no hay evidencia fuerte de dependencia"
    return f"""
### Diagnóstico ESDA — {variable}

- **Índice de Moran I** = `{mi.I:.4f}`
- **p-valor simulado** = `{mi.p_sim:.4f}`
- **Interpretación Automática:** {sig}.

_Si I > 0 y significativo, los departamentos vecinos se parecen: las políticas agropecuarias deben coordinarse regionalmente._
"""

def plot_3d_static(year):
    panel = st.session_state.panel
    gdf = st.session_state.gdf_static
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
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0))
    return fig

def plot_3d_anim():
    panel = st.session_state.panel
    gdf = st.session_state.gdf_static
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
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    return fig

def spvar_story():
    r1, r2 = st.session_state.res1, st.session_state.res2
    p1, p2 = r1["params"], r2["params"]
    rho1 = r1["rho"] if r1["rho"] is not None else 0.0
    rho2 = r2["rho"] if r2["rho"] is not None else 0.0
    
    return f"""
### Resultados SpVAR & Interpretación Automática

**Especificaciones elegidas**
- Producción → `{st.session_state.spec1}`
- Área sembrada → `{st.session_state.spec2}`
- Matriz W → `{st.session_state.W_name}`

#### Ecuación de producción
| Mecanismo | Coeficiente |
|-----------|-------------|
| Autodinámica (propia) | `{p1[1]:.4f}` |
| Cruce local (área) | `{p1[2]:.4f}` |
| Espacial directa W·producción | `{p1[3]:.4f}` |
| Espacial cruzada W·área | `{p1[4]:.4f}` |
| ρ contemporáneo | `{rho1:.4f}` |

#### Ecuación de área sembrada
| Mecanismo | Coeficiente |
|-----------|-------------|
| Autodinámica (propia) | `{p2[1]:.4f}` |
| Cruce local (producción) | `{p2[2]:.4f}` |
| Espacial directa W·área | `{p2[3]:.4f}` |
| Espacial cruzada W·producción | `{p2[4]:.4f}` |
| ρ contemporáneo | `{rho2:.4f}` |

**Interpretación clave:**  
Los choques en el agro no ocurren de forma aislada; la red de vecinos (`W`) transmite dinámicas de área y producción que deben considerarse en la planificación territorial.
"""

# -----------------------------------------------------------------------------
# INTERFAZ
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:1.2rem; background:linear-gradient(135deg,#0f766e,#134e4a); color:white; border-radius:12px; margin-bottom:1.5rem;">
    <h1 style="margin-bottom:0.3rem;">🌾 AgroEspacio Colombia</h1>
    <p style="opacity:0.92; font-size:1.05rem;">De datos EVA abiertos a inteligencia territorial espacial — SpVAR · ESDA · 3D</p>
    <p style="font-size:0.9rem; margin-top:0.5rem;">Demo inmersiva · datos observados · interpretabilidad automática</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("🚀 Cargar datos EVA + calibrar modelo", type="primary", use_container_width=True):
        with st.spinner("Ejecutando pipeline completo (puede tardar 2-5 minutos)..."):
            success = run_pipeline()
            if success:
                st.success("¡Modelo calibrado correctamente!")
                st.rerun()

with col2:
    if st.session_state.ready:
        st.success(f"✅ Listo · {len(st.session_state.depts)} depts · W={st.session_state.W_name}")
    else:
        st.info("Presiona **Cargar** para iniciar")

if not st.session_state.ready:
    st.markdown(st.session_state.get("auto_insight", "Esperando carga de datos..."))
    st.stop()

# Sidebar
st.sidebar.header("Controles")
year = st.sidebar.selectbox("Año", st.session_state.years, index=len(st.session_state.years)-1)
variable = st.sidebar.selectbox(
    "Variable",
    ["Producción (t)", "Área sembrada (ha)", "Rendimiento (t/ha)"],
    index=0
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "① Explorar territorio",
    "② Diagnóstico espacial (ESDA)",
    "③ Inmersión 3D",
    "④ Decisión SpVAR"
])

with tab1:
    st.markdown("### Mapa coroplético")
    fig = plot_choropleth(year, variable)
    st.pyplot(fig)

with tab2:
    st.markdown("### ESDA e Interpretación Automática")
    st.markdown(moran_card(variable))
    fig_lisa = plot_lisa(variable)
    st.pyplot(fig_lisa)

with tab3:
    st.markdown("### Grafología satelital-espacial 3D")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("3D del año seleccionado"):
            fig3d = plot_3d_static(year)
            st.plotly_chart(fig3d, use_container_width=True)
    with col_b:
        if st.button("Animación temporal"):
            fig_anim = plot_3d_anim()
            st.plotly_chart(fig_anim, use_container_width=True)

with tab4:
    st.markdown("### Mecanismos de transmisión y Narrativa Automática")
    st.markdown(spvar_story())
    st.markdown(st.session_state.auto_insight)

st.markdown("---")
st.caption("**AgroEspacio Colombia** · Transforma datos agropecuarios abiertos en evidencia espacial accionable.")
