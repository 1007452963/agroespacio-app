import sys, subprocess, importlib, warnings, zipfile, io, requests, tempfile, os
from pathlib import Path

def _install(pkg, imp=None):
    imp = imp or pkg.split("==")[0].split("[")[0]
    try:
        importlib.import_module(imp)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg, imp in [
    ("libpysal", "libpysal"), ("esda", "esda"),
    ("spreg", "spreg"), ("geopandas", "geopandas"), ("statsmodels", "statsmodels"),
    ("matplotlib", "matplotlib"), ("seaborn", "seaborn"), ("folium", "folium"),
    ("streamlit_folium", "streamlit_folium"),
    ("mapclassify", "mapclassify"), ("numpy", "numpy"), ("pandas", "pandas"),
    ("scipy", "scipy"), ("splot", "splot"), ("plotly", "plotly"), ("requests", "requests"),
]:
    _install(pkg, imp)

warnings.filterwarnings("ignore")
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
from streamlit_folium import st_folium
from splot.esda import lisa_cluster, moran_scatterplot
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroEspacio Colombia",
    page_icon="🌾",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
<style>
.hero {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, #0f766e, #134e4a);
    color: white;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2rem; margin-bottom: 0.3rem; color: white; }
.hero p { opacity: 0.92; font-size: 1.05rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ESTADO GLOBAL CON STREAMLIT SESSION STATE
# -----------------------------------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False
if "gdf_static" not in st.session_state:
    st.session_state.gdf_static = None
if "panel" not in st.session_state:
    st.session_state.panel = None
if "gdf_avg" not in st.session_state:
    st.session_state.gdf_avg = None
if "W_opt" not in st.session_state:
    st.session_state.W_opt = None
if "W_name" not in st.session_state:
    st.session_state.W_name = None
if "panel_sp" not in st.session_state:
    st.session_state.panel_sp = None
if "res1" not in st.session_state:
    st.session_state.res1 = None
if "res2" not in st.session_state:
    st.session_state.res2 = None
if "spec1" not in st.session_state:
    st.session_state.spec1 = None
if "spec2" not in st.session_state:
    st.session_state.spec2 = None
if "depts" not in st.session_state:
    st.session_state.depts = None

# -----------------------------------------------------------------------------
# CARGA DE DATOS (Pipeline EVA)
# -----------------------------------------------------------------------------
def load_pipeline():
    SHP_DIR = Path("/tmp/gadm_col")
    SHP_DIR.mkdir(exist_ok=True, parents=True)
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_COL_shp.zip"
    if not list(SHP_DIR.glob("*_1.shp")):
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(SHP_DIR)
    
    shp_files = list(SHP_DIR.glob("*_1.shp"))
    if not shp_files:
        raise FileNotFoundError("No se encontró el archivo shapefile de GADM.")
    
    gdf = gpd.read_file(shp_files[0])
    gdf = gdf.rename(columns={"NAME_1": "departamento", "GID_1": "dept_id"})
    gdf["departamento"] = (
        gdf["departamento"].str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    gdf = gdf[["dept_id", "departamento", "geometry"]].drop_duplicates("departamento")

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
    W_name = max(scores, key=scores.get) if scores else "Queen"
    W_opt = candidates.get(W_name, Queen.from_dataframe(gdf, silence_warnings=True))
    W_opt.transform = "r"

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
    st.session_state.ready = True

# -----------------------------------------------------------------------------
# INTERFAZ STREAMLIT
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>AgroEspacio Colombia</h1>
    <p>De datos EVA abiertos a inteligencia territorial espacial — SpVAR · ESDA · 3D</p>
    <p style="font-size:0.9rem;margin-top:0.5rem">Demo inmersiva · datos observados · lista para pitch, tesis y decisiones</p>
</div>
""", unsafe_allow_html=True)

col_load, col_stat = st.columns([2, 3])
with col_load:
    load_btn = st.button("🚀 Cargar datos EVA + calibrar modelo", type="primary", use_container_width=True)

with col_stat:
    if load_btn:
        with st.spinner("Descargando y procesando datos EVA y GADM (2–5 min la primera vez)..."):
            try:
                load_pipeline()
                st.success("¡Datos cargados y modelos calibrados con éxito!")
            except Exception as e:
                st.error(f"Error al cargar el pipeline: {e}")
    
    if not st.session_state.ready:
        st.info("Presiona **Cargar** para iniciar.")
    else:
        st.success("Sistema listo y en memoria.")

if st.session_state.ready:
    years_list = sorted(st.session_state.panel["year"].unique().astype(int).tolist())
    default_year = years_list[-1]
else:
    years_list = [2023]
    default_year = 2023

col_y, col_v = st.columns(2)
with col_y:
    year_selected = st.selectbox("Año", choices=years_list, index=len(years_list)-1, disabled=not st.session_state.ready)
with col_v:
    var_selected = st.selectbox(
        "Variable",
        choices=["Producción (t)", "Área sembrada (ha)", "Rendimiento (t/ha)"],
        value="Producción (t)",
        disabled=not st.session_state.ready
    )

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① Explorar territorio",
    "② Diagnóstico espacial",
    "③ Inmersión 3D",
    "④ Decisión SpVAR",
    "⑤ Compartir"
])

with tab1:
    st.markdown("### Mapa coroplético de alto espectro")
    st.markdown("Elige año y variable. Ideal para captar atención en los primeros 10 segundos.")
    
    def build_folium_map(year, variable):
        if not st.session_state.ready:
            return None
        panel, gdf = st.session_state.panel, st.session_state.gdf_static
        year = int(year)
        col_map = {
            "Producción (t)": ("agriculture", "YlGn"),
            "Área sembrada (ha)": ("planted_area", "YlOrBr"),
            "Rendimiento (t/ha)": ("yield_tha", "YlOrRd"),
        }
        col, fill = col_map[variable]
        g = gdf.merge(panel[panel["year"] == year], on="departamento", how="left")
        g[col] = g[col].fillna(0)
        g["lon"] = g.geometry.centroid.x
        g["lat"] = g.geometry.centroid.y
        m = folium.Map(location=[g["lat"].mean(), g["lon"].mean()], zoom_start=5, tiles="CartoDB positron")
        folium.Choropleth(
            geo_data=g.__geo_interface__, data=g,
            columns=["departamento", col],
            key_on="feature.properties.departamento",
            fill_color=fill, fill_opacity=0.75, line_opacity=0.3,
            legend_name=f"{variable} · {year}",
        ).add_to(m)
        return m

    if st.button("Actualizar mapa") or st.session_state.ready:
        if st.session_state.ready:
            m = build_folium_map(year_selected, var_selected)
            if m:
                st_folium(m, width=1100, height=550)

with tab2:
    st.markdown("### ESDA en lenguaje de negocio")
    st.markdown("¿Los vecinos se parecen? ¿Dónde hay clusters?")
    
    if st.button("Calcular Moran + LISA"):
        if not st.session_state.ready:
            st.warning("Carga los datos primero.")
        else:
            gdf_avg, W = st.session_state.gdf_avg, st.session_state.W_opt
            col_map = {
                "Producción (t)": "agriculture",
                "Área sembrada (ha)": "planted_area",
                "Rendimiento (t/ha)": "yield_tha",
            }
            col = col_map[var_selected]
            mi = Moran(gdf_avg[col].values, W, permutations=499)
            sig = "sí hay dependencia espacial significativa" if mi.p_sim < 0.05 else "no hay evidencia fuerte de dependencia"
            
            st.markdown(f"""
            ### Diagnóstico espacial — {var_selected}
            - **Índice de Moran I** = `{mi.I:.4f}`
            - **p-valor simulado** = `{mi.p_sim:.4f}`
            - **Interpretación:** {sig}.
            """)
            
            lisa = Moran_Local(gdf_avg[col].values, W, permutations=499)
            fig, ax = plt.subplots(figsize=(7, 7))
            lisa_cluster(lisa, gdf_avg, p=0.05, ax=ax, legend=True)
            ax.set_title(f"LISA · {var_selected}\nHH rojo · LL azul · outliers HL/LH", fontsize=11)
            ax.axis("off")
            st.pyplot(fig)

with tab3:
    st.markdown("### Grafología satelital-espacial 3D")
    st.markdown("Volumen = producción · color = rendimiento · tamaño = área.")
    
    col_3d1, col_3d2 = st.columns(2)
    with col_3d1:
        btn3d = st.button("3D del año seleccionado")
    with col_3d2:
        btn_anim = st.button("Animación temporal")
        
    if btn3d:
        if not st.session_state.ready:
            st.warning("Carga los datos primero.")
        else:
            panel, gdf = st.session_state.panel, st.session_state.gdf_static
            year = int(year_selected)
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
            st.plotly_chart(fig, use_container_width=True)
            
    if btn_anim:
        if not st.session_state.ready:
            st.warning("Carga los datos primero.")
        else:
            panel, gdf = st.session_state.panel, st.session_state.gdf_static
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
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Mecanismos de transmisión")
    st.markdown("Historia lista para presentar a gobierno, gremios o banca de desarrollo.")
    
    if st.button("Generar narrativa de resultados"):
        if not st.session_state.ready:
            st.warning("Carga los datos primero.")
        else:
            r1, r2 = st.session_state.res1, st.session_state.res2
            p1, p2 = r1["params"], r2["params"]
            rho1 = r1["rho"] if r1["rho"] is not None else 0.0
            rho2 = r2["rho"] if r2["rho"] is not None else 0.0
            
            st.markdown(f"""
### Resultados SpVAR — narrativa para decisión

**Especificaciones elegidas**
- Producción → `{st.session_state.spec1}`
- Área sembrada → `{st.session_state.spec2}`
- Matriz W → `{st.session_state.W_name}`

#### Ecuación de producción
- Autodinámica (propia): `{p1[1]:.4f}`
- Cruce local (área): `{p1[2]:.4f}`
- Espacial directa W·producción: `{p1[3]:.4f}`
- Espacial cruzada W·área: `{p1[4]:.4f}`
- ρ contemporáneo: `{rho1:.4f}`

#### Ecuación de área sembrada
- Autodinámica (propia): `{p2[1]:.4f}`
- Cruce local (producción): `{p2[2]:.4f}`
- Espacial directa W·área: `{p2[3]:.4f}`
- Espacial cruzada W·producción: `{p2[4]:.4f}`
- ρ contemporáneo: `{rho2:.4f}`

**Mensaje clave para stakeholders**  
Los efectos no son solo “del departamento”: lo que ocurre en los vecinos (área y producción) entra en la dinámica. Útil para focalizar programas de extensión agrícola, infraestructura rural y ordenamiento territorial.
            """)

with tab5:
    st.markdown("### Activo digital exportable")
    st.markdown("HTML del mapa para embed en web, Notion o reporte PDF.")
    
    if st.button("Generar mapa interactivo Folium"):
        if not st.session_state.ready:
            st.warning("Carga los datos primero.")
        else:
            panel, gdf = st.session_state.panel, st.session_state.gdf_static
            year = int(year_selected)
            col_map = {
                "Producción (t)": ("agriculture", "YlGn"),
                "Área sembrada (ha)": ("planted_area", "YlOrBr"),
                "Rendimiento (t/ha)": ("yield_tha", "YlOrRd"),
            }
            col, fill = col_map[var_selected]
            g = gdf.merge(panel[panel["year"] == year], on="departamento", how="left")
            g[col] = g[col].fillna(0)
            g["lon"] = g.geometry.centroid.x
            g["lat"] = g.geometry.centroid.y
            m = folium.Map(location=[g["lat"].mean(), g["lon"].mean()], zoom_start=5, tiles="CartoDB positron")
            folium.Choropleth(
                geo_data=g.__geo_interface__, data=g,
                columns=["departamento", col],
                key_on="feature.properties.departamento",
                fill_color=fill, fill_opacity=0.75, line_opacity=0.3,
                legend_name=f"{var_selected} · {year}",
            ).add_to(m)
            
            path = "/tmp/agroespacio_map.html"
            m.save(path)
            with open(path, "rb") as f:
                st.download_button(
                    label="📥 Descargar archivo HTML del mapa",
                    data=f,
                    file_name=f"agroespacio_map_{year}.html",
                    mime="text/html"
                )

st.markdown("""
---
**Posicionamiento:** *AgroEspacio* transforma datos agropecuarios abiertos en evidencia espacial accionable.  
Uso sugerido en marketing: demo en Streamlit Cloud → captura de 3D + frase de insight Moran.
""")
