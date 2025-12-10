import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys


st.set_page_config(
    page_title="Analytics Gastronómico",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import fn.plotting as plot_lib
    import fn.analysis as analysis_lib
    from fn.utils import is_continuous
except ImportError as e:
    st.error(f"Error importando módulos locales: {e}. Asegúrate de ejecutar 'streamlit run app.py' desde la raíz del proyecto.")
    st.stop()


st.markdown("""
<style>
    /* Fondo general de la app */
    .main { background-color: #0E1117; }
    h1 { color: #FAFAFA; font-family: 'Helvetica', sans-serif; }
    h2, h3 { color: #3498DB; }
    
    /* === ESTILO DE LAS TARJETAS KPI (METRIC) === */
    div[data-testid="stMetric"] {
        /* Cambio clave: Fondo GRIS OSCURO/AZULADO para contraste */
        background-color: #262730 !important; 
        border: 1px solid #41444d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #b0b3b8 !important; /* Gris claro para el título */
        font-size: 14px;
    }

    .css-1d391kg { padding-top: 1rem; } 
</style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    path = os.path.join('resultados', 'data_dashboard.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

df_raw = load_data()

if df_raw is None:
    st.error("⚠️ No se encontró el archivo de datos ('resultados/data_dashboard.csv'). Ejecuta primero el notebook 'main.ipynb'.")
    st.stop()

# --- SIDEBAR: FILTROS GLOBALES ---
st.sidebar.header("🎛️ Filtros de Análisis")
st.sidebar.markdown("---")

# Inicializamos el dataframe filtrado
df_filtered = df_raw.copy()


if 'rest_city' in df_filtered.columns:
    # Obtenemos lista única de ciudades ordenadas
    ciudades_disponibles = sorted(df_filtered['rest_city'].unique())
    
    st.sidebar.subheader("📍 Ciudad / Zona")
    

    with st.sidebar.expander("Seleccionar Ciudades", expanded=False):

        all_cities = st.checkbox("Seleccionar Todas", value=True, key="select_all_cities")
        
        selected_cities = []
        
        if all_cities:
            selected_cities = ciudades_disponibles
            st.info(f"Todas las ciudades seleccionadas ({len(ciudades_disponibles)})")
        else:
           
            for ciudad in ciudades_disponibles:
                
                if st.checkbox(ciudad, value=False, key=f"city_{ciudad}"):
                    selected_cities.append(ciudad)

  
    if selected_cities:
        df_filtered = df_filtered[df_filtered['rest_city'].isin(selected_cities)]
    else:
        st.sidebar.warning("⚠️ Selecciona al menos una ciudad.")

        df_filtered = df_filtered[df_filtered['rest_city'].isin([])] 


if 'price' in df_filtered.columns:
    st.sidebar.markdown("---")
    st.sidebar.subheader("💲 Nivel de Precio")
    
    precios_disponibles = sorted(df_filtered['price'].unique())
    selected_prices = []
    

    
    col_p1, col_p2 = st.sidebar.columns(2) 
    
    for i, precio in enumerate(precios_disponibles):

        if st.sidebar.checkbox(f"{precio.title()}", value=True, key=f"price_{precio}"):
            selected_prices.append(precio)
            
    if selected_prices:
        df_filtered = df_filtered[df_filtered['price'].isin(selected_prices)]
    else:
        st.sidebar.warning("⚠️ Selecciona un nivel de precio.")
        df_filtered = df_filtered[df_filtered['price'].isin([])]


if 'alcohol' in df_filtered.columns:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🍷 Servicio de Alcohol")
    
    opciones_alcohol = ["Todos"] + sorted(df_filtered['alcohol'].unique().tolist())
    opciones_format = [op.replace("_", " ").title() for op in opciones_alcohol]
    
    idx_alcohol = st.sidebar.radio(
        "Seleccione tipo:",
        options=range(len(opciones_alcohol)), 
        format_func=lambda x: opciones_format[x] 
    )
    
    alcohol_opt = opciones_alcohol[idx_alcohol]
    
    if alcohol_opt != "Todos":
        df_filtered = df_filtered[df_filtered['alcohol'] == alcohol_opt]

st.sidebar.markdown("---")
st.sidebar.metric("Muestra actual", f"{len(df_filtered)} registros")

# --- CABECERA PRINCIPAL ---
st.title("🍽️ Tablero de Inteligencia de Restaurantes")
st.markdown("Plataforma interactiva para el análisis de factores de éxito en el sector gastronómico.")

# --- TABS (PESTAÑAS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen Ejecutivo", 
    "🔍 Exploración de Datos", 
    "🧠 Análisis Avanzado (PCA)",
    "📂 Datos"
])


with tab1:
    st.markdown("### Indicadores Clave de Desempeño (KPIs)")
    
    # KPIs en columnas
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_rest = df_filtered['placeID'].nunique() if 'placeID' in df_filtered.columns else len(df_filtered)
    success_rate = df_filtered['target_exitoso'].mean() * 100
    avg_rating = df_filtered['rating'].mean() if 'rating' in df_filtered.columns else 0
    total_users = df_filtered['userID'].nunique() if 'userID' in df_filtered.columns else 0

    kpi1.metric("Total Restaurantes", f"{total_rest}", delta_color="off")
    kpi2.metric("Tasa de Éxito", f"{success_rate:.1f}%", delta=f"{success_rate - 50:.1f}% vs Base")
    kpi3.metric("Rating Promedio", f"{avg_rating:.2f} / 2.0", delta_color="normal")
    kpi4.metric("Usuarios Únicos", f"{total_users}")

    st.markdown("---")
    
    col_map, col_bar = st.columns([2, 1])
    
    with col_map:
        st.subheader("📍 Mapa de Calor Geográfico")
        if 'rest_latitude' in df_filtered.columns and 'rest_longitude' in df_filtered.columns:
            fig_map = plot_lib.plot_geo_map(
                df_filtered, 
                lat_col='rest_latitude', 
                lon_col='rest_longitude', 
                target_col='target_exitoso'
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No se encontraron coordenadas (latitude/longitude) para generar el mapa.")

    with col_bar:
        st.subheader("🏆 Top Factores de Éxito")
        if 'price' in df_filtered.columns:
            fig_price = px.histogram(
                df_filtered, x='price', color='target_exitoso', 
                barmode='group', title="Éxito por Rango de Precios",
                color_discrete_sequence=[plot_lib.COLORS['text'], plot_lib.COLORS['success']]
            )
            fig_price.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_price, use_container_width=True)


with tab2:
    st.markdown("### 🔬 Análisis Univariado y Bivariado")
    
    col_ctrl, col_chart = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("**Configuración del Gráfico**")
        
        num_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df_filtered.select_dtypes(exclude=['number']).columns.tolist()
        
        tipo_grafico = st.selectbox("Tipo de Gráfico", ["Histograma", "Boxplot", "Dispersión"])
        
        x_axis = st.selectbox("Eje X", df_filtered.columns, index=0)
        
        y_axis = None
        if tipo_grafico in ["Boxplot", "Dispersión"]:
            y_axis = st.selectbox("Eje Y", num_cols, index=min(1, len(num_cols)-1))
            
        color_dim = st.selectbox("Color (Agrupación)", ["Ninguno"] + list(df_filtered.columns), index=list(df_filtered.columns).index('target_exitoso') if 'target_exitoso' in df_filtered.columns else 0)

    with col_chart:
        color_arg = None if color_dim == "Ninguno" else color_dim
        
        if tipo_grafico == "Histograma":
            fig = px.histogram(df_filtered, x=x_axis, color=color_arg, barmode="overlay", title=f"Distribución de {x_axis}")
        
        elif tipo_grafico == "Boxplot":
            fig = px.box(df_filtered, x=x_axis, y=y_axis, color=color_arg, title=f"Distribución de {y_axis} por {x_axis}")
        
        elif tipo_grafico == "Dispersión":
            fig = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_arg, opacity=0.6, title=f"Relación {x_axis} vs {y_axis}")
        
        
        fig = plot_lib._apply_layout(fig, fig.layout.title.text if fig.layout.title.text else "")
        st.plotly_chart(fig, use_container_width=True)

    # Matriz de Correlación
    st.markdown("---")
    st.subheader("🔥 Matriz de Correlación (Numérica)")
    with st.expander("Ver Mapa de Calor de Correlaciones"):
        fig_corr = plot_lib.plot_correlation_heatmap(df_filtered)
        st.plotly_chart(fig_corr, use_container_width=True)


with tab3:
    st.markdown("### 🧠 Reducción de Dimensiones (PCA en Tiempo Real)")
    st.info("Este módulo recalcula el Análisis de Componentes Principales basándose en los filtros actuales de la barra lateral.")
    
    if st.button("🔄 Calcular PCA con datos actuales"):
        with st.spinner("Ejecutando algoritmos de reducción dimensional..."):

            df_pca_input = df_filtered.select_dtypes(include=['number']).copy()
            
            df_pca_input = df_pca_input.fillna(df_pca_input.median())
            
            df_pca_res, var_ratio, loadings = analysis_lib.run_pca_analysis(df_pca_input, n_components=2)
            
            if not df_pca_res.empty:
                col_pca_viz, col_pca_load = st.columns([2, 1])
                
                with col_pca_viz:
                    target_col = df_filtered['target_exitoso'] if 'target_exitoso' in df_filtered.columns else None
                    if target_col is not None:
                        target_col = target_col.reset_index(drop=True)
                    
                    fig_pca = plot_lib.plot_pca_scatter(df_pca_res, target_col, var_ratio)
                    st.plotly_chart(fig_pca, use_container_width=True)
                
                with col_pca_load:
                    st.markdown("#### Drivers del Componente 1")
                    st.dataframe(loadings[['Feature', 'PC1']].sort_values(by='PC1', key=abs, ascending=False).head(10), hide_index=True)
                    
                    st.markdown("#### Drivers del Componente 2")
                    st.dataframe(loadings[['Feature', 'PC2']].sort_values(by='PC2', key=abs, ascending=False).head(10), hide_index=True)
            
            else:
                st.warning("No hay suficientes datos o variables numéricas para ejecutar PCA con la selección actual.")
    else:
        st.write("Presiona el botón para generar el análisis.")


with tab4:
    st.markdown("### 💾 Acceso a los Datos Procesados")
    st.dataframe(df_filtered)
    
    # Botón de descarga
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados (CSV)",
        data=csv,
        file_name='restaurantes_filtrados.csv',
        mime='text/csv',
    )