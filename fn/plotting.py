import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt 


COLORS = {
    'primary': '#2E86C1',    # Azul 
    'secondary': '#E67E22',  # Naranja 
    'success': '#27AE60',    # Verde 
    'background': '#FFFFFF', # Blanco 
    'text': '#2C3E50',       # Gris 
    'grid': '#ECF0F1'        # Gris muy claro
}

def _apply_layout(fig, title, subtitle=""):
    """Aplica un tema personalizado a cualquier figura Plotly."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:gray'>{subtitle}</span>",
            font=dict(family="Arial", size=20, color=COLORS['text'])
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(family="Arial", color=COLORS['text']),
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode="closest"
    )
    return fig

# --- GRÁFICOS DE ANÁLISIS ---

def plot_missing_values(df: pd.DataFrame):
    """Gráfico de barras de valores nulos por columna."""
    nulls = df.isnull().sum().reset_index()
    nulls.columns = ['Variable', 'Nulos']
    nulls = nulls[nulls['Nulos'] > 0].sort_values('Nulos', ascending=True)

    if nulls.empty:
        return None

    fig = px.bar(nulls, x='Nulos', y='Variable', orientation='h',
                 title="Valores Ausentes", text='Nulos')
    fig.update_traces(marker_color=COLORS['secondary'], textposition='outside')
    return _apply_layout(fig, "Diagnóstico de Calidad de Datos", "Conteo de valores nulos por variable")

def plot_correlation_heatmap(df: pd.DataFrame, title="Mapa de Calor de Correlación"):
    """Heatmap interactivo de correlaciones."""
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr().round(2)

    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    return _apply_layout(fig, title, "Relación entre variables numéricas")

def plot_pca_scatter(df_pca: pd.DataFrame, target_col: pd.Series, explained_var: list):
    """Scatter plot del PCA 2D."""
    plot_data = df_pca.copy()
    plot_data['Target'] = target_col.astype(str) 

    fig = px.scatter(plot_data, x='PC1', y='PC2', color='Target',
                     color_discrete_map={'0': COLORS['text'], '1': COLORS['success']},
                     opacity=0.7, hover_data=plot_data.columns)
    
    subtitle = f"Varianza Explicada: PC1 ({explained_var[0]*100:.1f}%) + PC2 ({explained_var[1]*100:.1f}%)"
    return _apply_layout(fig, "Reducción de Dimensiones (PCA)", subtitle)

def plot_pca_loadings(loadings_df: pd.DataFrame, n_top=10):
    """
    Muestra qué variables originales influyen más en cada Componente Principal.
    loadings_df: DataFrame generado en analysis.py
    """
    top_pc1 = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index).head(n_top)
    
    fig = px.bar(top_pc1, x='PC1', y='Feature', orientation='h',
                 title=f"Top {n_top} Variables influyentes en PC1")
    fig.update_traces(marker_color=COLORS['primary'])
    _apply_layout(fig, "Interpretación de PCA (Loadings)", "Contribución de variables originales al Componente 1")
    
    return fig

def plot_geo_map(df: pd.DataFrame, lat_col='latitude', lon_col='longitude', target_col=None):
    """Mapa interactivo de restaurantes."""
    if lat_col not in df.columns or lon_col not in df.columns:
        return None
        
    fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, 
                            color=target_col,
                            color_continuous_scale=px.colors.sequential.Bluered,
                            size_max=15, zoom=10,
                            mapbox_style="carto-positron")
    
    return _apply_layout(fig, "Distribución Geográfica", "Ubicación de los restaurantes/usuarios")



def plot_raw_eda_summary(df: pd.DataFrame, title="EDA Summary", n_cols=3):
    """
    Genera un grid de gráficos (Histogramas para Numéricas, Barras para Categóricas).
    """
    # Identificar tipos
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    cols_to_plot = list(num_cols[:6]) + list(cat_cols[:3])
    
    if not cols_to_plot:
        return None

    from plotly.subplots import make_subplots
    rows = (len(cols_to_plot) // n_cols) + 1
    
    fig = make_subplots(rows=rows, cols=n_cols, subplot_titles=cols_to_plot)

    for i, col in enumerate(cols_to_plot):
        row = (i // n_cols) + 1
        c = (i % n_cols) + 1
        
        if col in num_cols:
            trace = go.Histogram(x=df[col], name=col, marker_color=COLORS['primary'], opacity=0.7)
            fig.add_trace(trace, row=row, col=c)
        else:
            val_counts = df[col].value_counts().head(10)
            trace = go.Bar(x=val_counts.index, y=val_counts.values, name=col, marker_color=COLORS['secondary'])
            fig.add_trace(trace, row=row, col=c)

    fig.update_layout(height=300*rows, showlegend=False, title_text=f"<b>{title}</b><br><sup>Distribución de variables principales</sup>")
    return _apply_layout(fig, title)


def plot_varclus_rs_ratio(df_report: pd.DataFrame):
    """
    Grafica el RS_Ratio de las variables ganadoras vs perdedoras.
    Recibe el reporte generado por variable_clustering_removal.
    """

    import re
    
    df_plot = df_report.copy()
    df_plot['Ratio'] = df_plot['Ganadora (RS_Ratio)'].apply(lambda x: float(re.search(r'\((.*?)\)', x).group(1)))
    df_plot['Var_Name'] = df_plot['Ganadora (RS_Ratio)'].apply(lambda x: x.split(' (')[0])
    
    fig = px.bar(df_plot, x='Cluster_ID', y='Ratio', text='Var_Name',
                 title="Calidad de Representantes por Clúster (VarClusHi)",
                 labels={'Ratio': 'RS Ratio (Menor es mejor)', 'Cluster_ID': 'Clúster'})
    
    fig.update_traces(textposition='outside', marker_color=COLORS['primary'])
    
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Alta Representatividad")
    
    return _apply_layout(fig, "VarClus: Selección de Variables", "Variables con menor RS Ratio explican mejor su grupo")

def plot_iv_ranking(iv_dict: dict):
    """Gráfico de barras horizontal ordenado por IV."""
    df_iv = pd.DataFrame(list(iv_dict.items()), columns=['Variable', 'IV'])
    df_iv = df_iv.sort_values('IV', ascending=True)
    
    def get_color(val):
        if val < 0.02: return 'gray'
        if val < 0.1: return COLORS['secondary']
        if val < 0.3: return COLORS['primary']
        if val < 0.5: return COLORS['success'] 
        return '#C0392B' 

    colors = [get_color(x) for x in df_iv['IV']]

    fig = px.bar(df_iv, x='IV', y='Variable', orientation='h', text='IV')
    fig.update_traces(marker_color=colors, texttemplate='%{text:.3f}', textposition='outside')
    
    # Líneas de referencia
    fig.add_vline(x=0.02, line_dash="dot", annotation_text="Mínimo útil")
    fig.add_vline(x=0.5, line_dash="dot", line_color="red", annotation_text="Sospechoso")
    
    return _apply_layout(fig, "Information Value (IV) Ranking", "Poder predictivo de cada variable (Top es mejor)")