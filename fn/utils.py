import logging
import os
import sys
import pandas as pd
import numpy as np

def setup_logger(log_file='execution.log'):
    """Configura el logger para escribir en archivo y consola (UTF-8)."""
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Advertencia: No se pudo crear log file: {e}")
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
        
    return logger

def get_project_root():
    current_file_path = os.path.abspath(__file__)
    fn_dir = os.path.dirname(current_file_path)
    root_dir = os.path.dirname(fn_dir)
    return root_dir

def get_dynamic_path(folder_name):
    root = get_project_root()
    return os.path.join(root, folder_name)

def is_continuous(series: pd.Series, threshold: int = 10) -> bool:
    """
    Una variable es continua SI:
    1. Es numérica (int/float)
    2. Y tiene MÁS de 'threshold' valores únicos (por defecto 10)
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    return series.nunique() > threshold

def report_df_status(df: pd.DataFrame, step_name: str, logger=None):
    """
    Genera un reporte detallado SIN TRUNCAR las listas de variables numéricas.
    """
    n_rows, n_cols = df.shape
    n_nulls = df.isnull().sum().sum()
    
    cols_num_total = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    cols_cont = [c for c in cols_num_total if is_continuous(df[c])]
    cols_disc = [c for c in cols_num_total if c not in cols_cont]
    
    msg = (
        f"\n{'='*60}\n"
        f"📍 REPORTE: {step_name}\n"
        f"{'='*60}\n"
        f"📊 Dimensiones: {n_rows} filas x {n_cols} columnas\n"
        f"🧪 Nulos Totales: {n_nulls}\n"
        f"🔢 Numéricas Totales ({len(cols_num_total)}):\n"
        f"      📈 Continuas ({len(cols_cont)}) [Se limpian outliers]:\n"
        f"          {cols_cont}\n"
        f"      🔢 Discretas ({len(cols_disc)}) [Se protegen]:\n"
        f"          {cols_disc}\n"
        f"🔤 Categóricas ({len(cols_cat)}): {cols_cat[:5]} ... (Top 5)\n"
        f"{'-'*60}"
    )
    
    if logger:
        logger.info(msg)
    else:
        print(msg)

def get_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera la matriz técnica de auditoría.
    """
    report = []
    for col in df.columns:
        dtype = df[col].dtype
        n_null = df[col].isnull().sum()
        pct_null = (n_null / len(df)) * 100
        n_unique = df[col].nunique()
        pct_unique = (n_unique / len(df)) * 100
        
        # Inferencia simple de tipo para el reporte
        inferred = "Categorical"
        if is_continuous(df[col]): inferred = "Continuous"
        elif pd.api.types.is_numeric_dtype(df[col]): inferred = "Discrete (Numeric)"
        
        if pct_unique == 100: inferred = "Unique Key"
        if n_unique <= 1: inferred = "Constant"

        min_val, max_val, mean_val = "-", "-", "-"
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = round(df[col].mean(), 2)

        report.append({
            'Variable': col,
            'Inferred_Type': inferred,
            'Dtype': str(dtype),
            'Missing': n_null,
            'Missing_(%)': round(pct_null, 2),
            'Unique': n_unique,
            'Cardinality_(%)': round(pct_unique, 2),
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val
        })
        
    return pd.DataFrame(report).sort_values('Missing_(%)', ascending=False)