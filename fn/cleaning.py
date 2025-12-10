import pandas as pd
import numpy as np
from .utils import is_continuous

def normalize_text_data(df: pd.DataFrame, text_cols: list = None, logger=None) -> pd.DataFrame:
    """Homogeniza columnas de texto."""
    if text_cols is None:
        text_cols = ['rest_city', 'rest_state', 'city', 'state', 'Rcuisine']
    
    valid_cols = [c for c in text_cols if c in df.columns]
    df_clean = df.copy()
    
    correcciones = {
        's.l.p.': 'san luis potosi', 's.l.p': 'san luis potosi', 'slp': 'san luis potosi',
        'san luis potos': 'san luis potosi', 'san luis potosi ': 'san luis potosi',
        'cd. victoria': 'ciudad victoria', 'cd victoria': 'ciudad victoria',
        'victoria': 'ciudad victoria', 'cuernavaca': 'cuernavaca'
    }

    for col in valid_cols:
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
        df_clean[col] = df_clean[col].str.replace('.', '', regex=False)
        df_clean[col] = df_clean[col].replace(correcciones)
        df_clean[col] = df_clean[col].str.title()
        
    if logger:
        logger.info(f"Normalización de texto aplicada en: {valid_cols}")
        
    return df_clean

def treat_outliers_iqr(df: pd.DataFrame, method='clip', factor=1.5, logger=None) -> pd.DataFrame:
    """Detecta y trata outliers usando IQR."""
    df_out = df.copy()
    cont_cols = [c for c in df.columns if is_continuous(df[c])]
    
    if not cont_cols:
        return df_out

    impact_report = {}

    for col in cont_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        
        outliers_mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            impact_report[col] = {'count': n_outliers, 'limits': (round(lower_bound, 2), round(upper_bound, 2))}
            if method == 'clip':
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'drop':
                df_out = df_out[~outliers_mask]

    if logger:
        if impact_report:
            logger.info(f"🔎 DETALLE DE OUTLIERS ({method.upper()}):")
            for col, data in impact_report.items():
                logger.info(f"   -> {col}: {data['count']} valores ajustados. Límites: {data['limits']}")
        else:
            logger.info("   -> No se detectaron outliers fuera del rango IQR en las variables continuas.")

    return df_out

def clean_data_advanced(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Pipeline maestro de limpieza.
    Incluye regla de completitud (min 65% de datos).
    """
    df_clean = df.copy()
    
    # ELIMINICACION DE VARIABLES CONSTANTES
    cols_const = [c for c in df_clean.columns if df_clean[c].nunique() <= 1]
    if cols_const:
        df_clean.drop(columns=cols_const, inplace=True)
        if logger:
            logger.warning(f"🗑️ Eliminadas por ser Constantes (Varianza 0): {cols_const}")

    # COMPLETITUD MINIMA 65%
    threshold_missing = 0.35 # Máximo 35% de nulos permitido
    cols_empty = []
    
    for col in df_clean.columns:
        pct_missing = df_clean[col].isnull().mean()
        if pct_missing > threshold_missing:
            cols_empty.append(col)
            
    if cols_empty:
        df_clean.drop(columns=cols_empty, inplace=True)
        if logger:
            logger.warning(f"📉 Eliminadas por falta de datos (>35% Nulos): {cols_empty}")
    else:
        if logger:
            logger.info("   -> Todas las variables cumplen con el criterio de completitud (min 65%).")

    # IMPUTACION DE VALORES NULOS
    if logger: logger.info("💉 Iniciando Imputación de Valores Nulos...")
    
    imputed_count = 0
    for col in df_clean.columns:
        n_missing = df_clean[col].isnull().sum()
        
        if n_missing > 0:
            if is_continuous(df_clean[col]):
                fill_val = df_clean[col].median()
                method_name = "Mediana"
            else:
                if not df_clean[col].mode().empty:
                    fill_val = df_clean[col].mode()[0]
                else:
                    fill_val = 0
                method_name = "Moda"
            
            df_clean[col] = df_clean[col].fillna(fill_val)
            imputed_count += 1
            
            if logger:
                val_str = f"{fill_val:.2f}" if isinstance(fill_val, (int, float)) else str(fill_val)
                logger.info(f"   -> {col:<25} | Nulos: {n_missing:<4} | Método: {method_name:<7} | Valor: {val_str}")

    if imputed_count == 0 and logger:
        logger.info("   -> No se encontraron nulos para imputar.")

    # TRATAMIENTO DE OUTLIERS
    df_clean = treat_outliers_iqr(df_clean, method='clip', logger=logger)
    
    return df_clean