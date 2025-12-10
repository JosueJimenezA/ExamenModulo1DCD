import pandas as pd
import numpy as np
from .utils import report_df_status, is_continuous

def generate_analytical_dataset(data: dict, logger) -> pd.DataFrame:
    """
    Crea una tabla maestra uniendo Ratings + Users + Restaurants.
    NOTA: Se conserva la columna 'rating' original para KPIs del Dashboard.
    """
    logger.info("Iniciando ingeniería de variables y construcción de dataset...")


    
    if 'restaurants' in data:
        df_features = data['restaurants'][['placeID']].drop_duplicates().copy()
    else:
        df_features = pd.DataFrame(data['ratings']['placeID'].unique(), columns=['placeID'])

    # --- Variables Sintéticas ---
    if 'cuisine' in data:
        feat_cuisine = data['cuisine'].groupby('placeID')['Rcuisine'].count().rename('num_tipos_cocina')
        df_features = df_features.merge(feat_cuisine, on='placeID', how='left')
        logger.info("   -> Variable creada: num_tipos_cocina.")

    if 'parking' in data:
        park_mapping = {
            'none': 0, 'street': 1, 'public': 1, 'fee': 1, 'yes': 1, 
            'validated parking': 2, 'valet parking': 2
        }
        df_park = data['parking'].copy()
        df_park['score_temp'] = df_park['parking_lot'].str.lower().str.strip().map(park_mapping)
        feat_park = df_park.groupby('placeID')['score_temp'].max().rename('nivel_estacionamiento')
        df_features = df_features.merge(feat_park, on='placeID', how='left')
        logger.info("   -> Variable creada: nivel_estacionamiento.")

    if 'hours' in data:
        feat_hours = data['hours'].groupby('placeID')['days'].nunique().rename('dias_abiertos_semana')
        df_features = df_features.merge(feat_hours, on='placeID', how='left')
        logger.info("   -> Variable creada: dias_abiertos_semana.")

    if 'ratings' in data:
        feat_hist_rating = data['ratings'].groupby('placeID')['rating'].mean().rename('rating_promedio_historico')
        df_features = df_features.merge(feat_hist_rating, on='placeID', how='left')
        logger.info("   -> Variable creada: rating_promedio_historico.")

    # MERGE
    
    df_main = data['ratings'].copy()
    df_main = df_main.merge(df_features, on='placeID', how='left')
    logger.info("   -> Rating unido con variables de ingenieria.")
    
    if 'users' in data:
        cols_user = data['users'].columns.difference(df_main.columns).tolist()
        cols_user.append('userID')
        df_users_clean = data['users'][cols_user].copy()
        df_users_clean.rename(columns={'latitude': 'user_latitude', 'longitude': 'user_longitude'}, inplace=True)
        df_main = df_main.merge(df_users_clean, on='userID', how='left')
        logger.info("   -> Unión realizada con datos de usuarios por userID.")

    if 'restaurants' in data:
        cols_rest = data['restaurants'].columns.difference(df_main.columns).tolist()
        cols_rest.append('placeID')
        df_rest_clean = data['restaurants'][cols_rest].copy()
        df_rest_clean.rename(columns={'latitude': 'rest_latitude', 'longitude': 'rest_longitude', 
                                      'name': 'rest_name', 'city': 'rest_city', 'state': 'rest_state'}, inplace=True)
        df_main = df_main.merge(df_rest_clean, on='placeID', how='left')
        logger.info("   -> Unión realizada con datos de restaurantes por placeID.")

    # VARIABLES POST-MERGE Y TARGET

    if 'birth_year' in df_main.columns:
        df_main['user_age'] = 2025 - df_main['birth_year']
        df_main.loc[(df_main['user_age'] > 100) | (df_main['user_age'] < 10), 'user_age'] = np.nan
        df_main.drop('birth_year', axis=1, inplace=True)
        logger.info("   -> Variable creada: user_age.")

    # Target
    df_main['target_exitoso'] = (df_main['rating'] >= 2).astype(int)
    logger.info("   -> Variable objetivo creada: target_exitoso (rating >= 2.0).")


    cols_drop = ['food_rating', 'service_rating', 'the_geom_meter'] 
    
    df_main.drop(columns=[c for c in cols_drop if c in df_main.columns], inplace=True, errors='ignore')
    logger.info("   -> Columnas redundantes eliminadas (Se conserva 'rating' para KPI).")
    

    ids_cols = ['userID', 'placeID']
    for col in ids_cols:
        if col in df_main.columns:
            df_main[col] = df_main[col].astype(str)

    cat_cols = df_main.select_dtypes(include='object').columns
    for col in cat_cols:
        if col not in ids_cols: 
            df_main[col] = df_main[col].astype('category')

    print("\n" + "="*50)
    print("✅ DATASET MAESTRO CREADO (rating preservado)")
    print(f"Filas: {df_main.shape[0]} | Columnas: {df_main.shape[1]}")
    print("="*50 + "\n")
    
    report_df_status(df_main, "Dataset Integrado Final", logger)
    
    return df_main