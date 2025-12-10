import os
import glob
import pandas as pd
from typing import Dict
from .utils import get_dynamic_path 

def load_all_data(logger) -> Dict[str, pd.DataFrame]:
    datos_dir = get_dynamic_path('datos')
    if not os.path.exists(datos_dir):
        logger.error(f"No directorio: {datos_dir}")
        raise FileNotFoundError(f"Revisar ruta: {datos_dir}")

    temp_data = {}
    logger.info(f"Cargando desde: {datos_dir}")

    all_files = glob.glob(os.path.join(datos_dir, '*.*'))
    
    for path in all_files:
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        
        if ext.lower() not in ['.csv', '.xlsx', '.xls']:
            continue
            
        try:
            if ext.lower() == '.csv':
                try:
                    df = pd.read_csv(path)
                except:
                    df = pd.read_csv(path, encoding='latin1')
            else:
                df = pd.read_excel(path)
            
            temp_data[filename] = df
            
        except Exception as e:
            logger.error(f"Error cargando {filename}: {e}")

    final_data = {}
    processed_bases = set()
    
    unique_bases = set([os.path.splitext(k)[0] for k in temp_data.keys()])
    
    for base in unique_bases:
        variations = [k for k in temp_data.keys() if os.path.splitext(k)[0] == base]
        
        if len(variations) == 1:
            final_data[base] = temp_data[variations[0]]
            logger.info(f"Cargado: {base} (Origen: {variations[0]})")
            
        else:
            logger.warning(f"Duplicados encontrados para '{base}': {variations}")
            
            winner_key = next((v for v in variations if '.csv' in v), variations[0])
            
            final_data[base] = temp_data[winner_key]
            logger.info(f" -> Conflicto resuelto: Se usará '{winner_key}' bajo la llave '{base}'")
            
    return final_data