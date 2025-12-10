import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from varclushi import VarClusHi
from .utils import is_continuous
from sklearn.feature_selection import SelectKBest, f_classif



def calculate_iv(df, feature, target):
    lst = []
    if np.issubdtype(df[feature].dtype, np.number) and df[feature].nunique() > 10:
        try:
            df[feature] = pd.qcut(df[feature], q=5, duplicates='drop')
        except:
            pass 
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        try:
            good = df[(df[feature] == val) & (df[target] == 0)].count()[feature]
            bad = df[(df[feature] == val) & (df[target] == 1)].count()[feature]
        except KeyError:
            continue
        lst.append([feature, val, good, bad])
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'Good', 'Bad'])
    data['Good'] = data['Good'].replace(0, 0.5)
    data['Bad'] = data['Bad'].replace(0, 0.5)
    total_good = data['Good'].sum()
    total_bad = data['Bad'].sum()
    if total_good == 0 or total_bad == 0: return 0
    data['Dist_Good'] = data['Good'] / total_good
    data['Dist_Bad'] = data['Bad'] / total_bad
    data['WoE'] = np.log(data['Dist_Good'] / data['Dist_Bad'])
    data['IV'] = (data['Dist_Good'] - data['Dist_Bad']) * data['WoE']
    return data['IV'].sum()



def variable_clustering_removal(df: pd.DataFrame, max_eigenvalue=1.0, logger=None):
    """
    Ejecuta el algoritmo VarClusHi real (estilo SAS).
    """
    protected_cols = ['target_exitoso', 'placeID']
    
    cols_candidatas = [c for c in df.columns if is_continuous(df[c])]
    cols_to_cluster = [c for c in cols_candidatas if c not in protected_cols]

    if len(cols_to_cluster) < 2:
        if logger: logger.warning("No hay suficientes variables para VarClus.")
        return df, pd.DataFrame()

    df_x = df[cols_to_cluster]
    
    vc = VarClusHi(df_x, maxeigval2=max_eigenvalue, maxclus=None)
    
    vc.varclus()
    
    rsquare_report = vc.rsquare
    
    winners = rsquare_report.sort_values('RS_Ratio').groupby('Cluster').first()['Variable'].tolist()
    
    selection_report = []
    for cluster_id in rsquare_report['Cluster'].unique():
        cluster_vars = rsquare_report[rsquare_report['Cluster'] == cluster_id]
        
        winner_row = cluster_vars.sort_values('RS_Ratio').iloc[0]
        winner_name = winner_row['Variable']
        losers = cluster_vars[cluster_vars['Variable'] != winner_name]['Variable'].tolist()
        
        selection_report.append({
            'Cluster_ID': cluster_id,
            'Ganadora (RS_Ratio)': f"{winner_name} ({winner_row['RS_Ratio']:.3f})",
            'Eliminadas': ", ".join(losers) if losers else "-",
            'Total_Vars': len(cluster_vars)
        })

    if logger:
        logger.info(f"VarClusHi: {len(cols_to_cluster)} variables reducidas a {len(winners)} clusters.")

    vars_ignored = [c for c in df.columns if c not in cols_to_cluster]
    final_cols = winners + vars_ignored
    final_cols = list(set(final_cols).intersection(set(df.columns)))
    
    return df[final_cols], pd.DataFrame(selection_report)

def select_best_features_iv(df: pd.DataFrame, target_col='target_exitoso', logger=None):
    """
    Selecciona variables basado en Information Value (IV).
    Excluye automáticamente variables ID y el Target.
    """
    if target_col not in df.columns:
        return [], {}

    iv_values = {}
    
    # Se excluyen identificadores, rating_promedio_historico por data leakage, y el target
    excluded_cols = [target_col, 'placeID', 'userID','rating_promedio_historico','rating']
    
    features = [c for c in df.columns if c not in excluded_cols]
    
    for feat in features:
        try:
            # Trabajamos en copia temporal
            temp_df = df[[feat, target_col]].copy()
            temp_df = temp_df.dropna()
            
            # Si es numérica continua, el calculate_iv interno hace binning automático
            # Si es categórica (como rest_city), lo calcula directo
            iv = calculate_iv(temp_df, feat, target_col)
            iv_values[feat] = iv
        except Exception:
            iv_values[feat] = 0
            
    sorted_iv = sorted(iv_values.items(), key=lambda x: x[1], reverse=True)
    
    top_5 = [x[0] for x in sorted_iv[:5]]
    
    return top_5, iv_values


def run_pca_analysis(df: pd.DataFrame, n_components=2):
    cols_pca = [c for c in df.columns if is_continuous(df[c])]
    cols_pca = [c for c in cols_pca if c not in ['target_exitoso', 'placeID']]
    if not cols_pca: return pd.DataFrame(), [], pd.DataFrame()
    
    features = df[cols_pca]
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    df_pca = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(n_components)])
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)], index=cols_pca)
    loadings.reset_index(inplace=True)
    loadings.rename(columns={'index': 'Feature'}, inplace=True)
    return df_pca, pca.explained_variance_ratio_, loadings



def run_select_k_best(df: pd.DataFrame, target_col='target_exitoso', k='all'):
    """
    Ejecuta SelectKBest usando f_classif (ANOVA) para clasificar la importancia
    de las variables continuas frente al target.
    """
    excluded = [target_col, 'placeID', 'userID', 'rating_promedio_historico','rating']
    
    cols_input = [c for c in df.select_dtypes(include=[np.number]).columns 
                  if c not in excluded]
    
    if not cols_input:
        return pd.DataFrame()

    X = df[cols_input].fillna(0) 
    y = df[target_col]
    
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    scores = pd.DataFrame({
        'Feature': cols_input,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    })
    
    return scores.sort_values('F_Score', ascending=False).reset_index(drop=True)