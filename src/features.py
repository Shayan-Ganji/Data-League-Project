import pandas as pd
import numpy as np

def build_features(df):
    print("🏗 Building features...")
    df_feat = df.copy()
    if 'G1' in df_feat.columns and 'G2' in df_feat.columns:
        df_feat['Grade_Trend'] = df_feat['G2'] - df_feat['G1']
        df_feat['Weighted_Score'] = (df_feat['G1'] * 0.3) + (df_feat['G2'] * 0.7)
    
    if all(col in df_feat.columns for col in ['goout', 'Dalc', 'Walc']):
        df_feat['Social_Distraction'] = df_feat['goout'] * (df_feat['Dalc'] + df_feat['Walc'])

    if 'failures' in df_feat.columns and 'age' in df_feat.columns:
        df_feat['Failure_Ratio'] = df_feat['failures'] / df_feat['age']
    binary_mapping = {'yes': 1, 'no': 0, 'F': 0, 'M': 1, 
                      'GP': 0, 'MS': 1, 'U': 0, 'R': 1, 
                      'LE3': 0, 'GT3': 1, 'T': 0, 'A': 1}
    
    for col in df_feat.columns:
        if df_feat[col].dtype == 'object':
            unique_vals = set(df_feat[col].unique())
            if unique_vals.issubset(set(binary_mapping.keys())):
                df_feat[col] = df_feat[col].map(binary_mapping)
    categorical_cols = df_feat.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df_feat = pd.get_dummies(df_feat, columns=categorical_cols, drop_first=True)
        
    for col in df_feat.columns:
        if df_feat[col].dtype == 'bool':
             df_feat[col] = df_feat[col].astype(int)

    print(f"✅ Feature engineering completed. New shape: {df_feat.shape}")
    return df_feat