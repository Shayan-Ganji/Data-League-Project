import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EllipticEnvelope # <-- ایمپورت جدید

class SmartOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, target_cols=None, threshold=3.0, iqr_multiplier=1.5):
        self.target_cols = target_cols
        self.threshold = threshold
        self.iqr_multiplier = iqr_multiplier
        self.bounds_ = {}

    def _is_normal(self, data):
        if len(data) < 8: return False, data.skew()
        stat, p_value = stats.shapiro(data)
        skewness = data.skew()
        if p_value > 0.05 and abs(skewness) < 0.8:
            return True, skewness
        return False, skewness

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.target_cols is None:
            self.target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in self.target_cols:
            if col not in df.columns: continue
            data = df[col].dropna()
            is_normal_dist, skew_val = self._is_normal(data)
            
            if is_normal_dist:
                mean, std = data.mean(), data.std()
                lower = mean - (self.threshold * std)
                upper = mean + (self.threshold * std)
            else:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - (self.iqr_multiplier * IQR)
                upper = Q3 + (self.iqr_multiplier * IQR)
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, (lower, upper) in self.bounds_.items():
            if col in df.columns:
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])
        return df

# --- کلاس جدید برای حذف پرت‌های چند متغیره ---
class ContextualOutlierCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        if 'G1' in df.columns and 'G3' in df.columns:
            print("🕵️‍♂️ Running Contextual Outlier Detection (G1 vs G3)...")
            features = df[['G1', 'G3']]
            is_outlier = self.envelope.fit_predict(features)
            
            initial_count = len(df)
            df_clean = df[is_outlier == 1]
            removed_count = initial_count - len(df_clean)
            
            print(f"   ✂️ Removed {removed_count} contextual outliers (e.g., Zero-Graders).")
            return df_clean
        else:
            print("⚠️ Skipping contextual outlier removal (Target column 'G3' missing).")
            return df

def clean_data(df):
    print("🧹 Starting data cleaning...")
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    cols_to_check = ['age', 'absences', 'G1', 'G2']
    valid_cols = [c for c in cols_to_check if c in df_clean.columns]
    
    outlier_handler = SmartOutlierHandler(target_cols=valid_cols)
    outlier_handler.fit(df_clean)
    df_clean = outlier_handler.transform(df_clean)
    contextual_cleaner = ContextualOutlierCleaner(contamination=0.06) # تنظیم دقیق‌تر
    df_clean = contextual_cleaner.transform(df_clean)
    
    print("✅ Data cleaning completed.")
    return df_clean