import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from scipy import stats


class DataAnalyzer:
    """データの自動分析を行うクラス"""
    
    def __init__(self):
        pass
    
    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """数値型のカラムを取得"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """カテゴリ型のカラムを取得"""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_basic_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本統計量を計算"""
        numeric_cols = self.get_numeric_columns(df)
        if not numeric_cols:
            return pd.DataFrame()
        
        return df[numeric_cols].describe()
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値の分析"""
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'column': missing_count.index,
            'missing_count': missing_count.values,
            'missing_percentage': missing_percentage.values
        })
        
        # 欠損値がある列のみ返す
        return missing_df[missing_df['missing_count'] > 0].reset_index(drop=True)
    
    def calculate_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """相関行列を計算"""
        numeric_cols = self.get_numeric_columns(df)
        if not numeric_cols:
            return pd.DataFrame()
        
        return df[numeric_cols].corr()
    
    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """各列の分布を分析"""
        distribution_info = {}
        
        # 数値列の分析
        numeric_cols = self.get_numeric_columns(df)
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                distribution_info[col] = {
                    'skewness': stats.skew(col_data),
                    'kurtosis': stats.kurtosis(col_data),
                    'unique_values': len(col_data.unique()),
                    'type': 'numeric'
                }
        
        # カテゴリ列の分析
        categorical_cols = self.get_categorical_columns(df)
        for col in categorical_cols:
            distribution_info[col] = {
                'unique_values': len(df[col].unique()),
                'value_counts': df[col].value_counts().to_dict(),
                'type': 'categorical'
            }
        
        return distribution_info
    
    def detect_outliers_iqr(self, df: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, List[int]]:
        """IQR法による外れ値検出"""
        outliers = {}
        numeric_cols = self.get_numeric_columns(df)
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                # 外れ値のインデックスを取得
                outlier_indices = df[
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ].index.tolist()
                
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        return outliers
    
    def detect_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3) -> Dict[str, List[int]]:
        """Zスコア法による外れ値検出"""
        outliers = {}
        numeric_cols = self.get_numeric_columns(df)
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 1:  # 標準偏差を計算するには少なくとも2つの値が必要
                # Zスコアを計算
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                
                # 外れ値のインデックスを取得（元のデータフレームのインデックス）
                outlier_mask = np.abs(stats.zscore(df[col], nan_policy='omit')) > threshold
                outlier_indices = df[outlier_mask].index.tolist()
                
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        return outliers
    
    def get_outlier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """外れ値のサマリーを作成"""
        iqr_outliers = self.detect_outliers_iqr(df)
        zscore_outliers = self.detect_outliers_zscore(df)
        
        summary_data = []
        numeric_cols = self.get_numeric_columns(df)
        
        for col in numeric_cols:
            iqr_count = len(iqr_outliers.get(col, []))
            zscore_count = len(zscore_outliers.get(col, []))
            
            summary_data.append({
                'column': col,
                'iqr_outliers': iqr_count,
                'zscore_outliers': zscore_count,
                'total_values': df[col].notna().sum()
            })
        
        return pd.DataFrame(summary_data)