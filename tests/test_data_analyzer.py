import pytest
import pandas as pd
import numpy as np

from src.components.data_analyzer import DataAnalyzer


class TestDataAnalyzer:
    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        data = {
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100) * 2 + 5,
            'numeric3': np.random.exponential(2, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'with_missing': np.where(np.random.random(100) > 0.8, np.nan, np.random.randn(100))
        }
        df = pd.DataFrame(data)
        # 外れ値を追加
        df.loc[0, 'numeric1'] = 10
        df.loc[1, 'numeric1'] = -10
        return df
    
    @pytest.fixture
    def analyzer(self):
        """DataAnalyzerのインスタンス"""
        return DataAnalyzer()
    
    def test_get_basic_statistics(self, analyzer, sample_data):
        """基本統計量の計算テスト"""
        stats = analyzer.get_basic_statistics(sample_data)
        
        # 数値列のみ統計が計算されているか
        assert 'numeric1' in stats.columns
        assert 'numeric2' in stats.columns
        assert 'numeric3' in stats.columns
        assert 'with_missing' in stats.columns
        assert 'category' not in stats.columns
        
        # 必要な統計量が含まれているか
        expected_indices = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for idx in expected_indices:
            assert idx in stats.index
    
    def test_analyze_missing_values(self, analyzer, sample_data):
        """欠損値分析のテスト"""
        missing_info = analyzer.analyze_missing_values(sample_data)
        
        assert 'column' in missing_info.columns
        assert 'missing_count' in missing_info.columns
        assert 'missing_percentage' in missing_info.columns
        
        # with_missing列の欠損値が正しく検出されているか
        with_missing_row = missing_info[missing_info['column'] == 'with_missing']
        assert len(with_missing_row) == 1
        assert with_missing_row['missing_count'].iloc[0] > 0
        assert 0 < with_missing_row['missing_percentage'].iloc[0] < 100
    
    def test_calculate_correlations(self, analyzer, sample_data):
        """相関分析のテスト"""
        corr_matrix = analyzer.calculate_correlations(sample_data)
        
        # 数値列のみが含まれているか
        numeric_cols = ['numeric1', 'numeric2', 'numeric3', 'with_missing']
        for col in numeric_cols:
            assert col in corr_matrix.columns
            assert col in corr_matrix.index
        
        # 対角成分が1か
        for col in numeric_cols:
            if col in corr_matrix.columns:  # NaNが多い列は除外される可能性
                assert abs(corr_matrix.loc[col, col] - 1.0) < 1e-10 or pd.isna(corr_matrix.loc[col, col])
    
    def test_analyze_distributions(self, analyzer, sample_data):
        """分布分析のテスト"""
        dist_info = analyzer.analyze_distributions(sample_data)
        
        # 数値列の情報が含まれているか
        assert 'numeric1' in dist_info
        assert 'numeric2' in dist_info
        assert 'numeric3' in dist_info
        
        # 必要な統計量が含まれているか
        for col_info in dist_info.values():
            if isinstance(col_info, dict):  # 数値列の場合
                assert 'skewness' in col_info
                assert 'kurtosis' in col_info
                assert 'unique_values' in col_info
    
    def test_detect_outliers_iqr(self, analyzer, sample_data):
        """IQR法による外れ値検出のテスト"""
        outliers = analyzer.detect_outliers_iqr(sample_data)
        
        # 外れ値が検出されているか
        assert 'numeric1' in outliers
        assert len(outliers['numeric1']) > 0
        
        # 追加した外れ値が検出されているか
        assert 0 in outliers['numeric1'] or 1 in outliers['numeric1']
    
    def test_detect_outliers_zscore(self, analyzer, sample_data):
        """Zスコア法による外れ値検出のテスト"""
        outliers = analyzer.detect_outliers_zscore(sample_data, threshold=3)
        
        # 外れ値が検出されているか
        assert 'numeric1' in outliers
        
        # 閾値を下げるとより多くの外れ値が検出されるか
        outliers_low_threshold = analyzer.detect_outliers_zscore(sample_data, threshold=2)
        assert sum(len(v) for v in outliers_low_threshold.values()) >= sum(len(v) for v in outliers.values())
    
    def test_get_numeric_columns(self, analyzer, sample_data):
        """数値列の抽出テスト"""
        numeric_cols = analyzer.get_numeric_columns(sample_data)
        
        assert 'numeric1' in numeric_cols
        assert 'numeric2' in numeric_cols
        assert 'numeric3' in numeric_cols
        assert 'with_missing' in numeric_cols
        assert 'category' not in numeric_cols
    
    def test_get_categorical_columns(self, analyzer, sample_data):
        """カテゴリ列の抽出テスト"""
        categorical_cols = analyzer.get_categorical_columns(sample_data)
        
        assert 'category' in categorical_cols
        assert 'numeric1' not in categorical_cols
    
    def test_empty_dataframe(self, analyzer):
        """空のDataFrameのテスト"""
        empty_df = pd.DataFrame()
        
        stats = analyzer.get_basic_statistics(empty_df)
        assert stats.empty
        
        missing_info = analyzer.analyze_missing_values(empty_df)
        assert len(missing_info) == 0
        
        outliers = analyzer.detect_outliers_iqr(empty_df)
        assert len(outliers) == 0